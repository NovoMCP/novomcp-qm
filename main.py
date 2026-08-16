"""
NovoMCP QM Engine Service

Semi-empirical quantum chemistry: xTB, CREST, strain-corrected docking.
Exposes: run_qm_calculation, run_conformer_search, dock_with_strain
"""

import asyncio
import json
import os
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from app.engines import xtb, crest, alchemi_conformers
from app.engines.smiles_to_xyz import smiles_to_xyz, get_charge

logging.basicConfig(
    format="[NovoMCP] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("novomcp-qm")

PORT = int(os.getenv("PORT", "8031"))
API_KEY = os.getenv("QM_API_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", "")

# Redis client (initialized on startup)
redis_client = None

# Concurrency limiter for CREST (CPU-bound, 4 vCPU)
conformer_semaphore = asyncio.Semaphore(2)

app = FastAPI(
    title="NovoMCP QM Engine",
    description="Semi-empirical quantum chemistry: xTB calculations, CREST conformer search, strain-corrected docking",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth ---

def _check_key(key: Optional[str]):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# --- Request/Response Models ---

class QmRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    calculation: str = Field("optimize", description="Type: energy, optimize, solvation")
    charge: int = Field(0, description="Molecular charge")
    uhf: int = Field(0, description="Number of unpaired electrons (0=singlet, 1=doublet for radicals/ions, etc.)")
    solvent: Optional[str] = Field(None, description="Solvent for ALPB model (e.g., water, dmso, methanol)")
    xyz_input: Optional[str] = Field(None, description="Pre-optimized XYZ geometry. When provided, bypasses SMILES-to-3D conversion and uses these coordinates directly. Use this to pass an optimized neutral geometry into a cation/anion calculation for consistent thermodynamic cycles.")

class QmResponse(BaseModel):
    smiles: str
    energy_hartree: Optional[float]
    energy_kcal_mol: Optional[float]
    solvation_energy_kcal_mol: Optional[float] = None
    homo_ev: Optional[float] = None
    lumo_ev: Optional[float] = None
    gap_ev: Optional[float] = None
    dipole_debye: Optional[float] = None
    partial_charges: Optional[list[float]] = None
    optimized_xyz: Optional[str] = None
    method: str
    wall_time_seconds: Optional[float]

class ExcitedStateRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    charge: int = Field(0, description="Molecular charge")
    num_states: int = Field(10, description="Number of excited states to compute", ge=1, le=50)
    xyz_input: Optional[str] = Field(None, description="Pre-optimized XYZ geometry (recommended for accurate excited states)")

class ExcitedStateResponse(BaseModel):
    smiles: str
    excited_states: list[dict]
    s1_energy_ev: Optional[float] = None
    t1_energy_ev: Optional[float] = None
    singlet_triplet_gap_ev: Optional[float] = None
    emission_wavelength_nm: Optional[float] = None
    phosphorescence_wavelength_nm: Optional[float] = None
    n_states: int
    ground_state_energy_hartree: Optional[float] = None
    method: str
    wall_time_seconds: Optional[float]

class ConformerRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    max_conformers: int = Field(20, description="Maximum conformers to return", ge=1, le=100)
    energy_window: float = Field(6.0, description="Energy window in kcal/mol")
    quick: bool = Field(False, description="Use quick mode (faster, less thorough)")
    engine: str = Field("crest", description="Search engine: 'crest' (CREST/GFN2-xTB, default) or 'alchemi' (ETKDG + NVIDIA ALCHEMI Toolkit batched MACE relaxation on GPU; falls back to CREST when a GPU/toolkit isn't present)")


async def _route_conformer_search(engine: str, xyz: str, smiles: str, charge: int,
                                  ewin: float, max_conformers: int, quick: bool):
    """Dispatch to the requested engine; 'alchemi' falls back to CREST when the
    GPU/toolkit isn't available. The result's `method` field reports which ran."""
    if engine == "alchemi" and alchemi_conformers.is_available():
        return await alchemi_conformers.search_conformers_alchemi(
            xyz_content=xyz, smiles=smiles, charge=charge, ewin=ewin,
            max_conformers=max_conformers, quick=quick)
    return await crest.search_conformers(
        xyz_content=xyz, smiles=smiles, charge=charge, ewin=ewin,
        max_conformers=max_conformers, quick=quick)

class ConformerResponse(BaseModel):
    smiles: str
    n_conformers: int
    energy_range_kcal: Optional[float]
    method: str
    wall_time_seconds: Optional[float]
    conformers: list[dict]

class StrainRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the ligand")
    docked_xyz: Optional[str] = Field(None, description="XYZ of docked pose (if available)")
    charge: int = Field(0, description="Molecular charge")

class StrainResponse(BaseModel):
    smiles: str
    strain_energy_kcal_mol: float
    interpretation: str
    method: str
    wall_time_seconds: Optional[float]

class HessianRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string (used for charge detection and response labeling)")
    charge: int = Field(0, description="Molecular charge")
    uhf: int = Field(0, description="Number of unpaired electrons")
    solvent: Optional[str] = Field(None, description="Solvent for ALPB model")
    temperature: float = Field(298.15, description="Temperature in K for thermochemistry", gt=0)
    xyz_input: Optional[str] = Field(None, description="Pre-optimized XYZ geometry. Recommended: pass the optimized geometry from a prior /api/qm-calculate call for meaningful thermochemistry at a true minimum.")
    optimize_first: bool = Field(False, description="If true, optimize geometry before Hessian (--ohess). If false, run Hessian at given geometry (--hess).")

class HessianResponse(BaseModel):
    smiles: str
    energy_hartree: Optional[float]
    energy_kcal_mol: Optional[float]
    zpe_kcal_mol: Optional[float] = None
    enthalpy_correction_kcal_mol: Optional[float] = None
    gibbs_correction_kcal_mol: Optional[float] = None
    entropy_cal_mol_k: Optional[float] = None
    temperature_k: float
    frequencies_cm1: Optional[list[float]] = None
    imaginary_frequencies_cm1: list[float] = []
    n_imaginary: int = 0
    is_true_minimum: bool = True
    optimized_xyz: Optional[str] = None
    method: str
    wall_time_seconds: Optional[float]


# --- Redis Job Tracking ---

async def _init_redis():
    """Initialize Redis connection for job tracking."""
    global redis_client
    if not REDIS_URL:
        logger.warning("No REDIS_URL configured — conformer search will run synchronously")
        return
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connected for async job tracking")
    except Exception as e:
        logger.warning(f"Redis not available — conformer search will run synchronously: {e}")
        redis_client = None


async def update_job_status(job_id: str, status: str, progress: dict):
    """Update job progress in Redis (novomcp compatible format)."""
    if not redis_client:
        return
    try:
        key = f"novomcp:job:{job_id}"
        await redis_client.hset(key, mapping={
            "job_id": job_id,
            "status": status,
            "progress": json.dumps(progress),
            "last_updated": datetime.utcnow().isoformat(),
        })
        await redis_client.expire(key, 86400)  # 24 hours
        logger.info(f"Job {job_id}: {progress.get('percentage', 0)}% - {progress.get('message', '')}")
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")


async def complete_job(job_id: str, result: dict):
    """Mark job as completed with results."""
    if not redis_client:
        return
    try:
        now = datetime.utcnow().isoformat()
        key = f"novomcp:job:{job_id}"
        await redis_client.hset(key, mapping={
            "job_id": job_id,
            "status": "completed",
            "completed_at": now,
            "result": json.dumps(result),
            "progress": json.dumps({"percentage": 100, "message": "Completed", "step": "completed"}),
            "last_updated": now,
        })
        await redis_client.expire(key, 604800)  # 7 days

        # Legacy keys for novomcp compatibility
        await redis_client.set(f"novomcp:job_result:{job_id}", json.dumps(result), ex=604800)
    except Exception as e:
        logger.error(f"Failed to complete job: {e}")


async def fail_job(job_id: str, error: str):
    """Mark job as failed."""
    if not redis_client:
        return
    try:
        now = datetime.utcnow().isoformat()
        key = f"novomcp:job:{job_id}"
        await redis_client.hset(key, mapping={
            "job_id": job_id,
            "status": "failed",
            "completed_at": now,
            "error": error,
            "progress": json.dumps({"percentage": 0, "message": f"Failed: {error}", "step": "failed"}),
            "last_updated": now,
        })
        await redis_client.expire(key, 86400)
    except Exception as e:
        logger.error(f"Failed to mark job as failed: {e}")


# --- Startup ---

@app.on_event("startup")
async def startup_event():
    logger.info("Starting NovoMCP QM Engine...")
    xtb_ok = xtb.is_available()
    crest_ok = crest.is_available()
    logger.info(f"xTB binary: {'found' if xtb_ok else 'NOT FOUND'}")
    logger.info(f"CREST binary: {'found' if crest_ok else 'NOT FOUND (RDKit fallback)'}")
    await _init_redis()


# --- Health ---

@app.get("/health")
async def health():
    xtb_ok = xtb.is_available()
    crest_ok = crest.is_available()
    status = "healthy" if xtb_ok else "degraded"

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=200 if xtb_ok else 503,
        content={
            "status": status,
            "service": "novomcp-qm",
            "version": "1.1.0",
            "port": PORT,
            "engines": {
                "xtb": {"available": xtb_ok, "method": "GFN2-xTB", "hessian": xtb_ok},
                "stda": {"available": xtb.is_stda_available(), "method": "sTDA-xTB"},
                "crest": {"available": crest_ok, "fallback": "RDKit-ETKDG"},
            },
            "async_jobs": redis_client is not None,
        },
    )


@app.get("/")
async def root():
    return {"service": "novomcp-qm", "version": "1.1.0"}


# --- QM Calculation ---

@app.post("/api/qm-calculate", response_model=QmResponse)
async def qm_calculate(req: QmRequest, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)

    # Use pre-optimized geometry if provided, otherwise generate from SMILES
    if req.xyz_input:
        xyz = req.xyz_input
    else:
        xyz = smiles_to_xyz(req.smiles)
        if not xyz:
            raise HTTPException(status_code=422, detail=f"Could not generate 3D coordinates for: {req.smiles}")

    charge = req.charge or get_charge(req.smiles)

    if req.calculation == "energy":
        result = await xtb.run_single_point(xyz, charge=charge, uhf=req.uhf, solvent=req.solvent)
    elif req.calculation == "optimize":
        result = await xtb.run_optimization(xyz, charge=charge, uhf=req.uhf, solvent=req.solvent)
    elif req.calculation == "solvation":
        # Run in gas phase and with solvent, return difference
        gas = await xtb.run_single_point(xyz, charge=charge, uhf=req.uhf)
        solv = await xtb.run_single_point(xyz, charge=charge, uhf=req.uhf, solvent=req.solvent or "water")

        if gas.success and solv.success and gas.energy_hartree and solv.energy_hartree:
            solv_energy = (solv.energy_hartree - gas.energy_hartree) * 627.509
            result = solv
            result.solvation_energy_kcal_mol = round(solv_energy, 2)
        else:
            raise HTTPException(status_code=500, detail="Solvation calculation failed")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown calculation type: {req.calculation}")

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Calculation failed")

    return QmResponse(
        smiles=req.smiles,
        energy_hartree=result.energy_hartree,
        energy_kcal_mol=result.energy_kcal_mol,
        solvation_energy_kcal_mol=result.solvation_energy_kcal_mol,
        homo_ev=result.homo_ev,
        lumo_ev=result.lumo_ev,
        gap_ev=result.gap_ev,
        dipole_debye=result.dipole_debye,
        partial_charges=result.partial_charges,
        optimized_xyz=result.optimized_xyz,
        method=result.method,
        wall_time_seconds=result.wall_time_seconds,
    )


# --- Hessian / Frequency Calculation ---

@app.post("/api/qm-hessian", response_model=HessianResponse)
async def qm_hessian(req: HessianRequest, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)

    if req.xyz_input:
        xyz = req.xyz_input
    else:
        xyz = smiles_to_xyz(req.smiles)
        if not xyz:
            raise HTTPException(status_code=422, detail=f"Could not generate 3D coordinates for: {req.smiles}")

    charge = req.charge or get_charge(req.smiles)

    result = await xtb.run_hessian(
        xyz_content=xyz,
        charge=charge,
        uhf=req.uhf,
        solvent=req.solvent,
        temperature=req.temperature,
        optimize_first=req.optimize_first,
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Hessian calculation failed")

    return HessianResponse(
        smiles=req.smiles,
        energy_hartree=result.energy_hartree,
        energy_kcal_mol=result.energy_kcal_mol,
        zpe_kcal_mol=result.zpe_kcal_mol,
        enthalpy_correction_kcal_mol=result.enthalpy_correction_kcal_mol,
        gibbs_correction_kcal_mol=result.gibbs_correction_kcal_mol,
        entropy_cal_mol_k=result.entropy_cal_mol_k,
        temperature_k=result.temperature_k,
        frequencies_cm1=result.frequencies_cm1,
        imaginary_frequencies_cm1=result.imaginary_frequencies_cm1 or [],
        n_imaginary=result.n_imaginary,
        is_true_minimum=result.is_true_minimum,
        optimized_xyz=result.optimized_xyz,
        method=result.method,
        wall_time_seconds=result.wall_time_seconds,
    )


# --- Excited States (sTDA-xTB) ---

@app.post("/api/qm-excited-states", response_model=ExcitedStateResponse)
async def qm_excited_states(req: ExcitedStateRequest, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)

    if req.xyz_input:
        xyz = req.xyz_input
    else:
        xyz = smiles_to_xyz(req.smiles)
        if not xyz:
            raise HTTPException(status_code=422, detail=f"Could not generate 3D coordinates for: {req.smiles}")

    charge = req.charge or get_charge(req.smiles)

    result = await xtb.run_excited_states(
        xyz_content=xyz,
        charge=charge,
        num_states=req.num_states,
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Excited state calculation failed")

    return ExcitedStateResponse(
        smiles=req.smiles,
        excited_states=result.excited_states or [],
        s1_energy_ev=result.s1_energy_ev,
        t1_energy_ev=result.t1_energy_ev,
        singlet_triplet_gap_ev=result.singlet_triplet_gap_ev,
        emission_wavelength_nm=result.emission_wavelength_nm,
        phosphorescence_wavelength_nm=result.phosphorescence_wavelength_nm,
        n_states=result.n_states,
        ground_state_energy_hartree=result.ground_state_energy_hartree,
        method=result.method,
        wall_time_seconds=result.wall_time_seconds,
    )


# --- Conformer Search (Async Job) ---

async def _run_conformer_job(job_id: str, smiles: str, xyz: str, charge: int,
                             energy_window: float, max_conformers: int, quick: bool,
                             engine: str = "crest"):
    """Background task for conformer search."""
    async with conformer_semaphore:
        try:
            await update_job_status(job_id, "running", {
                "percentage": 10,
                "message": "Starting conformer search",
                "step": "starting",
            })

            result = await _route_conformer_search(
                engine, xyz, smiles, charge, energy_window, max_conformers, quick,
            )

            if not result.success:
                await fail_job(job_id, result.error or "Conformer search failed")
                return

            await update_job_status(job_id, "processing", {
                "percentage": 90,
                "message": f"Found {result.n_conformers} conformers, preparing results",
                "step": "formatting",
            })

            conf_dicts = []
            for c in result.conformers:
                conf_dicts.append({
                    "rank": c.rank,
                    "energy_kcal_mol": c.energy_kcal_mol,
                    "population": c.population,
                    "xyz": c.xyz if len(result.conformers) <= 10 else None,
                })

            job_result = {
                "smiles": result.smiles,
                "n_conformers": result.n_conformers,
                "energy_range_kcal": result.energy_range_kcal,
                "method": result.method,
                "wall_time_seconds": result.wall_time_seconds,
                "conformers": conf_dicts,
            }

            await complete_job(job_id, job_result)

        except Exception as e:
            logger.exception(f"Conformer job {job_id} failed: {e}")
            await fail_job(job_id, str(e))


@app.post("/api/conformer-search")
async def conformer_search(req: ConformerRequest, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)

    xyz = smiles_to_xyz(req.smiles)
    if not xyz:
        raise HTTPException(status_code=422, detail=f"Could not generate 3D coordinates for: {req.smiles}")

    charge = get_charge(req.smiles)

    # If Redis is available, run as async job
    if redis_client is not None:
        job_id = f"qm_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{abs(hash(req.smiles)) % 100000:05d}"

        await update_job_status(job_id, "queued", {
            "percentage": 0,
            "message": "Conformer search queued",
            "step": "queued",
        })

        asyncio.create_task(_run_conformer_job(
            job_id, req.smiles, xyz, charge,
            req.energy_window, req.max_conformers, req.quick, req.engine,
        ))

        if req.engine == "alchemi" and alchemi_conformers.is_available():
            method = "ALCHEMI-MACE-MPA-0"
        else:
            method = "CREST" if crest.is_available() else "RDKit-ETKDG"
        return {
            "job_id": job_id,
            "status": "submitted",
            "service": "novomcp-qm",
            "smiles": req.smiles,
            "max_conformers": req.max_conformers,
            "method": method,
            "poll_url": f"/status/{job_id}",
        }

    # Fallback: synchronous execution (no Redis)
    result = await _route_conformer_search(
        req.engine, xyz, req.smiles, charge,
        req.energy_window, req.max_conformers, req.quick,
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Conformer search failed")

    conf_dicts = []
    for c in result.conformers:
        conf_dicts.append({
            "rank": c.rank,
            "energy_kcal_mol": c.energy_kcal_mol,
            "population": c.population,
            "xyz": c.xyz if len(result.conformers) <= 10 else None,
        })

    return ConformerResponse(
        smiles=result.smiles,
        n_conformers=result.n_conformers,
        energy_range_kcal=result.energy_range_kcal,
        method=result.method,
        wall_time_seconds=result.wall_time_seconds,
        conformers=conf_dicts,
    )


# --- Job Status & Results ---

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Poll job progress."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Job tracking not available (no Redis)")

    key = f"novomcp:job:{job_id}"
    data = await redis_client.hgetall(key)
    if not data:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    progress = json.loads(data.get("progress", "{}"))
    resp = {
        "job_id": job_id,
        "status": data.get("status"),
        "progress": progress.get("percentage", 0),
        "message": progress.get("message"),
        "step": progress.get("step"),
        "last_updated": data.get("last_updated"),
    }

    if data.get("status") == "failed":
        resp["error"] = data.get("error")

    return resp


@app.get("/results/{job_id}")
async def get_job_results(job_id: str):
    """Get completed job results."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Job tracking not available (no Redis)")

    key = f"novomcp:job:{job_id}"
    data = await redis_client.hgetall(key)
    if not data:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    status = data.get("status")
    if status == "completed":
        return {
            "job_id": job_id,
            "status": "completed",
            "result": json.loads(data.get("result", "{}")),
        }
    elif status == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "error": data.get("error"),
        }
    else:
        progress = json.loads(data.get("progress", "{}"))
        return {
            "job_id": job_id,
            "status": status,
            "progress": progress,
        }


# --- Strain-Corrected Docking ---

@app.post("/api/strain-energy", response_model=StrainResponse)
async def strain_energy(req: StrainRequest, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)

    if req.docked_xyz:
        xyz = req.docked_xyz
    else:
        xyz = smiles_to_xyz(req.smiles)
        if not xyz:
            raise HTTPException(status_code=422, detail=f"Could not generate 3D coordinates for: {req.smiles}")

    charge = req.charge or get_charge(req.smiles)

    result = await xtb.run_strain_energy(xyz, charge=charge)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Strain calculation failed")

    strain = result.energy_kcal_mol or 0.0
    if strain < 2.0:
        interpretation = f"Strain energy {strain:.1f} kcal/mol — minimal strain, docked pose is geometrically reasonable."
    elif strain < 5.0:
        interpretation = f"Strain energy {strain:.1f} kcal/mol — moderate strain, acceptable for most drug-like molecules."
    elif strain < 10.0:
        interpretation = f"Strain energy {strain:.1f} kcal/mol — significant strain, pose may be unreliable. Consider re-docking with flexible residues."
    else:
        interpretation = f"Strain energy {strain:.1f} kcal/mol — severe strain, docked pose is likely an artifact. Discard or re-dock."

    return StrainResponse(
        smiles=req.smiles,
        strain_energy_kcal_mol=round(strain, 2),
        interpretation=interpretation,
        method=result.method,
        wall_time_seconds=result.wall_time_seconds,
    )


if __name__ == "__main__":
    logger.info(f"Starting NovoMCP QM Engine on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
