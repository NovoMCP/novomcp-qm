"""
CREST Engine — Conformer-Rotamer Ensemble Sampling.

Uses CREST (Conformer-Rotamer Ensemble Sampling Tool) for:
- Conformer search (iMTD-GC algorithm)
- Tautomer enumeration
- Quick conformer ranking

Falls back to RDKit ETKDG if CREST binary is not available.
"""

import asyncio
import logging
import os
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("novomcp-qm.crest")

CREST_BIN = os.getenv("CREST_BIN", "crest")
SCRATCH_DIR = Path(os.getenv("SCRATCH_DIR", "/app/scratch"))


@dataclass
class Conformer:
    energy_hartree: float
    energy_kcal_mol: float
    population: float  # Boltzmann population (0-1)
    xyz: str
    rank: int


@dataclass
class ConformerResult:
    success: bool
    smiles: str
    conformers: list[Conformer]
    n_conformers: int
    energy_range_kcal: Optional[float] = None
    wall_time_seconds: Optional[float] = None
    method: str = "CREST-GFN2"
    error: Optional[str] = None


def is_available() -> bool:
    return shutil.which(CREST_BIN) is not None


async def search_conformers(
    xyz_content: str,
    smiles: str,
    charge: int = 0,
    uhf: int = 0,
    ewin: float = 6.0,
    max_conformers: int = 50,
    quick: bool = False,
) -> ConformerResult:
    """
    Run CREST conformer search.

    Args:
        xyz_content: Initial 3D geometry in XYZ format
        smiles: SMILES for identification
        charge: Molecular charge
        uhf: Number of unpaired electrons
        ewin: Energy window in kcal/mol (conformers within this range kept)
        max_conformers: Maximum conformers to return
        quick: Use quick mode (less thorough but faster)
    """
    if not is_available():
        return _fallback_rdkit(smiles, max_conformers)

    import time

    workdir = tempfile.mkdtemp(dir=SCRATCH_DIR, prefix="crest_")
    try:
        input_file = Path(workdir) / "input.xyz"
        input_file.write_text(xyz_content)

        cmd = [CREST_BIN, str(input_file)]
        cmd.extend(["--chrg", str(charge)])
        cmd.extend(["--ewin", str(ewin)])
        if uhf:
            cmd.extend(["--uhf", str(uhf)])
        if quick:
            cmd.append("--quick")

        # Use GFN2-xTB
        cmd.extend(["--gfn", "2"])
        # Limit threads
        cmd.extend(["-T", str(min(os.cpu_count() or 4, 8))])

        t0 = time.time()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=1800)
        wall_time = time.time() - t0

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")[:500]
            logger.warning(f"CREST failed (rc={proc.returncode}), falling back to RDKit: {err}")
            return _fallback_rdkit(smiles, max_conformers)

        # Parse CREST output ensemble
        ensemble_file = Path(workdir) / "crest_conformers.xyz"
        if not ensemble_file.exists():
            return ConformerResult(
                success=False, smiles=smiles, conformers=[], n_conformers=0,
                error="CREST produced no conformers",
            )

        conformers = _parse_ensemble(ensemble_file.read_text(), max_conformers)

        energy_range = None
        if len(conformers) >= 2:
            energy_range = round(conformers[-1].energy_kcal_mol - conformers[0].energy_kcal_mol, 2)

        return ConformerResult(
            success=True,
            smiles=smiles,
            conformers=conformers,
            n_conformers=len(conformers),
            energy_range_kcal=energy_range,
            wall_time_seconds=round(wall_time, 1),
            method="CREST-GFN2",
        )

    except asyncio.TimeoutError:
        logger.warning("CREST timed out, falling back to RDKit")
        return _fallback_rdkit(smiles, max_conformers)
    except Exception as e:
        logger.error(f"CREST error: {e}")
        return _fallback_rdkit(smiles, max_conformers)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _parse_ensemble(content: str, max_conf: int) -> list[Conformer]:
    """Parse multi-structure XYZ file from CREST."""
    conformers = []
    blocks = content.strip().split("\n")

    i = 0
    rank = 1
    while i < len(blocks) and rank <= max_conf:
        try:
            n_atoms = int(blocks[i].strip())
        except (ValueError, IndexError):
            i += 1
            continue

        comment = blocks[i + 1].strip()
        # CREST puts energy in comment line
        energy_hartree = None
        for part in comment.split():
            try:
                energy_hartree = float(part)
                break
            except ValueError:
                continue

        xyz_lines = blocks[i : i + n_atoms + 2]
        xyz_text = "\n".join(xyz_lines)

        if energy_hartree is not None:
            conformers.append(Conformer(
                energy_hartree=energy_hartree,
                energy_kcal_mol=round(energy_hartree * 627.509, 2),
                population=0.0,  # Calculated below
                xyz=xyz_text,
                rank=rank,
            ))
            rank += 1

        i += n_atoms + 2

    # Calculate Boltzmann populations
    if conformers:
        import math
        kbt = 0.001987 * 298.15  # kcal/mol at 298K
        min_e = conformers[0].energy_kcal_mol
        weights = []
        for c in conformers:
            de = c.energy_kcal_mol - min_e
            weights.append(math.exp(-de / kbt))
        total = sum(weights)
        for c, w in zip(conformers, weights):
            c.population = round(w / total, 4)

    return conformers


def _fallback_rdkit(smiles: str, max_conf: int) -> ConformerResult:
    """RDKit ETKDG conformer search as fallback."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ConformerResult(
                success=False, smiles=smiles, conformers=[], n_conformers=0,
                error="Invalid SMILES",
            )

        mol = Chem.AddHs(mol)
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_confs = min(max(50, n_rot * 10), max_conf * 3)

        params = AllChem.ETKDGv3()
        params.numThreads = min(os.cpu_count() or 4, 4)
        params.pruneRmsThresh = 0.5
        params.randomSeed = 42

        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        if not conf_ids:
            return ConformerResult(
                success=False, smiles=smiles, conformers=[], n_conformers=0,
                error="RDKit ETKDG failed to generate conformers",
            )

        # Minimize with MMFF
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500, numThreads=4)

        conformers = []
        for idx, (converged, energy) in enumerate(results):
            if len(conformers) >= max_conf:
                break
            conf = mol.GetConformer(idx)
            xyz_lines = [str(mol.GetNumAtoms()), f"Energy: {energy:.6f} kcal/mol"]
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                pos = conf.GetAtomPosition(atom_idx)
                xyz_lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")

            conformers.append(Conformer(
                energy_hartree=energy / 627.509,
                energy_kcal_mol=round(energy, 2),
                population=0.0,
                xyz="\n".join(xyz_lines),
                rank=idx + 1,
            ))

        # Sort by energy and assign populations
        conformers.sort(key=lambda c: c.energy_kcal_mol)
        for i, c in enumerate(conformers):
            c.rank = i + 1

        if conformers:
            import math
            kbt = 0.001987 * 298.15
            min_e = conformers[0].energy_kcal_mol
            weights = [math.exp(-(c.energy_kcal_mol - min_e) / kbt) for c in conformers]
            total = sum(weights)
            for c, w in zip(conformers, weights):
                c.population = round(w / total, 4)

        conformers = conformers[:max_conf]

        return ConformerResult(
            success=True,
            smiles=smiles,
            conformers=conformers,
            n_conformers=len(conformers),
            energy_range_kcal=round(conformers[-1].energy_kcal_mol - conformers[0].energy_kcal_mol, 2) if len(conformers) >= 2 else 0.0,
            method="RDKit-ETKDG-MMFF",
        )

    except Exception as e:
        logger.error(f"RDKit conformer fallback failed: {e}")
        return ConformerResult(
            success=False, smiles=smiles, conformers=[], n_conformers=0,
            error=str(e),
        )
