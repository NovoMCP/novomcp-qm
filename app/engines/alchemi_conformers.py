"""
ALCHEMI conformer engine — RDKit ETKDG generation + NVIDIA ALCHEMI Toolkit
batched MLIP (MACE-MPA-0) relaxation + energy ranking, on GPU.

engine="alchemi": generate a diverse ensemble with ETKDG, relax the *whole*
ensemble in one batched FIRE pass on the GPU, then rank/deduplicate by MLIP
energy. Faster than CREST metadynamics; far better energies than the MMFF
fallback. The batch is homogeneous (one molecule, many geometries), which the
ALCHEMI batched kernels handle especially well.

Requires nvalchemi-toolkit + mace-torch + a CUDA GPU. Absent those, is_available()
is False and the caller falls back to CREST/RDKit.
"""

import asyncio
import logging
import math
import os
import time

import numpy as np

from app.engines.crest import Conformer, ConformerResult  # reuse the shared shapes

logger = logging.getLogger("novomcp-qm.alchemi")

EV_TO_KCAL = 23.0609
HARTREE_PER_EV = 1.0 / 27.2114
_RMSD_DEDUP_ANG = 0.125  # naive per-atom RMSD threshold for near-duplicate conformers

_model = None  # cached MACEWrapper


def is_available() -> bool:
    """True when the ALCHEMI Toolkit + mace + a CUDA GPU are present."""
    try:
        import torch
        import nvalchemi  # noqa: F401
        from nvalchemi.models.mace import MACEWrapper  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_model():
    global _model
    if _model is None:
        from nvalchemi.models.mace import MACEWrapper
        from mace.calculators.foundations_models import mace_mp
        raw = mace_mp(model="medium-mpa-0", device="cuda", default_dtype="float32").models[0]
        _model = MACEWrapper(raw).to("cuda").eval()
        logger.info("ALCHEMI MACE-MPA-0 model loaded on GPU")
    return _model


def _generate_etkdg(smiles: str, max_conformers: int):
    """Generate a diverse ETKDG ensemble. Returns (species, [positions...]) or None."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_confs = min(max(50, n_rot * 10), max_conformers * 3)

    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = 0.5
    params.randomSeed = 42
    params.numThreads = min(os.cpu_count() or 4, 4)

    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if not ids:
        return None

    species = [a.GetAtomicNum() for a in mol.GetAtoms()]
    n = mol.GetNumAtoms()
    positions = []
    for cid in ids:
        conf = mol.GetConformer(cid)
        positions.append(np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
             for i in range(n)], dtype=np.float32))
    return species, positions


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def _batch_relax(species, positions_list, fmax: float, max_steps: int):
    """Relax every conformer (same molecule, different geometry) in one batched
    FIRE pass on the GPU. Returns (energies_eV, relaxed_positions). Synchronous —
    call via run_in_executor. Mirrors the validated novomcp-nnp recipe."""
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import FIRE, ConvergenceHook
    try:
        import torch._dynamo as _dynamo
        _dynamo.config.disable = True
    except Exception:
        pass

    model = _get_model()
    hooks = model.make_neighbor_hooks()
    n = len(species)
    znums = torch.tensor(list(species), dtype=torch.long)

    datas = []
    for pos in positions_list:
        datas.append(AtomicData(
            atomic_numbers=znums.clone(),
            positions=torch.tensor(np.asarray(pos), dtype=torch.float32),
            forces=torch.zeros((n, 3), dtype=torch.float32),   # pre-allocated for FIRE step-0
            energy=torch.zeros((1, 1), dtype=torch.float32),
        ))
    batch = Batch.from_data_list(datas, device="cuda")

    opt = FIRE(model=model, dt=0.1, n_steps=max_steps,
               convergence_hook=ConvergenceHook.from_fmax(fmax), hooks=hooks)
    opt.compile_step = False  # eager: toolkit's host-side auto neighbor method can't run under compile
    with opt:
        result = opt.run(batch)

    energies, relaxed = [], []
    for i in range(len(positions_list)):
        ad = result.get_data(i)
        energies.append(float(ad.energy.reshape(-1)[0]))
        relaxed.append(ad.positions.detach().cpu().numpy())
    return energies, relaxed


async def search_conformers_alchemi(
    xyz_content: str,
    smiles: str,
    charge: int = 0,
    uhf: int = 0,
    ewin: float = 6.0,
    max_conformers: int = 50,
    quick: bool = False,
) -> ConformerResult:
    """ETKDG ensemble → batched MACE-MPA-0 FIRE relaxation on GPU → energy ranking.

    `xyz_content` is ignored (the ensemble is generated from SMILES via ETKDG).
    Returns the same ConformerResult shape as the CREST engine.
    """
    gen = _generate_etkdg(smiles, max_conformers)
    if gen is None:
        return ConformerResult(success=False, smiles=smiles, conformers=[], n_conformers=0,
                               error="ETKDG generation failed (invalid SMILES or no conformers)")
    species, positions_list = gen

    t0 = time.time()
    try:
        loop = asyncio.get_event_loop()
        energies_ev, relaxed = await loop.run_in_executor(
            None, _batch_relax, species, positions_list, 0.05, 60 if quick else 200)
    except Exception as e:
        logger.error(f"ALCHEMI conformer relaxation failed: {e}")
        return ConformerResult(success=False, smiles=smiles, conformers=[], n_conformers=0, error=str(e))
    wall = time.time() - t0

    # Rank by MLIP energy, then keep within the energy window, deduping near-identical geometries.
    from ase.data import chemical_symbols
    order = sorted(range(len(energies_ev)), key=lambda i: energies_ev[i])
    e_min = energies_ev[order[0]]
    kept = []
    for idx in order:
        if (energies_ev[idx] - e_min) * EV_TO_KCAL > ewin:
            break
        if any(_rmsd(relaxed[idx], relaxed[k]) < _RMSD_DEDUP_ANG for k in kept):
            continue
        kept.append(idx)
        if len(kept) >= max_conformers:
            break

    conformers = []
    for rank, idx in enumerate(kept, 1):
        pos = relaxed[idx]
        xyz_lines = [str(len(species)), f"Energy: {energies_ev[idx]:.6f} eV"]
        for j in range(len(species)):
            x, y, z = pos[j]
            xyz_lines.append(f"{chemical_symbols[species[j]]} {x:.6f} {y:.6f} {z:.6f}")
        conformers.append(Conformer(
            energy_hartree=energies_ev[idx] * HARTREE_PER_EV,
            energy_kcal_mol=round(energies_ev[idx] * EV_TO_KCAL, 2),
            population=0.0,
            xyz="\n".join(xyz_lines),
            rank=rank,
        ))

    if conformers:
        kbt = 0.001987 * 298.15
        min_e = conformers[0].energy_kcal_mol
        weights = [math.exp(-(c.energy_kcal_mol - min_e) / kbt) for c in conformers]
        total = sum(weights)
        for c, w in zip(conformers, weights):
            c.population = round(w / total, 4)

    energy_range = (round(conformers[-1].energy_kcal_mol - conformers[0].energy_kcal_mol, 2)
                    if len(conformers) >= 2 else 0.0)
    return ConformerResult(
        success=True, smiles=smiles, conformers=conformers, n_conformers=len(conformers),
        energy_range_kcal=energy_range, wall_time_seconds=round(wall, 1), method="ALCHEMI-MACE-MPA-0",
    )
