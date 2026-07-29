"""
SMILES to 3D XYZ coordinate conversion using RDKit.
"""

import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger("novomcp-qm.smiles_to_xyz")


def smiles_to_xyz(smiles: str, optimize: bool = True) -> Optional[str]:
    """Convert SMILES to XYZ string with 3D coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)

    if status != 0:
        # Fallback 1: try without random seed
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    if status != 0:
        # Fallback 2: random coordinate generation (works for tiny molecules
        # like HCN/HNC/H2O/CO2 where ETKDG fails due to insufficient distance
        # constraints — xTB will optimize to the correct geometry anyway).
        params2 = AllChem.ETKDGv3()
        params2.useRandomCoords = True
        params2.maxAttempts = 50
        status = AllChem.EmbedMolecule(mol, params2)

    if status != 0:
        return None

    if optimize:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    lines = [str(n_atoms), f"Generated from {smiles}"]
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")

    return "\n".join(lines)


def get_charge(smiles: str) -> int:
    """Determine formal charge of a molecule from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return Chem.GetFormalCharge(mol)
