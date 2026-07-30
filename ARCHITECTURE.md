# NovoMCP QM Engine Service

Semi-empirical quantum chemistry service for the NovoMCP engine. Provides xTB calculations, CREST conformer search, and strain-corrected docking scores.

**Port:** 8031 | **Ingress:** Internal only | **Contacted by:** the NovoMCP engine

---

## Tools

### run_qm_calculation
Run GFN2-xTB semi-empirical quantum mechanics on a molecule. Supports single-point energy, geometry optimization, and solvation free energy (ALPB model).

- **Engine:** xTB 6.7.1 (Grimme group, open source)
- **Method:** GFN2-xTB — parametrized for all elements up to radon
- **Capabilities:**
  - `energy` — single-point electronic energy (Hartree)
  - `optimize` — geometry optimization to local minimum
  - `solvation` — solvation free energy via ALPB implicit solvent (water, DMSO, methanol, etc.)
- **Returns:** Energy (Hartree + kcal/mol), HOMO-LUMO gap (eV), dipole moment (Debye), optimized XYZ geometry
- **Typical runtime:** 1-10 seconds per molecule (depends on size)
- **Credit cost:** 20

### run_conformer_search
Generate a conformer ensemble ranked by energy with Boltzmann populations. Essential before docking — ensures the bioactive conformer is sampled.

- **Primary engine:** CREST 3.0.2 (iMTD-GC algorithm with GFN2-xTB)
- **Fallback engine:** RDKit ETKDG v3 + MMFF optimization
- **Output:** Ranked conformers with energies (kcal/mol) and Boltzmann populations (0-1)
- **Parameters:**
  - `max_conformers` — limit returned conformers (default 20, max 100)
  - `energy_window` — keep conformers within this range of global minimum (default 6 kcal/mol)
  - `quick` — fast mode for larger molecules
- **Typical runtime:** CREST: 30-300 seconds, RDKit fallback: 1-5 seconds
- **Credit cost:** 25

### dock_with_strain
Calculate internal strain energy of a docked ligand pose. Strain = E(docked_pose) - E(optimized). High strain indicates the docking score may be an artifact.

- **Engine:** xTB 6.7.1
- **Method:** Single-point on docked geometry, then full optimization, compute energy difference
- **Interpretation:**
  - < 2 kcal/mol: minimal strain, pose is geometrically reasonable
  - 2-5 kcal/mol: moderate strain, acceptable for drug-like molecules
  - 5-10 kcal/mol: significant strain, pose may be unreliable
  - > 10 kcal/mol: severe strain, likely an artifact — discard
- **Use case:** Run after `dock_molecules` to filter false positives before MD simulation
- **Typical runtime:** 5-30 seconds (two xTB calculations)
- **Credit cost:** 15

---

## Architecture

```
the NovoMCP engine (tools.py)
  → _call_service("novomcp-qm", "/api/qm-calculate", {...})
  → novomcp-qm:8031  (the address the engine is configured with)
```

### Request flow
1. the NovoMCP engine receives MCP tool call (e.g., `run_qm_calculation`)
2. Executor method sends HTTP POST to novomcp-qm internal endpoint
3. novomcp-qm converts SMILES → 3D XYZ (RDKit ETKDG + MMFF)
4. Runs xTB/CREST as subprocess in temp directory
5. Parses stdout for energies, orbital data, geometries
6. Returns structured JSON response
7. the NovoMCP engine adds interpretation and returns to client

### SMILES → XYZ conversion
All tools accept SMILES input. 3D coordinates are generated via:
1. RDKit `AllChem.EmbedMolecule()` with ETKDG v3
2. MMFF force field optimization (500 iterations)
3. Output as XYZ format for xTB/CREST input

### Scratch directory
Each calculation runs in an isolated temp directory under `/app/scratch/`, cleaned up after completion. This prevents conflicts between concurrent calculations.

### Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/api/qm-calculate` | POST | xTB energy, optimization, or solvation |
| `/api/conformer-search` | POST | CREST/RDKit conformer ensemble |
| `/api/strain-energy` | POST | Strain energy for docked poses |
| `/health` | GET | Health check (reports xTB + CREST availability) |

### Container specs
- **Image:** python:3.11-slim-bullseye + xTB 6.7.1 + CREST 3.0.2 + RDKit
- **CPU/Memory:** 4 vCPU / 8Gi
- **Min/Max replicas:** 1 / 10
- **OMP_NUM_THREADS:** 4 (xTB parallelism)
- **OMP_STACKSIZE:** 1G

---

## Binaries

| Binary | Version | Source | Size | Linking |
|---|---|---|---|---|
| xTB | 6.7.1 | github.com/grimme-lab/xtb/releases | ~50 MB | Dynamic (needs libgomp, libopenblas) |
| CREST | 3.0.2 | github.com/crest-lab/crest/releases | ~9 MB | Static (no dependencies) |

Both installed at build time from GitHub releases. xTB in `/opt/xtb/bin/`, CREST in `/usr/local/bin/`.

---

## Files

```
novomcp-qm/
├── main.py                      # FastAPI app, endpoints, request/response models
├── app/
│   └── engines/
│       ├── xtb.py               # xTB wrapper (energy, optimize, solvation, strain)
│       ├── crest.py             # CREST wrapper + RDKit ETKDG fallback
│       └── smiles_to_xyz.py     # SMILES → 3D coordinate generation
├── Dockerfile                   # Service image with xTB + CREST binaries
├── requirements.txt
└── .github/workflows/deploy-azure.yml
```

---

## Relationship to novomcp-nnp

The novomcp-nnp service (ANI-2x, MACE neural potentials) is ~100x faster than xTB for single-point energies but less accurate for orbital properties, solvation, and geometry optimization. The two services complement each other:

| Task | Use novomcp-qm (xTB) | Use novomcp-nnp (ANI-2x/MACE) |
|---|---|---|
| Conformer ranking by energy | When accuracy matters | When speed matters (large batches) |
| Strain energy | Production (accurate) | Quick screening (approximate) |
| Geometry optimization | Yes (xTB) | No (NNPs don't optimize well) |
| Solvation free energy | Yes (ALPB model) | No |
| HOMO-LUMO gap | Yes | No |
| Batch energetics (>50 molecules) | Too slow | Ideal |

---

## Funnel integration

These tools fit into the NovoMCP discovery funnel:

1. **Target discovery** → `target_discovery`
2. **Literature search** → `search_literature`
3. **Lead optimization** → `lead_optimization` (scaffold hopping)
4. **Conformer generation** → `run_conformer_search` (before docking)
5. **Fast energy screening** → `compute_energy` (novomcp-nnp, rank candidates)
6. **Docking** → `dock_molecules` (AutoDock-GPU)
7. **Strain correction** → `dock_with_strain` (filter false positives)
8. **QM refinement** → `run_qm_calculation` (solvation, orbital analysis)
9. **MD simulation** → `run_molecular_dynamics` (GROMACS)
10. **Patient stratification** → `stratify_patients`
