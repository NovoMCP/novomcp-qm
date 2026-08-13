# novomcp-qm

Quantum-chemistry service for the [NovoMCP](https://github.com/NovoMCP/novomcp) engine — semi-empirical QM (xTB / GFN2), conformer search (CREST), and downstream properties.

Runs on CPU. An optional **GPU conformer path** (`engine="alchemi"`) uses the NVIDIA ALCHEMI Toolkit — see [`GPU.md`](./GPU.md).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness + available engines |
| `POST` | `/api/qm-calculate` | GFN2-xTB single point / optimization |
| `POST` | `/api/conformer-search` | conformer ensemble (CREST or ALCHEMI) |
| `POST` | `/api/predict-frontier-orbitals` | HOMO/LUMO/gap |
| `POST` | `/api/qm-excited-states` | sTDA-xTB excited states |
| `POST` | `/api/predict-redox-potential` | oxidation/reduction potentials |
| `POST` | `/api/predict-reaction-thermo` | ΔE/ΔH/ΔG feasibility |

Backends: **xTB** (GFN2), **CREST** (iMTD-GC conformer sampling), **xtb4stda/stda** (excited states). Long-running calls run as async jobs (poll `/status/{job_id}`) when Redis is wired; otherwise synchronously.

## Conformer search — `engine` axis

`run_conformer_search` / `/api/conformer-search` takes an `engine`:

- **`crest`** (default) — CREST + GFN2-xTB metadynamics sampling.
- **`alchemi`** — RDKit ETKDG generation + **whole-ensemble relaxation in one batched GPU pass** via the [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) (FIRE + MACE-MP-0), ranked by MLIP energy. Faster than CREST; better energies than the MMFF fallback. Requires the GPU image + a GPU (see [`GPU.md`](./GPU.md)); otherwise falls back to CREST. The result's `method` field reports which engine ran.

## Run

Pull the published image and run it — no build required:

```bash
docker run -p 8031:8031 ghcr.io/novomcp/novomcp-qm:latest

curl -s localhost:8031/health
curl -s -X POST localhost:8031/api/conformer-search \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CC(C)CC(=O)O","max_conformers":8}'
```

Or build from source: `docker build -t novomcp-qm . && docker run -p 8031:8031 novomcp-qm`.

GPU conformer path: `docker build -f Dockerfile.gpu -t novomcp-qm:gpu . && docker run --gpus all -p 8031:8031 novomcp-qm:gpu` — full guide in [`GPU.md`](./GPU.md).

## Wire it to the engine

```bash
export NOVOMCP_QM_URL=http://localhost:8031
```

Then `run_qm_calculation`, `run_conformer_search`, `predict_frontier_orbitals`, `predict_redox_potential`, and related tools light up in the engine. See the engine's [deploying-services guide](https://github.com/NovoMCP/novomcp/tree/main/docs/deploying-services).

## Auth

Set `QM_API_KEY` to require an `X-API-Key` header. Unset (default) = no inbound auth — fine for localhost or a private network.

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
