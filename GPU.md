# Running the ALCHEMI conformer path

`run_conformer_search` defaults to **CREST** (GFN2-xTB metadynamics) on CPU. The
optional **ALCHEMI GPU path** (`engine="alchemi"`) instead generates a diverse
ensemble with RDKit ETKDG and relaxes the **whole ensemble in one batched pass on
the GPU** using the [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit)
(FIRE + MACE-MPA-0, the MIT-licensed foundation potential), then ranks/deduplicates by MLIP energy.

- **Faster** than CREST (no xTB metadynamics), and **better energies/geometries**
  than the CREST→RDKit-MMFF fallback (a real MLIP vs MMFF).
- Activates when you run the **GPU image**, a GPU is present, and the request uses
  `engine="alchemi"`. Otherwise it **transparently falls back to CREST/RDKit** —
  the result's `method` field reports which engine actually ran.

## What you need

| Requirement | Detail |
|---|---|
| GPU | Any NVIDIA CUDA GPU. Validated on an A10G (24 GB) and L40S. |
| Driver | **>= 580** (CUDA 13). Check with `nvidia-smi`. |
| Container runtime | [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so `--gpus all` exposes the GPU. |

Any single CUDA-13 GPU host works — a cloud GPU VM (an image that ships the
NVIDIA driver + CUDA, e.g. an AWS "Deep Learning OSS Nvidia Driver AMI" on a
`g5`/`g6e` instance, or Lambda / RunPod / GCP / Azure GPU images), or your own
workstation with a >= 580 driver. Bring the GPU up, run your batch, tear it down.

## Build & run

The GPU image extends the base image, so build the base first:

```bash
docker build -t novomcp-qm .                        # base (xtb / CREST)
docker build -f Dockerfile.gpu -t novomcp-qm:gpu .  # + ALCHEMI toolkit
docker run --gpus all -p 8031:8031 novomcp-qm:gpu
```

First `engine="alchemi"` request downloads the MACE-MPA-0 weights and warms the
model (a minute or two) — subsequent calls are fast.

## Verify

```bash
curl -s -X POST localhost:8031/api/conformer-search \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CC(C)CC(=O)O","max_conformers":8,"engine":"alchemi"}'
```

A working GPU path returns `"method": "ALCHEMI-MACE-MPA-0"` and a ranked, Boltzmann-
weighted ensemble (populations sum to ~1.0). If you instead see `CREST-GFN2` or
`RDKit-ETKDG-MMFF`, the GPU/toolkit wasn't visible — check `nvidia-smi` inside the
container (`docker run --gpus all novomcp-qm:gpu nvidia-smi`).

## Notes

- **Python version.** The toolkit is validated on Python 3.12. If the base image
  is on an older Python without toolkit wheels, bump the base's `FROM` to
  `python:3.12-slim` (xtb is a static binary and is unaffected).
- **Eager mode.** The toolkit's host-side auto neighbor-list selection can't run
  inside `torch.compile`, so the conformer path runs eager (`TORCHDYNAMO_DISABLE=1`,
  set in the image). Still GPU-batched.
- **Potential.** The ALCHEMI path uses MACE-MPA-0 (MIT-licensed); its absolute
  energies differ from GFN2-xTB, but conformer *ranking* (relative energies) is
  what matters. MPA-0 is a medium-size model — more accurate, and heavier, than
  the previous small model.
- **Neutral, closed-shell only** for the ALCHEMI path.
