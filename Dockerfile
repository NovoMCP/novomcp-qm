FROM python:3.11-slim-bookworm

# buildx sets TARGETARCH (amd64 / arm64). The upstream grimme-lab release
# binaries (xtb, crest, xtb4stda/stda) are x86_64-only, so the arm64 build takes
# a different route: conda-forge has linux-aarch64 builds of xtb/crest (verified
# 2026-09-28), and the sTDA excited-state tools — which have NO aarch64 binary
# anywhere — are compiled from source. amd64 keeps the exact proven binary path.
ARG TARGETARCH

WORKDIR /app

# System deps (git/cmake/meson/ninja added for the aarch64 sTDA source build)
RUN apt-get update && apt-get install -y \
    curl gcc g++ gfortran libxrender1 libxext6 libgomp1 \
    libopenblas-dev wget xz-utils git cmake meson ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- xtb 6.7.1 + CREST 3.0.2 -------------------------------------------------
# amd64: grimme-lab release binaries into /opt/xtb + /usr/local/bin (unchanged).
# arm64: no release binary exists -> conda-forge (micromamba), symlinked so the
#        app's XTBHOME=/opt/xtb + PATH assumptions still hold.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xj -C /usr/local bin/micromamba && \
      /usr/local/bin/micromamba create -y -p /opt/conda-qm -c conda-forge xtb=6.7.1 crest=3.0.2 && \
      /usr/local/bin/micromamba clean -afy && \
      mkdir -p /opt/xtb/bin /opt/xtb/share && \
      ln -sf /opt/conda-qm/bin/xtb /opt/xtb/bin/xtb && \
      ln -sf /opt/conda-qm/bin/crest /usr/local/bin/crest && \
      ln -sf /opt/conda-qm/share/xtb /opt/xtb/share/xtb ; \
    else \
      mkdir -p /opt/xtb && \
      wget -q "https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz" -O /tmp/xtb.tar.xz && \
      tar -xJf /tmp/xtb.tar.xz -C /opt/xtb --strip-components=1 && rm /tmp/xtb.tar.xz && \
      wget -q "https://github.com/crest-lab/crest/releases/download/v3.0.2/crest-gnu-12-ubuntu-latest.tar.xz" -O /tmp/crest.tar.xz && \
      tar -xJf /tmp/crest.tar.xz -C /tmp && cp /tmp/crest/crest /usr/local/bin/crest && \
      chmod +x /usr/local/bin/crest && rm -rf /tmp/crest /tmp/crest.tar.xz ; \
    fi

# ---- xtb4stda + stda (sTDA excited states: predict_frontier_orbitals / run_excited_states) ----
# No conda-forge or aarch64 binary exists for either tool.
# amd64: grimme-lab release binaries (unchanged, proven).
# arm64: compile from source (stda = CMake/Fortran; xtb4stda = Makefile/Fortran).
#   NOTE: this arm64 source build is drafted from the upstream build docs and is
#   VALIDATED ON A DGX SPARK before this branch merges (can't build arm64 in CI).
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      git clone --depth 1 https://github.com/grimme-lab/stda.git /tmp/stda && \
      cd /tmp/stda && cmake -B build -G Ninja && cmake --build build && \
      cp build/stda /usr/local/bin/stda && chmod +x /usr/local/bin/stda && \
      git clone --depth 1 https://github.com/grimme-lab/xtb4stda.git /tmp/xtb4stda && \
      cd /tmp/xtb4stda && make && cp exe/xtb4stda /usr/local/bin/xtb4stda && \
      chmod +x /usr/local/bin/xtb4stda && rm -rf /tmp/stda /tmp/xtb4stda ; \
    else \
      wget -q "https://github.com/grimme-lab/xtb4stda/releases/download/v1.0/xtb4stda" -O /usr/local/bin/xtb4stda && \
      wget -q "https://github.com/grimme-lab/xtb4stda/releases/download/v1.0/stda_v1.6.1" -O /usr/local/bin/stda && \
      chmod +x /usr/local/bin/xtb4stda /usr/local/bin/stda ; \
    fi

# sTDA parameter files (architecture-independent — always fetched)
RUN mkdir -p /opt/xtb4stda-params && \
    for f in .param_stda1.xtb .param_stda2.xtb .xtb4stdarc \
             .param_gbsa_h2o .param_gbsa_acetonitrile .param_gbsa_dmso \
             .param_gbsa_toluene .param_gbsa_thf .param_gbsa_methanol \
             .param_gbsa_ch2cl2 .param_gbsa_chcl3 .param_gbsa_ether \
             .param_gbsa_acetone .param_gbsa_benzene .param_gbsa_cs2; do \
        wget -q "https://raw.githubusercontent.com/grimme-lab/xtb4stda/master/$f" -O "/opt/xtb4stda-params/$f"; \
    done && \
    test -s /opt/xtb4stda-params/.param_stda2.xtb || (echo "FATAL: param_stda2.xtb is empty" && exit 1)

ENV PATH="/opt/xtb/bin:${PATH}"
ENV XTBHOME="/opt/xtb"
ENV OMP_NUM_THREADS=4
ENV OMP_STACKSIZE=1G

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app/ app/
COPY main.py .

# Non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/scratch && \
    chown -R appuser:appuser /app && \
    cp /opt/xtb4stda-params/.param_stda* /home/appuser/ 2>/dev/null || true && \
    cp /opt/xtb4stda-params/.param_gbsa* /home/appuser/ 2>/dev/null || true && \
    cp /opt/xtb4stda-params/.xtb4stdarc /home/appuser/ 2>/dev/null || true && \
    chown appuser:appuser /home/appuser/.param* /home/appuser/.xtb4stdarc 2>/dev/null || true
USER appuser

ENV PORT=8031
ENV PYTHONUNBUFFERED=1
EXPOSE 8031

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8031/health || exit 1

CMD ["python", "main.py"]
