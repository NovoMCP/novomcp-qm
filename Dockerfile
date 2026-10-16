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
# arm64: no release binary exists.
#   - CREST 3.0.2 from conda-forge (micromamba). Its internal GFN2 is correct on
#     GB10 (conformer energies agree with x86 xtb).
#   - xtb 6.7.1 compiled FROM SOURCE at -O2 with conda-forge gfortran 13 (Debian
#     bookworm's gfortran 12 can't compile xtb 6.7.1's gfnff_eg.f90). Only the
#     runtime libs it needs are kept, in /opt/xtb/lib (rpath). The conda-forge linux-aarch64 xtb
#     (6.6.1 and 6.7.1) and any -O3 build compute WRONG GFN1/GFN2 analytic
#     gradients on aarch64 (benzene dE/dx 0.0153 vs numerical 0.0913), so
#     optimizations climb uphill into broken geometries while still reporting
#     "converged". Same source at -O2 matches x86 xtb to ~1e-6 Eh (validated on a
#     DGX Spark GB10). Also patches an upstream format string in
#     src/optimizer.f90 that newer libgfortran rejects at runtime.
#   Both are linked so the app's XTBHOME=/opt/xtb + PATH assumptions still hold.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xj -C /usr/local bin/micromamba && \
      /usr/local/bin/micromamba create -y -p /opt/conda-qm -c conda-forge crest=3.0.2 && \
      ln -sf /opt/conda-qm/bin/crest /usr/local/bin/crest && \
      /usr/local/bin/micromamba create -y -p /opt/xtb-toolchain -c conda-forge \
        "gfortran_linux-aarch64=13" "gcc_linux-aarch64=13" meson ninja pkg-config \
        libopenblas "liblapack=*=*openblas" && \
      T=/opt/xtb-toolchain && \
      export PATH=$T/bin:$PATH PKG_CONFIG_PATH=$T/lib/pkgconfig LIBRARY_PATH=$T/lib CPATH=$T/include \
             FC=$T/bin/aarch64-conda-linux-gnu-gfortran CC=$T/bin/aarch64-conda-linux-gnu-gcc && \
      git clone --depth 1 --branch v6.7.1 https://github.com/grimme-lab/xtb.git /tmp/xtb-src && \
      cd /tmp/xtb-src && \
      sed -i 's/"("f7.2"%)"/"(",f7.2,"%)"/' src/optimizer.f90 && \
      grep -q '"(",f7.2,"%)"' src/optimizer.f90 && \
      meson setup build --buildtype=release -Doptimization=2 -Dlapack=netlib -Ddefault_library=static \
        -Dfortran_link_args=-Wl,-rpath,/opt/xtb/lib -Dc_link_args=-Wl,-rpath,/opt/xtb/lib && \
      ninja -C build xtb && \
      mkdir -p /opt/xtb/bin /opt/xtb/lib /opt/xtb/share/xtb && \
      cp build/xtb /opt/xtb/bin/xtb && chmod +x /opt/xtb/bin/xtb && \
      cp param_*.txt .param_gfnff.xtb /opt/xtb/share/xtb/ && \
      for l in $(LD_LIBRARY_PATH=$T/lib ldd /opt/xtb/bin/xtb | awk '/xtb-toolchain/ {print $3}'); do cp -L "$l" /opt/xtb/lib/; done && \
      cd / && rm -rf /tmp/xtb-src /opt/xtb-toolchain && \
      /usr/local/bin/micromamba clean -afy && \
      ! ldd /opt/xtb/bin/xtb | grep -E "not found|xtb-toolchain" && \
      /opt/xtb/bin/xtb --version | grep -q "6.7.1" ; \
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
# arm64: compile from source with meson + openblas (MKL is x86-only). Both tools
#   are plain Fortran and build cleanly on aarch64 — VALIDATED ON A DGX SPARK
#   (formaldehyde S1 n->pi* 4.33 eV / 286 nm, end-to-end through run_excited_states).
#   Three packaging gotchas, all handled here: (1) stda's binary is named `std2`,
#   install it as `stda` (the name the service calls); (2) stda dyn-links the meson
#   libcint.so subproject -> copy it onto the lib path + ldconfig; (3) xtb4stda
#   resolves its params via XTB4STDAHOME (set below), not ~/$HOME.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      git clone --depth 1 https://github.com/grimme-lab/stda.git /tmp/stda && \
      cd /tmp/stda && meson setup build --buildtype=release -Dla_backend=openblas "-Dfortran_args=-std=legacy -w -fallow-argument-mismatch" && ninja -C build && \
      cp "$(find build -name std2 -type f -executable | head -1)" /usr/local/bin/stda && chmod +x /usr/local/bin/stda && \
      cp "$(find build -name 'libcint.so*' -type f | head -1)" /usr/local/lib/ && ldconfig && \
      git clone --depth 1 https://github.com/grimme-lab/xtb4stda.git /tmp/xtb4stda && \
      cd /tmp/xtb4stda && meson setup build --buildtype=release -Dla_backend=openblas "-Dfortran_args=-std=legacy -w -fallow-argument-mismatch" && ninja -C build && \
      cp "$(find build -name xtb4stda -type f -executable | head -1)" /usr/local/bin/xtb4stda && chmod +x /usr/local/bin/xtb4stda && \
      rm -rf /tmp/stda /tmp/xtb4stda ; \
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
ENV XTBPATH="/opt/xtb/share/xtb"
ENV OMP_NUM_THREADS=4
ENV OMP_STACKSIZE=1G
# sTDA: xtb4stda finds its parameter files via XTB4STDAHOME (it does not expand
# ~/$HOME); stda loads the libcint.so copied onto the lib path above.
ENV XTB4STDAHOME="/opt/xtb4stda-params"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

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
