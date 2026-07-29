FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    curl gcc g++ gfortran libxrender1 libxext6 libgomp1 \
    libopenblas-dev wget xz-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtb (v6.7.1) from GitHub releases
RUN mkdir -p /opt/xtb && \
    wget -q "https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz" -O /tmp/xtb.tar.xz && \
    tar -xJf /tmp/xtb.tar.xz -C /opt/xtb --strip-components=1 && \
    rm /tmp/xtb.tar.xz

# Install xtb4stda + stda (v1.6.1) for excited state calculations
# Two-step workflow: xtb4stda generates wavefunction → stda computes excited states
# xtb4stda requires parameter files (.param_stda1.xtb, .param_stda2.xtb) in $HOME.
# Also needs GBSA solvent param files for solvated excited states.
RUN wget -q "https://github.com/grimme-lab/xtb4stda/releases/download/v1.0/xtb4stda" -O /usr/local/bin/xtb4stda && \
    wget -q "https://github.com/grimme-lab/xtb4stda/releases/download/v1.0/stda_v1.6.1" -O /usr/local/bin/stda && \
    chmod +x /usr/local/bin/xtb4stda /usr/local/bin/stda && \
    mkdir -p /opt/xtb4stda-params && \
    for f in .param_stda1.xtb .param_stda2.xtb .xtb4stdarc \
             .param_gbsa_h2o .param_gbsa_acetonitrile .param_gbsa_dmso \
             .param_gbsa_toluene .param_gbsa_thf .param_gbsa_methanol \
             .param_gbsa_ch2cl2 .param_gbsa_chcl3 .param_gbsa_ether \
             .param_gbsa_acetone .param_gbsa_benzene .param_gbsa_cs2; do \
        wget -q "https://raw.githubusercontent.com/grimme-lab/xtb4stda/master/$f" -O "/opt/xtb4stda-params/$f"; \
    done && \
    test -s /opt/xtb4stda-params/.param_stda2.xtb || (echo "FATAL: param_stda2.xtb is empty" && exit 1)

# Install CREST (v3.0.2) — statically linked binary, extract to /usr/local/bin
RUN wget -q "https://github.com/crest-lab/crest/releases/download/v3.0.2/crest-gnu-12-ubuntu-latest.tar.xz" -O /tmp/crest.tar.xz && \
    tar -xJf /tmp/crest.tar.xz -C /tmp && \
    cp /tmp/crest/crest /usr/local/bin/crest && \
    chmod +x /usr/local/bin/crest && \
    rm -rf /tmp/crest /tmp/crest.tar.xz && \
    crest --version || echo "CREST installed"

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
