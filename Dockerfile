# ╔══════════════════════════════════════════════════════════════╗
# ║  Climate-Fed Orchestrator — Production Environment           ║
# ║  Architecture: Debian-slim + Python 3.11 + Torch (CPU)       ║
# ╚══════════════════════════════════════════════════════════════╝

FROM python:3.11-slim-bookworm

# ── System Dependencies ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Project Setup ────────────────────────────────────────────────
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch CPU-only first (200MB vs 915MB full bundle)
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 \
    numpy>=1.24.0 matplotlib>=3.7.0 seaborn>=0.12.0 PyYAML>=6.0 tqdm>=4.64.0 aiohttp python-dotenv

# Copy source code
COPY . .

# ── Environment Configuration ────────────────────────────────────
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Create results directory
RUN mkdir -p /app/results

# ── Default Execution ────────────────────────────────────────────
# Default command runs a full comparison experiment
ENTRYPOINT ["python3", "main.py"]
CMD ["--mode", "full", "--rounds", "10", "--viz", "--out", "/app/results"]
