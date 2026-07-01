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

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 \
    -r requirements.txt

# Copy source code
COPY . .

# ── Environment Configuration ────────────────────────────────────
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Create results directory
RUN mkdir -p /app/results

# ── Default Execution ────────────────────────────────────────────
# Hugging Face Spaces expects the container to run on port 7860
EXPOSE 7860
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
