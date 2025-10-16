# Dockerfile
FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies with uv
RUN uv sync --frozen --no-dev

COPY . /app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["uv", "run", "streamlit", "run", "web/entrypoint.py", "--server.port=8501", "--server.address=0.0.0.0"]