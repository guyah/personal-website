FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Install deps via uv (docker-only; no venv on host)
COPY pyproject.toml ./
# uv.lock is optional for now; keep copy permissive
COPY uv.lock ./
RUN uv pip install --system -r pyproject.toml

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
