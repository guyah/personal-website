# Codex Skill: Backend (FastAPI + Postgres)

Goal: docker-only backend for storing data (SQL).

Rules:
- Use Python + FastAPI.
- Dependency management: **uv** (pyproject.toml).
- No host venv assumptions.
- Database: Postgres (SQL).

Commands:
- docker compose up --build
- Health: http://localhost:8000/health

Definition of done:
- `docker compose up --build` starts db + backend.
- /health returns {"status":"ok"}.
