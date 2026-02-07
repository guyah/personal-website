# personal-website

Minimal personal site + backend scaffold.

## Structure
- `site/` — Astro frontend
- `backend/` — FastAPI service (Docker only)
- `docker-compose.yml` — Postgres + backend + optional site dev service

## Frontend (Astro)
```bash
cd site
npm install
npm run dev
```

The RSS feed uses `site/astro.config.mjs -> site`. Update it to your production domain when ready.

## Backend + DB (Docker only)
```bash
docker compose up --build
```

Services:
- Postgres: `localhost:5432`
- Backend health: `http://localhost:8000/health`
- Site dev server (optional): `http://localhost:4321`

## Talk to Guy (voice)
The voice cockpit uses a WebSocket at `ws://localhost:8000/ws/talk` powered by Pipecat + OpenAI.

### Local setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY=your_key_here
uvicorn app.main:app --reload --port 8000
```

In another terminal:
```bash
cd site
npm install
npm run dev
```

### One-command dev
With your backend venv already active and `OPENAI_API_KEY` exported:
```bash
./scripts/dev.sh
```

## Goals
- Minimal, fast, content-first site
- Blog + RSS
- Deployed only with explicit Guy “GO”
