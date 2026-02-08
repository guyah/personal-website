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

## Goals
- Minimal, fast, content-first site
- Blog + RSS
- Deployed only with explicit Guy “GO”
