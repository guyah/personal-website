# Analysis scripts for blog posts

This folder contains reproducible scripts that generate plots used in `src/content/blog/*`.

## Setup

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r site/analysis/requirements.txt
python site/analysis/generate_all.py
```

Plots are exported under `site/public/blog/<slug>/...` and referenced from the markdown posts.
