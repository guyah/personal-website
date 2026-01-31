# Codex Data Science Workflow

A lightweight, repeatable workflow for data-science tasks in this repo. The goal is reproducible, minimal-dependency analysis with clear caveats.

## 1) Clarify the question

- Define the hypothesis (or decision) and what would change your mind.
- List assumptions and what “real data” means for this task.
- Identify time ranges, filters, and known rule changes.

## 2) Fetch or load data

- Prefer existing fetch scripts in `site/analysis/`.
- Store raw data under `site/analysis/data/` and don’t overwrite raw inputs.
- Record data provenance: URL, date range, and any scrape warnings.

## 3) Sanity checks (always)

- Schema: expected columns and types.
- Ranges: min/max values, out-of-range counts.
- Duplicates: within a row and across rows.
- Date coverage: start/end, missing gaps, weekday patterns.
- Report any anomalies explicitly in the write-up.

## 4) Core analysis

- Start with simple baselines (frequency, summary stats).
- Add at least one null model (permutation or Monte Carlo) if inference is involved.
- Use minimal dependencies (stdlib + `numpy` preferred).
- Record RNG seeds and simulation counts for reproducibility.

## 5) Multiple testing & caveats

- If you compute many metrics or pairs, call it out.
- Prefer effect sizes and confidence intervals over just p-values.
- Avoid claiming bias without a plausible mechanism.

## 6) Plotting (SVG-first)

- Use `site/analysis/generate_all.py` for deterministic SVGs.
- Place outputs under `site/public/blog/<slug>/`.
- Keep plots labeled and include units.

## 7) Write-up

- Lead with crisp takeaways, then show evidence.
- Include data source caveats and limitations.
- Separate observations from interpretations.

## 8) Verification checklist

- Re-run the pipeline end-to-end.
- Confirm plot paths referenced in the post exist.
- Check the blog post for stale dates.
- If a result changes with a new fetch, note the data version.

## Suggested command flow

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r site/analysis/requirements.txt
python site/analysis/fetch_lebanon_loto.py
python site/analysis/generate_all.py
```
