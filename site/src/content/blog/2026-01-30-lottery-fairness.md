---
title: "Is the lottery rigged? A deeper, data-first audit of Lebanese Loto (6/42)"
description: "Real draw history, multiple sanity checks, and Monte Carlo baselines: frequency, co-occurrence, streakiness, entropy, and drift — with strong caveats about multiple testing and data source quality."
pubDate: 2026-02-03
tags: [data-science, statistics]
icon: "🎲"
---

## TL;DR

- **Dataset:** 200 draws from **2024-02-22 → 2026-02-02** (scraped from a non-official site).
- **Global frequency test:** chi²(df=41)=40.26 → **p≈0.503** (nothing weird).
- **“Hot” & “cold” leaders exist** by chance: hottest **35 (+13.4)**, coldest **9 (−12.6)**.
- **Most “connected” pair:** (36, 41) appears **10×** vs ~**3.48× expected**, but there are **861 pairs**, so outliers are expected.
- **Longest drought:** ball **11** went **43 draws** without appearing (z≈2.2); with 42 balls, that’s not shocking.
- **Consecutive numbers, repeats, odd/even, low/high, and drift** all sit **well within 1–2σ** of a fair Monte Carlo baseline.

Everything below is reproducible from this repo.

## Data sanity (before statistics)

![](/blog/2026-01-30-lottery-fairness/data-sanity.svg)

Notes:

- Draws are **mostly Monday + Thursday** (plus **4 Tuesdays** — likely makeup/holiday shifts).
- **No duplicate balls inside a draw**, and **no out-of-range values** detected in the file.

## 1) Frequency deviations (still the base check)

![](/blog/2026-01-30-lottery-fairness/freq-deviation-lebanon.svg)

The plot is **observed count − expected count** for each ball.

## 2) Global chi-square (still boring — which is good)

![](/blog/2026-01-30-lottery-fairness/chi2-summary.svg)

A p-value around 0.5 is exactly what you expect from a fair process at this sample size.

## 3) Day-of-week effects (sanity check)

![](/blog/2026-01-30-lottery-fairness/weekday-counts.svg)

This is mostly to make sure the dataset behaves like the real schedule. It does.

## 4) Inter-number co-occurrence (with multiple-testing caveat)

![](/blog/2026-01-30-lottery-fairness/pair-cooccurrence-heatmap.svg)

Each cell shows **observed / expected** pair frequency (clipped to 0.6–1.4 so the color scale isn’t dominated by outliers).

- Top lift: **(36, 41)** and several others show **10 co-draws**.
- Expected for any pair is only **~3.48** in 200 draws.
- But there are **861 unordered pairs**, so you *will* see outliers even if nothing is biased.

## 5) Streakiness / droughts (permutation null)

![](/blog/2026-01-30-lottery-fairness/streakiness-gaps-z.svg)

For each ball, I compute its **longest drought** (max run of missing draws), then compare to a **permutation null** that preserves its total count.

- Largest z-score: **ball 11**, **43-draw drought**, **z≈2.2**.
- With 42 balls, “one ~2σ event” is not surprising.

## 6) Entropy of draw features (Monte Carlo baseline)

![](/blog/2026-01-30-lottery-fairness/entropy-sum-mc.svg)

I compute Shannon entropy of the **sum-of-draws** distribution and compare it to a Monte Carlo baseline (6/42 without replacement).

- Observed **H≈6.394 bits**, MC mean **≈6.286 bits** → **z≈1.54**.
- Odd-count and low/high entropies are even closer to the null (**|z| < 1**).

## 7) “Weirdness” dashboard (consecutive numbers, repeats, balance)

![](/blog/2026-01-30-lottery-fairness/metric-zscores.svg)

All of these land **well within 1σ** of the Monte Carlo baseline:

- Consecutive pairs: **z≈0.72**
- Repeats vs previous draw: **z≈0.23**
- Sum, odd count, low count: **|z| < 0.6**

## 8) Early vs late era (drift check)

![](/blog/2026-01-30-lottery-fairness/frequency-drift-halves.svg)

Splitting the dataset in half (first 100 vs last 100 draws):

- Largest drift is **ball 18 (−15)** in the late half.
- That’s not huge given the number of balls tested; it looks like normal variance.

## What this means (and what it doesn’t)

- **No compelling evidence of bias** in this sample.
- **Many “weird-looking” patterns appear automatically** when you have dozens of metrics and hundreds of pairs.
- To claim bias, you’d need **strong statistics + a plausible mechanism** (ball batch, machine change, venue switch, etc.).

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r site/analysis/requirements.txt
python site/analysis/generate_all.py
```

## Data source note

Draws are scraped from `lebanon-lotto.com`, which **explicitly says it is not an official LLDJ site**. Treat these results as best-effort public data unless you cross-check with the official operator.
