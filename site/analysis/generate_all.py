"""Generate plot images (SVG) referenced by blog posts.

Plots are rendered to SVG for deterministic, lightweight blog assets.

Run from repo root:

  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r site/analysis/requirements.txt
  python site/analysis/generate_all.py

Outputs:
  site/public/blog/<slug>/*.svg
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from plotting import bar_chart, heatmap, histogram, text_panel


ROOT = Path(__file__).resolve().parents[1]  # site/
PUBLIC = ROOT / "public" / "blog"


# -----------------------------
# Small, dependency-free SVG plotting
# -----------------------------

def _svg_header(w: int, h: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; fill:#111}</style>',
        """
<defs>
  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
  </marker>
</defs>
""".strip(),
    ]


def _svg_footer() -> list[str]:
    return ["</svg>"]


def _write_svg(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _line_chart(
    x: np.ndarray,
    ys: list[tuple[np.ndarray, str, str]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out: Path,
    w: int = 900,
    h: int = 420,
) -> None:
    pad_l, pad_r, pad_t, pad_b = 70, 20, 55, 60
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b

    x = np.asarray(x)
    xmin, xmax = float(x.min()), float(x.max())

    y_all = np.concatenate([np.asarray(y) for (y, _, _) in ys])
    ymin, ymax = float(y_all.min()), float(y_all.max())
    if ymin == ymax:
        ymin -= 1
        ymax += 1

    def X(v: float) -> float:
        return pad_l + (v - xmin) / (xmax - xmin) * iw

    def Y(v: float) -> float:
        return pad_t + (1 - (v - ymin) / (ymax - ymin)) * ih

    s = []
    s += _svg_header(w, h)
    s.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-size="16">{title}</text>')

    # axes
    s.append(f'<line x1="{pad_l}" y1="{pad_t+ih}" x2="{pad_l+iw}" y2="{pad_t+ih}" stroke="#333"/>')
    s.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ih}" stroke="#333"/>')

    # grid + ticks
    for i in range(6):
        t = i / 5
        yy = pad_t + t * ih
        val = ymax - t * (ymax - ymin)
        s.append(f'<line x1="{pad_l}" y1="{_fmt(yy)}" x2="{pad_l+iw}" y2="{_fmt(yy)}" stroke="#eee"/>')
        s.append(f'<text x="{pad_l-8}" y="{_fmt(yy+4)}" text-anchor="end" font-size="11" fill="#444">{_fmt(val)}</text>')

    for i in range(6):
        t = i / 5
        xx = pad_l + t * iw
        val = xmin + t * (xmax - xmin)
        s.append(f'<line x1="{_fmt(xx)}" y1="{pad_t}" x2="{_fmt(xx)}" y2="{pad_t+ih}" stroke="#f3f3f3"/>')
        s.append(f'<text x="{_fmt(xx)}" y="{pad_t+ih+18}" text-anchor="middle" font-size="11" fill="#444">{_fmt(val)}</text>')

    # labels
    s.append(f'<text x="{w/2}" y="{h-20}" text-anchor="middle" font-size="12" fill="#333">{xlabel}</text>')
    s.append(
        f'<text x="18" y="{h/2}" text-anchor="middle" font-size="12" fill="#333" transform="rotate(-90 18 {h/2})">{ylabel}</text>'
    )

    # lines
    for y, label, color in ys:
        pts = " ".join([f"{_fmt(X(float(xi)))},{_fmt(Y(float(yi)))}" for xi, yi in zip(x, y)])
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')

    # legend
    lx, ly = pad_l + 10, pad_t + 10
    for i, (_, label, color) in enumerate(ys):
        y0 = ly + i * 18
        s.append(f'<line x1="{lx}" y1="{y0}" x2="{lx+18}" y2="{y0}" stroke="{color}" stroke-width="3"/>')
        s.append(f'<text x="{lx+24}" y="{y0+4}" font-size="12" fill="#222">{label}</text>')

    s += _svg_footer()
    _write_svg(out, s)


def _bar_chart(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out: Path,
    highlight_x: int | None = None,
    x_labels: list[str] | None = None,
    label_every: int = 1,
    w: int = 1000,
    h: int = 420,
) -> None:
    pad_l, pad_r, pad_t, pad_b = 70, 20, 55, 60
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b

    x = np.asarray(x)
    y = np.asarray(y)

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(min(0.0, y.min())), float(max(0.0, y.max()))
    if ymin == ymax:
        ymin -= 1
        ymax += 1

    def X(v: float) -> float:
        return pad_l + (v - xmin) / (xmax - xmin) * iw

    def Y(v: float) -> float:
        return pad_t + (1 - (v - ymin) / (ymax - ymin)) * ih

    s = []
    s += _svg_header(w, h)
    s.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-size="16">{title}</text>')

    # axes
    s.append(f'<line x1="{pad_l}" y1="{pad_t+ih}" x2="{pad_l+iw}" y2="{pad_t+ih}" stroke="#333"/>')
    s.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ih}" stroke="#333"/>')

    # baseline at y=0
    y0 = Y(0.0)
    s.append(f'<line x1="{pad_l}" y1="{_fmt(y0)}" x2="{pad_l+iw}" y2="{_fmt(y0)}" stroke="#111" stroke-width="1" opacity="0.6"/>')

    # draw bars
    # assume x is 1..N
    n = len(x)
    bar_w = iw / n
    for i in range(n):
        xi = pad_l + i * bar_w
        val = float(y[i])
        y_top = Y(max(val, 0.0))
        y_bot = Y(min(val, 0.0))
        hh = y_bot - y_top
        color = "#2c7fb8" if (highlight_x is None or int(x[i]) != highlight_x) else "#d7301f"
        s.append(
            f'<rect x="{_fmt(xi)}" y="{_fmt(y_top)}" width="{_fmt(bar_w*0.92)}" height="{_fmt(abs(hh))}" fill="{color}" opacity="0.9"/>'
        )

        # optional labels (for categorical x)
        if x_labels and (i % max(1, label_every) == 0):
            lab = x_labels[i] if i < len(x_labels) else str(x[i])
            s.append(
                f'<text x="{_fmt(xi+bar_w*0.46)}" y="{pad_t+ih+20}" text-anchor="middle" font-size="11" fill="#333">{lab}</text>'
            )

    s.append(f'<text x="{w/2}" y="{h-20}" text-anchor="middle" font-size="12" fill="#333">{xlabel}</text>')
    s.append(
        f'<text x="18" y="{h/2}" text-anchor="middle" font-size="12" fill="#333" transform="rotate(-90 18 {h/2})">{ylabel}</text>'
    )

    s += _svg_footer()
    _write_svg(out, s)


def _histogram(
    values: np.ndarray,
    *,
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    out: Path,
    vline: float | None = None,
    w: int = 900,
    h: int = 420,
) -> None:
    values = np.asarray(values)
    counts, edges = np.histogram(values, bins=bins)

    # bar chart on bin centers
    centers = (edges[:-1] + edges[1:]) / 2

    pad_l, pad_r, pad_t, pad_b = 70, 20, 55, 60
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b

    xmin, xmax = float(edges[0]), float(edges[-1])
    ymin, ymax = 0.0, float(counts.max())

    def X(v: float) -> float:
        return pad_l + (v - xmin) / (xmax - xmin) * iw

    def Y(v: float) -> float:
        return pad_t + (1 - (v - ymin) / (ymax - ymin)) * ih

    s = []
    s += _svg_header(w, h)
    s.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-size="16">{title}</text>')

    # axes
    s.append(f'<line x1="{pad_l}" y1="{pad_t+ih}" x2="{pad_l+iw}" y2="{pad_t+ih}" stroke="#333"/>')
    s.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ih}" stroke="#333"/>')

    # bars
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        x0 = X(float(a))
        x1 = X(float(b))
        y1 = Y(float(c))
        s.append(
            f'<rect x="{_fmt(x0)}" y="{_fmt(y1)}" width="{_fmt(x1-x0-1)}" height="{_fmt(pad_t+ih-y1)}" fill="#41b6c4" opacity="0.9"/>'
        )

    if vline is not None:
        xv = X(float(vline))
        s.append(f'<line x1="{_fmt(xv)}" y1="{pad_t}" x2="{_fmt(xv)}" y2="{pad_t+ih}" stroke="#d7301f" stroke-width="2"/>')

    s.append(f'<text x="{w/2}" y="{h-20}" text-anchor="middle" font-size="12" fill="#333">{xlabel}</text>')
    s.append(
        f'<text x="18" y="{h/2}" text-anchor="middle" font-size="12" fill="#333" transform="rotate(-90 18 {h/2})">{ylabel}</text>'
    )

    s += _svg_footer()
    _write_svg(out, s)


def _heatmap(
    mat: np.ndarray,
    *,
    title: str,
    out: Path,
    xlabel: str = "key token",
    ylabel: str = "query token",
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    label_every: int = 2,
    w: int = 760,
    h: int = 560,
) -> None:
    mat = np.asarray(mat)
    r, c = mat.shape

    pad_l, pad_r, pad_t, pad_b = 80, 30, 55, 60
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b

    # Normalize 0..1 for colormap
    mn, mx = float(mat.min()), float(mat.max())
    if mn == mx:
        mx = mn + 1
    norm = (mat - mn) / (mx - mn)

    def color(v: float) -> str:
        # simple viridis-ish ramp (hand-rolled)
        # maps 0..1 to RGB through a few anchor points
        anchors = [
            (0.0, (68, 1, 84)),
            (0.25, (59, 82, 139)),
            (0.5, (33, 145, 140)),
            (0.75, (94, 201, 98)),
            (1.0, (253, 231, 37)),
        ]
        for (t0, c0), (t1, c1) in zip(anchors[:-1], anchors[1:]):
            if v <= t1:
                a = (v - t0) / (t1 - t0 + 1e-12)
                rgb = tuple(int(c0[i] + a * (c1[i] - c0[i])) for i in range(3))
                return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        return "rgb(253,231,37)"

    cell_w = iw / c
    cell_h = ih / r

    s = []
    s += _svg_header(w, h)
    s.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-size="16">{title}</text>')

    # axes frame
    s.append(f'<rect x="{pad_l}" y="{pad_t}" width="{iw}" height="{ih}" fill="none" stroke="#333"/>')

    for i in range(r):
        for j in range(c):
            x0 = pad_l + j * cell_w
            y0 = pad_t + i * cell_h
            s.append(
                f'<rect x="{_fmt(x0)}" y="{_fmt(y0)}" width="{_fmt(cell_w)}" height="{_fmt(cell_h)}" fill="{color(float(norm[i,j]))}"/>'
            )

    # ticks
    for j in range(c):
        xx = pad_l + (j + 0.5) * cell_w
        if j % max(1, label_every) == 0:
            lab = x_labels[j] if x_labels and j < len(x_labels) else str(j)
            s.append(f'<text x="{_fmt(xx)}" y="{pad_t+ih+18}" text-anchor="middle" font-size="10" fill="#444">{lab}</text>')
    for i in range(r):
        yy = pad_t + (i + 0.5) * cell_h + 3
        if i % max(1, label_every) == 0:
            lab = y_labels[i] if y_labels and i < len(y_labels) else str(i)
            s.append(f'<text x="{pad_l-8}" y="{_fmt(yy)}" text-anchor="end" font-size="10" fill="#444">{lab}</text>')

    s.append(f'<text x="{w/2}" y="{h-20}" text-anchor="middle" font-size="12" fill="#333">{xlabel}</text>')
    s.append(
        f'<text x="18" y="{h/2}" text-anchor="middle" font-size="12" fill="#333" transform="rotate(-90 18 {h/2})">{ylabel}</text>'
    )

    s += _svg_footer()
    _write_svg(out, s)


# -----------------------------
# Small SVG diagrams (boxes/arrows)
# -----------------------------

def _arrow(s: list[str], x1: float, y1: float, x2: float, y2: float, *, color: str = "#333") -> None:
    s.append(
        f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}" stroke="{color}" stroke-width="2" marker-end="url(#arrowhead)"/>'
    )


def _box(
    s: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str,
    sub: str | None = None,
    fill: str = "#f7fbff",
    stroke: str = "#2c7fb8",
) -> None:
    s.append(f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}" rx="10" fill="{fill}" stroke="{stroke}"/>')
    s.append(f'<text x="{_fmt(x+w/2)}" y="{_fmt(y+h/2-2)}" text-anchor="middle" font-size="14">{label}</text>')
    if sub:
        s.append(f'<text x="{_fmt(x+w/2)}" y="{_fmt(y+h/2+18)}" text-anchor="middle" font-size="11" fill="#444">{sub}</text>')


def _stacked_bar_chart(
    labels: list[str],
    stacks: list[list[float]],
    stack_labels: list[str],
    colors: list[str],
    *,
    title: str,
    ylabel: str,
    out: Path,
    w: int = 980,
    h: int = 440,
) -> None:
    pad_l, pad_r, pad_t, pad_b = 85, 20, 55, 80
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b

    stacks_np = np.asarray(stacks, dtype=float)
    totals = stacks_np.sum(axis=1)
    ymax = float(totals.max()) * 1.08

    def Y(v: float) -> float:
        return pad_t + (1 - v / (ymax + 1e-12)) * ih

    n = len(labels)
    bar_w = iw / n

    s: list[str] = []
    s += _svg_header(w, h)
    s.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-size="16">{title}</text>')

    # axes
    s.append(f'<line x1="{pad_l}" y1="{pad_t+ih}" x2="{pad_l+iw}" y2="{pad_t+ih}" stroke="#333"/>')
    s.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ih}" stroke="#333"/>')

    # y ticks
    for i in range(6):
        t = i / 5
        val = ymax * (1 - t)
        yy = pad_t + t * ih
        s.append(f'<line x1="{pad_l}" y1="{_fmt(yy)}" x2="{pad_l+iw}" y2="{_fmt(yy)}" stroke="#eee"/>')
        s.append(f'<text x="{pad_l-8}" y="{_fmt(yy+4)}" text-anchor="end" font-size="11" fill="#444">{_fmt(val)}</text>')

    # bars
    for i, lab in enumerate(labels):
        x0 = pad_l + i * bar_w + bar_w * 0.08
        bw = bar_w * 0.84
        y_base = 0.0
        for j, seg in enumerate(stacks_np[i]):
            y1 = y_base
            y2 = y_base + float(seg)
            y_top = Y(y2)
            y_bot = Y(y1)
            s.append(
                f'<rect x="{_fmt(x0)}" y="{_fmt(y_top)}" width="{_fmt(bw)}" height="{_fmt(y_bot-y_top)}" fill="{colors[j]}" opacity="0.95"/>'
            )
            y_base = y2
        s.append(f'<text x="{_fmt(x0+bw/2)}" y="{pad_t+ih+20}" text-anchor="middle" font-size="11" fill="#333">{lab}</text>')

    s.append(f'<text x="{w/2}" y="{h-20}" text-anchor="middle" font-size="12" fill="#333">{ylabel}</text>')

    # legend
    lx, ly = pad_l, h - 55
    for i, (sl, c) in enumerate(zip(stack_labels, colors)):
        x = lx + i * 190
        s.append(f'<rect x="{_fmt(x)}" y="{_fmt(ly-10)}" width="16" height="16" fill="{c}"/>')
        s.append(f'<text x="{_fmt(x+22)}" y="{_fmt(ly+3)}" font-size="12">{sl}</text>')

    s += _svg_footer()
    _write_svg(out, s)


# -----------------------------
# Stats helpers (no SciPy)
# -----------------------------

def _gammainc_lower_reg(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x).

    Uses series expansion for x < a+1, otherwise continued fraction.
    Based on Numerical Recipes-style implementations.
    """

    if x <= 0:
        return 0.0

    # constants
    eps = 1e-12
    itmax = 500

    gln = math.lgamma(a)

    if x < a + 1:
        # series
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(itmax):
            ap += 1
            delta *= x / ap
            summ += delta
            if abs(delta) < abs(summ) * eps:
                break
        return summ * math.exp(-x + a * math.log(x) - gln)

    # continued fraction for Q(a,x); then P = 1 - Q
    b = x + 1 - a
    c = 1 / 1e-30
    d = 1 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    q = h * math.exp(-x + a * math.log(x) - gln)
    return 1.0 - q


def chi2_sf(x: float, k: int) -> float:
    """Chi-square survival function (1 - CDF)."""
    a = k / 2.0
    return 1.0 - _gammainc_lower_reg(a, x / 2.0)


# -----------------------------
# Plot generators
# -----------------------------

def plot_lottery_fairness() -> None:
    """Lottery fairness check using *real* Lebanese Loto draw history (6/42).

    Data source is scraped via `site/analysis/fetch_lebanon_loto.py`.
    """

    import csv
    import datetime as dt
    import subprocess
    from collections import Counter

    slug_dir = PUBLIC / "2026-01-30-lottery-fairness"

    # Ensure dataset exists
    data_csv = ROOT / "analysis" / "data" / "lebanon_loto_draws.csv"
    if not data_csv.exists():
        subprocess.check_call(["python3", str(ROOT / "analysis" / "fetch_lebanon_loto.py")])

    # Load draws
    draws: list[list[int]] = []
    dates: list[dt.date] = []
    with data_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            nums = [int(row[f"n{i}"]) for i in range(1, 7) if row.get(f"n{i}")]
            if len(nums) != 6:
                continue
            try:
                d = dt.date.fromisoformat(row["date"])
            except Exception:
                continue
            draws.append(sorted(nums))
            dates.append(d)

    if not draws:
        raise RuntimeError(f"No draws loaded from {data_csv}")

    n_balls = 42
    picks = 6
    n_draws = len(draws)

    # Sort by date just in case the CSV isn't strictly ordered.
    order = np.argsort(np.array([d.toordinal() for d in dates]))
    draws = [draws[i] for i in order]
    dates = [dates[i] for i in order]

    rng = np.random.default_rng(7)

    # -----------------------------
    # Data sanity checks
    # -----------------------------
    out_of_range = 0
    duplicate_in_draw = 0
    for d in draws:
        if any(v < 1 or v > n_balls for v in d):
            out_of_range += 1
        if len(set(d)) != len(d):
            duplicate_in_draw += 1

    date_min = dates[0]
    date_max = dates[-1]
    weekday_counts = Counter([d.weekday() for d in dates])  # Mon=0

    text_panel(
        title="Lebanese Loto (6/42) — dataset sanity",
        lines=[
            f"draws: {n_draws}",
            f"date range: {date_min.isoformat()} → {date_max.isoformat()}",
            f"duplicates in draw: {duplicate_in_draw}",
            f"out-of-range draws: {out_of_range}",
            (
                "weekday counts: "
                f"Mon={weekday_counts.get(0,0)}, Tue={weekday_counts.get(1,0)}, "
                f"Wed={weekday_counts.get(2,0)}, Thu={weekday_counts.get(3,0)}, "
                f"Fri={weekday_counts.get(4,0)}, Sat={weekday_counts.get(5,0)}, "
                f"Sun={weekday_counts.get(6,0)}"
            ),
        ],
        footer="Source: lebanon-lotto.com (not official LLDJ).",
        out=slug_dir / "data-sanity.svg",
        w=980,
        h=240,
    )

    counts = np.zeros(n_balls, dtype=float)
    for d in draws:
        for v in d:
            if 1 <= v <= n_balls:
                counts[v - 1] += 1

    expected = n_draws * picks / n_balls
    x = np.arange(1, n_balls + 1)
    delta = counts - expected

    highlight_x = int(x[np.argmax(np.abs(delta))])

    bar_chart(
        x,
        delta,
        title=f"Lebanese Loto (6/42): deviation from expected frequency (real draws, n={n_draws})",
        xlabel="ball",
        ylabel="count − expected",
        highlight_x=highlight_x,
        out=slug_dir / "freq-deviation-lebanon.svg",
        w=1050,
    )

    # Global chi-square goodness-of-fit on real data.
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p = chi2_sf(chi2, n_balls - 1)

    text_panel(
        title="Lebanese Loto (6/42) — chi-square goodness-of-fit",
        lines=[
            f"draws: {n_draws}",
            f"chi² (df={n_balls-1}): {_fmt(chi2)}",
            f"p-value: {_fmt(p)}",
        ],
        footer=(
            "Interpretation: small p-values suggest counts are unlikely under a perfectly uniform model.",
            12,
            "#444",
        ),
        out=slug_dir / "chi2-summary.svg",
        w=900,
        h=220,
    )

    # -----------------------------
    # 1) Day-of-week effects (sanity)
    # -----------------------------
    week_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    week_counts = np.array([weekday_counts.get(i, 0) for i in range(7)], dtype=float)
    bar_chart(
        np.arange(1, 8),
        week_counts,
        title="Draw count by weekday (sanity check)",
        xlabel="weekday",
        ylabel="draws",
        out=slug_dir / "weekday-counts.svg",
        x_labels=week_labels,
        label_every=1,
        w=900,
    )

    # -----------------------------
    # 2) Inter-number co-occurrence (lift vs expected)
    # -----------------------------
    pair_counts = np.zeros((n_balls, n_balls), dtype=float)
    for d in draws:
        for i in range(picks):
            for j in range(i + 1, picks):
                a, b = d[i] - 1, d[j] - 1
                pair_counts[a, b] += 1
                pair_counts[b, a] += 1

    expected_pair = n_draws * (picks * (picks - 1) / 2) / (n_balls * (n_balls - 1) / 2)
    lift = pair_counts / max(1e-9, expected_pair)
    np.fill_diagonal(lift, 1.0)
    lift_clipped = np.clip(lift, 0.6, 1.4)
    labels = [str(i) for i in range(1, n_balls + 1)]
    heatmap(
        lift_clipped,
        title="Pair co-occurrence lift (observed / expected, clipped)",
        out=slug_dir / "pair-cooccurrence-heatmap.svg",
        xlabel="ball j",
        ylabel="ball i",
        x_labels=labels,
        y_labels=labels,
        label_every=4,
        w=940,
        h=720,
    )

    # -----------------------------
    # 3) Streakiness: longest gaps vs permutation null
    # -----------------------------
    def _longest_gap(bits: np.ndarray) -> int:
        longest = 0
        current = 0
        for v in bits:
            if v == 0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    max_gaps = []
    gap_z = []
    gap_mu = []
    gap_sd = []
    n_perm = 400

    for ball in range(1, n_balls + 1):
        bits = np.array([1 if ball in d else 0 for d in draws], dtype=np.int8)
        obs_gap = _longest_gap(bits)
        sims = []
        for _ in range(n_perm):
            perm = rng.permutation(bits)
            sims.append(_longest_gap(perm))
        mu = float(np.mean(sims))
        sd = float(np.std(sims) + 1e-9)
        z = (obs_gap - mu) / sd
        max_gaps.append(obs_gap)
        gap_z.append(z)
        gap_mu.append(mu)
        gap_sd.append(sd)

    histogram(
        np.array(gap_z, dtype=float),
        bins=16,
        title="Longest drought z-scores per ball (permutation null)",
        xlabel="z-score",
        ylabel="balls",
        out=slug_dir / "streakiness-gaps-z.svg",
        vline=0.0,
        w=900,
    )

    # -----------------------------
    # 4) Monte Carlo baselines (6/42 without replacement)
    # -----------------------------
    def _draw_once() -> list[int]:
        return sorted((rng.choice(n_balls, size=picks, replace=False) + 1).tolist())

    def _entropy(vals: list[int]) -> float:
        c = Counter(vals)
        total = sum(c.values())
        return float(-sum((v / total) * math.log2(v / total) for v in c.values() if v))

    # observed per-draw metrics
    sums_obs = [sum(d) for d in draws]
    odds_obs = [sum(1 for v in d if v % 2 == 1) for d in draws]
    lows_obs = [sum(1 for v in d if v <= 21) for d in draws]
    consec_obs = [sum(1 for a, b in zip(d, d[1:]) if b - a == 1) for d in draws]
    repeats_obs = [len(set(draws[i]) & set(draws[i - 1])) for i in range(1, n_draws)]

    # simulation
    mc_iters = 700
    mc_entropy_sum = []
    mc_entropy_odd = []
    mc_entropy_low = []
    mc_consec_mean = []
    mc_repeat_mean = []
    mc_sum_mean = []
    mc_odd_mean = []
    mc_low_mean = []

    for _ in range(mc_iters):
        sim_draws = [_draw_once() for _ in range(n_draws)]
        sim_sums = [sum(d) for d in sim_draws]
        sim_odds = [sum(1 for v in d if v % 2 == 1) for d in sim_draws]
        sim_lows = [sum(1 for v in d if v <= 21) for d in sim_draws]
        sim_consec = [sum(1 for a, b in zip(d, d[1:]) if b - a == 1) for d in sim_draws]
        sim_repeats = [len(set(sim_draws[i]) & set(sim_draws[i - 1])) for i in range(1, n_draws)]

        mc_entropy_sum.append(_entropy(sim_sums))
        mc_entropy_odd.append(_entropy(sim_odds))
        mc_entropy_low.append(_entropy(sim_lows))

        mc_consec_mean.append(float(np.mean(sim_consec)))
        mc_repeat_mean.append(float(np.mean(sim_repeats)))
        mc_sum_mean.append(float(np.mean(sim_sums)))
        mc_odd_mean.append(float(np.mean(sim_odds)))
        mc_low_mean.append(float(np.mean(sim_lows)))

    # entropy plot (sum distribution)
    histogram(
        np.array(mc_entropy_sum, dtype=float),
        bins=18,
        title="Entropy of draw sums: Monte Carlo null (6/42)",
        xlabel="entropy (bits)",
        ylabel="MC samples",
        out=slug_dir / "entropy-sum-mc.svg",
        vline=_entropy(sums_obs),
        w=900,
    )

    # z-score summary of several metrics
    def _z(obs: float, arr: list[float]) -> float:
        mu = float(np.mean(arr))
        sd = float(np.std(arr) + 1e-9)
        return (obs - mu) / sd

    metrics = [
        ("consec pairs", _z(float(np.mean(consec_obs)), mc_consec_mean)),
        ("repeats prev", _z(float(np.mean(repeats_obs)), mc_repeat_mean)),
        ("sum", _z(float(np.mean(sums_obs)), mc_sum_mean)),
        ("odd count", _z(float(np.mean(odds_obs)), mc_odd_mean)),
        ("low count", _z(float(np.mean(lows_obs)), mc_low_mean)),
    ]
    bar_chart(
        np.arange(1, len(metrics) + 1),
        np.array([m[1] for m in metrics], dtype=float),
        title="Observed minus expected (z-scores) for key metrics",
        xlabel="metric",
        ylabel="z-score",
        out=slug_dir / "metric-zscores.svg",
        x_labels=[m[0] for m in metrics],
        label_every=1,
        w=980,
    )

    # -----------------------------
    # 5) Early vs late era (concept drift)
    # -----------------------------
    mid = n_draws // 2
    early = draws[:mid]
    late = draws[mid:]

    def _counts(ds: list[list[int]]) -> np.ndarray:
        c = np.zeros(n_balls, dtype=float)
        for d in ds:
            for v in d:
                c[v - 1] += 1
        return c

    early_counts = _counts(early)
    late_counts = _counts(late)
    drift = late_counts - early_counts
    highlight_drift = int(x[np.argmax(np.abs(drift))])

    bar_chart(
        x,
        drift,
        title=f"Frequency drift: late − early (n={len(late)} vs {len(early)})",
        xlabel="ball",
        ylabel="count difference",
        highlight_x=highlight_drift,
        out=slug_dir / "frequency-drift-halves.svg",
        w=1050,
    )


def plot_sampling_controls() -> None:
    vocab_n = 100
    base_logits = np.linspace(2.5, -3.0, vocab_n)

    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        ez = np.exp(z)
        return ez / ez.sum()

    temps = np.array([0.2, 0.5, 0.8, 1.0, 1.3, 1.7])
    ent = []
    for T in temps:
        p = softmax(base_logits / T)
        ent.append(float(-(p * np.log2(p + 1e-18)).sum()))

    _line_chart(
        temps,
        [(np.array(ent), "entropy", "#2c7fb8")],
        title="Entropy increases with temperature (toy)",
        xlabel="temperature",
        ylabel="entropy (bits)",
        out=PUBLIC / "2026-01-31-sampling-controls" / "entropy-vs-temperature.svg",
        w=820,
    )

    p = softmax(base_logits)
    order = np.argsort(-p)
    cdf = np.cumsum(p[order])

    ks = np.arange(1, vocab_n + 1)
    _line_chart(
        ks,
        [(cdf, "CDF", "#41b6c4")],
        title="Nucleus sampling: how many tokens survive top-p? (toy)",
        xlabel="#tokens kept (rank)",
        ylabel="cumulative probability",
        out=PUBLIC / "2026-01-31-sampling-controls" / "top-p-cdf.svg",
        w=900,
    )


def plot_attention_toy() -> None:
    rng = np.random.default_rng(42)
    n_tokens = 12
    d = 16

    X = rng.normal(size=(n_tokens, d))
    Wq = rng.normal(size=(d, d)) / math.sqrt(d)
    Wk = rng.normal(size=(d, d)) / math.sqrt(d)

    Q = X @ Wq
    K = X @ Wk
    scores = (Q @ K.T) / math.sqrt(d)

    # row-wise softmax
    scores = scores - scores.max(axis=1, keepdims=True)
    A = np.exp(scores)
    A = A / A.sum(axis=1, keepdims=True)

    _heatmap(
        A,
        title="Toy self-attention matrix (softmax(QKᵀ/√d))",
        out=PUBLIC / "2026-02-01-attention-toy" / "attention-heatmap.svg",
    )

    row_ent = np.array([float(-(r * np.log2(r + 1e-18)).sum()) for r in A])
    _bar_chart(
        np.arange(n_tokens),
        row_ent,
        title="Attention row entropy (lower ⇒ more peaky)",
        xlabel="query token index",
        ylabel="entropy (bits)",
        out=PUBLIC / "2026-02-01-attention-toy" / "attention-row-entropy.svg",
        w=920,
    )


def plot_token_budgets() -> None:
    """Token budget plots based on real measurements via the OpenAI API.

    Source: site/analysis/data/chars_per_token.csv
    Generated by: site/analysis/collect_real_data.py
    """

    import csv

    data_csv = ROOT / "analysis" / "data" / "chars_per_token.csv"
    if not data_csv.exists():
        raise RuntimeError(f"Missing {data_csv}. Run: python site/analysis/collect_real_data.py")

    ratios = []
    tokens = []
    with data_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("ok") != "ok":
                continue
            try:
                t = int(row["input_tokens"])
                c = int(row["chars"])
                tokens.append(t)
                ratios.append(c / max(1, t))
            except Exception:
                pass

    if not ratios:
        raise RuntimeError("No usable rows in chars_per_token.csv")

    _histogram(
        np.array(ratios, dtype=float),
        bins=18,
        title="Chars per token (real prompts; measured via API usage)",
        xlabel="chars/token",
        ylabel="samples",
        out=PUBLIC / "2026-02-02-token-budgets" / "chars-per-token.svg",
        vline=4.0,
        w=900,
    )

    # Real (small-sample) prompt token counts from the same measurement pass
    window = 32_000
    _histogram(
        np.array(tokens, dtype=float),
        bins=14,
        title="Prompt token counts in the measurement set (real, small sample)",
        xlabel="input tokens",
        ylabel="requests",
        out=PUBLIC / "2026-02-02-token-budgets" / "prompt-size-hist.svg",
        vline=float(window),
        w=900,
    )


def plot_voice_latency() -> None:
    """Voice latency plot based on real API call timings.

    Source: site/analysis/data/voice_latency_samples.csv
    Generated by: site/analysis/collect_real_data.py

    Note: this measures request-level latency (non-streaming), not true token-level TTFB.
    """

    import csv

    data_csv = ROOT / "analysis" / "data" / "voice_latency_samples.csv"
    if not data_csv.exists():
        raise RuntimeError(f"Missing {data_csv}. Run: python site/analysis/collect_real_data.py")

    by_setting: dict[str, list[float]] = {}
    with data_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("ok") != "ok":
                continue
            setting = row.get("setting") or "default"
            try:
                by_setting.setdefault(setting, []).append(float(row["took_ms"]))
            except Exception:
                pass

    # Quantile curves per setting
    q = np.linspace(0.05, 0.95, 60)
    series = []
    colors = ["#2c7fb8", "#41b6c4", "#7fcdbb", "#d7301f"]
    for i, (setting, vals) in enumerate(sorted(by_setting.items())):
        if len(vals) < 5:
            continue
        series.append((np.quantile(np.array(vals, dtype=float), q), f"{setting} p(q)", colors[i % len(colors)]))

    _line_chart(
        q,
        series,
        title="API call latency quantiles (real measurements)",
        xlabel="quantile q",
        ylabel="ms at quantile",
        out=PUBLIC / "2026-02-03-voice-latency" / "component-quantiles.svg",
        w=900,
    )

    all_vals = np.array([v for vs in by_setting.values() for v in vs], dtype=float)
    _histogram(
        all_vals,
        bins=24,
        title="End-to-first-byte-ish latency (real measurements)",
        xlabel="ms",
        ylabel="requests",
        out=PUBLIC / "2026-02-03-voice-latency" / "end-to-first-audio-hist.svg",
        w=900,
    )


def plot_rag_eval() -> None:
    rng = np.random.default_rng(5)
    n_queries = 250
    kmax = 10
    answerable = rng.random(n_queries) < 0.72

    precision_at_k = []
    recall_at_k = []
    ks = np.arange(1, kmax + 1)

    for k in ks:
        precs, recs = [], []
        for a in answerable:
            rel_total = int(rng.integers(1, 6)) if a else 0
            if a:
                rel_in_topk = int(rng.binomial(int(k), p=0.28))
                rel_in_topk = min(rel_in_topk, rel_total)
            else:
                rel_in_topk = int(rng.binomial(int(k), p=0.02))
            precs.append(rel_in_topk / int(k))
            recs.append((rel_in_topk / rel_total) if rel_total > 0 else 0.0)
        precision_at_k.append(float(np.mean(precs)))
        recall_at_k.append(float(np.mean(recs)))

    _line_chart(
        ks,
        [
            (np.array(precision_at_k), "precision@k", "#2c7fb8"),
            (np.array(recall_at_k), "recall@k", "#d7301f"),
        ],
        title="RAG retrieval: precision@k vs recall@k (synthetic)",
        xlabel="k",
        ylabel="metric",
        out=PUBLIC / "2026-02-04-rag-eval" / "precision-recall-at-k.svg",
        w=900,
    )

    score = rng.normal(loc=0.7, scale=0.15, size=n_queries)
    score[~answerable] = rng.normal(loc=0.45, scale=0.18, size=(~answerable).sum())

    thresholds = np.linspace(0.1, 0.95, 30)
    tpr, fpr = [], []
    for t in thresholds:
        pred = score >= t
        tp = np.sum(pred & answerable)
        fp = np.sum(pred & ~answerable)
        fn = np.sum(~pred & answerable)
        tn = np.sum(~pred & ~answerable)
        tpr.append(tp / (tp + fn + 1e-9))
        fpr.append(fp / (fp + tn + 1e-9))

    _line_chart(
        np.array(fpr),
        [(np.array(tpr), "TPR", "#41b6c4")],
        title="Answerability gating: ROC-like trade-off (synthetic)",
        xlabel="false positive rate",
        ylabel="true positive rate",
        out=PUBLIC / "2026-02-04-rag-eval" / "answerability-roc.svg",
        w=700,
        h=520,
    )


def plot_embedding_drift() -> None:
    rng = np.random.default_rng(21)
    n = 800
    d = 32

    base = rng.normal(size=(n, d))
    shift = rng.normal(size=(1, d)) * 0.25
    drift = base * rng.normal(loc=1.0, scale=0.05, size=(1, d)) + shift

    X = np.vstack([base, drift])
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    z0, z1 = Z[:n], Z[n:]

    # simple scatter as SVG: plot 2D points
    def scatter_svg(a: np.ndarray, b: np.ndarray, title: str, out: Path) -> None:
        w, h = 700, 520
        pad_l, pad_r, pad_t, pad_b = 70, 20, 55, 60
        iw, ih = w - pad_l - pad_r, h - pad_t - pad_b

        mnx = float(min(a[:, 0].min(), b[:, 0].min()))
        mxx = float(max(a[:, 0].max(), b[:, 0].max()))
        mny = float(min(a[:, 1].min(), b[:, 1].min()))
        mxy = float(max(a[:, 1].max(), b[:, 1].max()))

        def X(v: float) -> float:
            return pad_l + (v - mnx) / (mxx - mnx + 1e-9) * iw

        def Y(v: float) -> float:
            return pad_t + (1 - (v - mny) / (mxy - mny + 1e-9)) * ih

        s = []
        s += _svg_header(w, h)
        s.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-size="16">{title}</text>')
        s.append(f'<rect x="{pad_l}" y="{pad_t}" width="{iw}" height="{ih}" fill="none" stroke="#333"/>')

        # points
        rng2 = np.random.default_rng(0)
        # downsample to keep SVG size reasonable
        idx_a = rng2.choice(len(a), size=450, replace=False)
        idx_b = rng2.choice(len(b), size=450, replace=False)

        for p in a[idx_a]:
            s.append(f'<circle cx="{_fmt(X(float(p[0])))}" cy="{_fmt(Y(float(p[1])))}" r="2" fill="#2c7fb8" opacity="0.35"/>')
        for p in b[idx_b]:
            s.append(f'<circle cx="{_fmt(X(float(p[0])))}" cy="{_fmt(Y(float(p[1])))}" r="2" fill="#d7301f" opacity="0.35"/>')

        # legend
        s.append(f'<rect x="{pad_l+10}" y="{pad_t+10}" width="150" height="44" fill="#fff" opacity="0.9" stroke="#ddd"/>')
        s.append(f'<circle cx="{pad_l+24}" cy="{pad_t+26}" r="4" fill="#2c7fb8" opacity="0.8"/>')
        s.append(f'<text x="{pad_l+36}" y="{pad_t+30}" font-size="12">week 0</text>')
        s.append(f'<circle cx="{pad_l+24}" cy="{pad_t+44}" r="4" fill="#d7301f" opacity="0.8"/>')
        s.append(f'<text x="{pad_l+36}" y="{pad_t+48}" font-size="12">week 4</text>')

        s += _svg_footer()
        _write_svg(out, s)

    scatter_svg(z0, z1, "Embedding drift visualized (PCA projection)", PUBLIC / "2026-02-05-embedding-drift" / "embedding-drift-pca.svg")

    _histogram(
        z0[:, 0],
        bins=28,
        title="Week 0: PC1 distribution (baseline)",
        xlabel="PC1",
        ylabel="count",
        out=PUBLIC / "2026-02-05-embedding-drift" / "pc1-week0.svg",
        w=900,
    )
    _histogram(
        z1[:, 0],
        bins=28,
        title="Week 4: PC1 distribution (drifted)",
        xlabel="PC1",
        ylabel="count",
        out=PUBLIC / "2026-02-05-embedding-drift" / "pc1-week4.svg",
        w=900,
    )


def plot_turn_taking_vad() -> None:
    rng = np.random.default_rng(13)
    n = 5000
    speech = rng.beta(a=6, b=2, size=n)
    nonspeech = rng.beta(a=1.5, b=7, size=n)

    thresholds = np.linspace(0.05, 0.95, 50)
    tpr = np.array([(speech >= t).mean() for t in thresholds])
    fpr = np.array([(nonspeech >= t).mean() for t in thresholds])

    _line_chart(
        fpr,
        [(tpr, "ROC", "#41b6c4")],
        title="VAD threshold trade-off (synthetic ROC)",
        xlabel="false positive rate",
        ylabel="true positive rate",
        out=PUBLIC / "2026-02-06-turn-taking" / "vad-roc.svg",
        w=700,
        h=520,
    )

    _histogram(
        speech,
        bins=35,
        title="VAD scores: speech frames (synthetic)",
        xlabel="score",
        ylabel="frames",
        out=PUBLIC / "2026-02-06-turn-taking" / "vad-speech-hist.svg",
    )

    _histogram(
        nonspeech,
        bins=35,
        title="VAD scores: non-speech frames (synthetic)",
        xlabel="score",
        ylabel="frames",
        out=PUBLIC / "2026-02-06-turn-taking" / "vad-nonspeech-hist.svg",
    )


def plot_asr_wer() -> None:
    rng = np.random.default_rng(9)
    snr = np.linspace(0, 30, 61)

    def wer_curve(base: float, slope: float) -> np.ndarray:
        return np.clip(base * np.exp(-slope * snr) + 0.06, 0, 1)

    wer_general = wer_curve(0.55, 0.08) + rng.normal(0, 0.01, size=snr.size)
    wer_domain = wer_curve(0.65, 0.07) + 0.05 + rng.normal(0, 0.01, size=snr.size)

    _line_chart(
        snr,
        [
            (wer_general, "general", "#2c7fb8"),
            (wer_domain, "domain-shifted", "#d7301f"),
        ],
        title="WER vs noise: domain shift adds an irreducible floor (synthetic)",
        xlabel="SNR (dB)",
        ylabel="WER",
        out=PUBLIC / "2026-02-07-asr-wer" / "wer-vs-snr.svg",
        w=900,
    )

    utter_wer = rng.beta(a=1.2, b=6.0, size=2500)
    utter_wer[:120] = rng.beta(a=4.0, b=1.3, size=120)

    _histogram(
        utter_wer,
        bins=30,
        title="Utterance-level WER is heavy-tailed (synthetic)",
        xlabel="utterance WER",
        ylabel="count",
        out=PUBLIC / "2026-02-07-asr-wer" / "utterance-wer-hist.svg",
        w=900,
    )


def plot_tts_durations() -> None:
    """TTS duration plot based on real audio samples.

    Source: site/analysis/data/tts_durations.csv
    Generated by: site/analysis/collect_real_data.py
    """

    import csv

    data_csv = ROOT / "analysis" / "data" / "tts_durations.csv"
    if not data_csv.exists():
        raise RuntimeError(f"Missing {data_csv}. Run: python site/analysis/collect_real_data.py")

    durs = []
    labels = []
    with data_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("ok") != "ok":
                continue
            try:
                durs.append(float(row["duration_s"]))
                labels.append(str(row.get("sample_id", "")))
            except Exception:
                pass

    if not durs:
        raise RuntimeError("No usable rows in tts_durations.csv")

    # Create a small heatmap (1 x N) of durations (seconds)
    mat = np.array([durs], dtype=float)
    _heatmap(
        mat,
        title="TTS sample durations (seconds, real measurements)",
        out=PUBLIC / "2026-02-08-tts-stack" / "duration-percentiles-heatmap.svg",
        w=820,
        h=260,
    )


def plot_gpus_vs_tpus() -> None:
    """Plots for the GPU/TPU/accelerator post.

    Notes:
    - Numbers are illustrative (order-of-magnitude) to show *shapes* and trade-offs.
    - Avoid vendor-specific claims unless the point is architectural.
    """

    # 1) Memory hierarchy diagram (GPU-ish)
    w, h = 980, 520
    s: list[str] = []
    s += _svg_header(w, h)
    s.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-size="16">Accelerator memory hierarchy (schematic)</text>')

    x = 90
    box_w, box_h = 800, 64
    gap = 22

    levels = [
        ("Registers", "~10s of TB/s effective, ~1 cycle", "#f7fbff", "#2c7fb8"),
        ("Shared memory / L1", "~10–100 TB/s, ~tens of cycles", "#f7fcf5", "#41ab5d"),
        ("L2 cache", "~few TB/s, ~100ns-ish", "#fffff5", "#d95f0e"),
        ("HBM (device DRAM)", "~1–3 TB/s, ~hundreds of ns", "#fff5f0", "#cb181d"),
        ("Host RAM / Storage", "~10–100+ GB/s, ~µs–ms", "#f7f7f7", "#636363"),
    ]

    y0 = 70
    for i, (lab, sub, fill, stroke) in enumerate(levels):
        y = y0 + i * (box_h + gap)
        _box(s, x, y, box_w, box_h, label=lab, sub=sub, fill=fill, stroke=stroke)
        if i < len(levels) - 1:
            _arrow(s, x + box_w / 2, y + box_h, x + box_w / 2, y + box_h + gap - 6, color="#333")

    s.append(
        '<text x="90" y="488" font-size="11" fill="#444">Schematic: bandwidth drops and latency rises as you go down the stack. Your kernel\'s job is to stay near the top.</text>'
    )
    s += _svg_footer()
    _write_svg(PUBLIC / "2026-02-09-gpus-vs-tpus" / "memory-hierarchy.svg", s)

    # 2) Throughput vs batch size (toy serving curve)
    batch = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    # saturating curve: throughput ~ a*(1-exp(-b*batch)) then taper due to memory/overheads
    thr = 2200 * (1 - np.exp(-batch / 18.0))
    thr = thr * (1 - 0.08 * np.log2(batch) / np.log2(128))

    _line_chart(
        batch,
        [(thr, "tokens/s (system)", "#2c7fb8")],
        title="Batching: throughput saturates (and then bends)",
        xlabel="batch size",
        ylabel="throughput (tokens/s)",
        out=PUBLIC / "2026-02-09-gpus-vs-tpus" / "throughput-vs-batch.svg",
        w=900,
    )

    # 3) Interconnect comparison (bandwidth; rough, directional)
    links = ["PCIe", "NVLink", "Ethernet"]
    # normalized GB/s (per direction, illustrative)
    bw = np.array([64, 300, 25], dtype=float)
    x = np.arange(1, len(links) + 1)

    # Use bar chart with x=1..N then label in the post.
    _bar_chart(
        x,
        bw,
        title="Interconnect bandwidth (illustrative, per direction)",
        xlabel="link type (1=PCIe, 2=NVLink, 3=Ethernet)",
        ylabel="GB/s",
        out=PUBLIC / "2026-02-09-gpus-vs-tpus" / "interconnect-bandwidth.svg",
        w=980,
    )


def plot_local_llm_setup() -> None:
    """Plots for the local LLM setup post (VRAM math + serving curves)."""

    # 1) KV cache memory vs context length for a few precisions (toy but close-ish)
    # KV bytes ~ 2 * n_layers * hidden * seq * bytes_per_elem (K and V)
    # We'll pick a representative model config to illustrate scaling.
    n_layers = 32
    hidden = 4096

    seq = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768])

    def kv_gb(bytes_per_elem: float) -> np.ndarray:
        kv_bytes = 2 * n_layers * hidden * seq * bytes_per_elem
        return kv_bytes / (1024**3)

    kv_fp16 = kv_gb(2.0)
    kv_int8 = kv_gb(1.0)
    kv_q4 = kv_gb(0.5)  # not literal, but good intuition for compressed cache variants

    _line_chart(
        seq,
        [
            (kv_fp16, "KV cache ~FP16", "#d7301f"),
            (kv_int8, "KV cache ~INT8", "#2c7fb8"),
            (kv_q4, "KV cache ~4-bit", "#41b6c4"),
        ],
        title="KV cache grows linearly with context length",
        xlabel="context length (tokens)",
        ylabel="KV cache size (GiB)",
        out=PUBLIC / "2026-02-10-local-llm-setup" / "kv-cache-vs-context.svg",
        w=900,
    )

    # 2) VRAM budget breakdown (stacked bars)
    # Example budgets for a few setups.
    labels = ["7B Q4", "13B Q4", "7B FP16"]
    # segments: weights, KV cache (8k), activations/overhead
    weights = np.array([4.2, 7.4, 13.5])
    kv = np.array([2.1, 2.8, 4.2])
    overhead = np.array([1.3, 1.6, 2.0])

    _stacked_bar_chart(
        labels,
        stacks=np.vstack([weights, kv, overhead]).T.tolist(),
        stack_labels=["weights", "KV cache", "runtime overhead"],
        colors=["#2c7fb8", "#41b6c4", "#bdbdbd"],
        title="VRAM budgeting: what actually occupies memory",
        ylabel="GiB",
        out=PUBLIC / "2026-02-10-local-llm-setup" / "vram-breakdown.svg",
        w=980,
    )

    # 3) Throughput vs batch size vs latency proxy
    batch = np.array([1, 2, 4, 8, 16, 32, 64])
    thr = 550 * (1 - np.exp(-batch / 10.0)) * (1 - 0.06 * np.log2(batch) / np.log2(64))
    # latency proxy (ms per token) grows after a point due to queueing and cache pressure
    ms_per_tok = 28 + 6 * np.log2(batch) + 0.9 * (batch / 16) ** 1.4

    _line_chart(
        batch,
        [
            (thr, "throughput (tok/s)", "#2c7fb8"),
            (ms_per_tok, "latency proxy (ms/token)", "#d7301f"),
        ],
        title="Batching shifts you on the throughput/latency frontier (toy)",
        xlabel="batch size",
        ylabel="(different units; compare shapes)",
        out=PUBLIC / "2026-02-10-local-llm-setup" / "throughput-latency-frontier.svg",
        w=900,
    )


@dataclass
class Task:
    name: str
    fn: callable


def main() -> None:
    tasks = [
        Task("lottery_fairness", plot_lottery_fairness),
        Task("sampling_controls", plot_sampling_controls),
        Task("attention_toy", plot_attention_toy),
        Task("token_budgets", plot_token_budgets),
        Task("voice_latency", plot_voice_latency),
        Task("rag_eval", plot_rag_eval),
        Task("embedding_drift", plot_embedding_drift),
        Task("turn_taking_vad", plot_turn_taking_vad),
        Task("asr_wer", plot_asr_wer),
        Task("tts_durations", plot_tts_durations),
        Task("gpus_vs_tpus", plot_gpus_vs_tpus),
        Task("local_llm_setup", plot_local_llm_setup),
    ]

    for t in tasks:
        print(f"[generate] {t.name}")
        t.fn()

    print(f"\nWrote SVG plots under: {PUBLIC}")


if __name__ == "__main__":
    main()
