"""Fetch Lebanese Loto (6/42) draw history and normalize to CSV.

Data source:
- https://www.lebanon-lotto.com/ (NOT an official LLDJ endpoint; they explicitly state this.)

Why curl?
- Python SSL certs on this machine can be brittle; curl works reliably.

Outputs:
- site/analysis/data/lebanon_loto_draws.csv

Schema:
- draw_id,date,n1,n2,n3,n4,n5,n6,bonus

Run from repo root:
  python3 site/analysis/fetch_lebanon_loto.py
"""

from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # site/
DATA_DIR = ROOT / "analysis" / "data"
CACHE_DIR = DATA_DIR / "cache" / "lebanon-lotto"


def _curl(url: str) -> str:
    # -L follow redirects, -f fail on non-2xx, -s silent, -S show errors
    # keep it bounded so a single slow page doesn't stall the whole pipeline
    out = subprocess.check_output(
        [
            "curl",
            "-fsSL",
            "-L",
            "--max-time",
            "20",
            "--retry",
            "2",
            "--retry-delay",
            "1",
            url,
        ],
        text=True,
    )
    return out


def _cached_get(url: str, *, cache_name: str) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / cache_name
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8", errors="ignore")
    html = _curl(url)
    path.write_text(html, encoding="utf-8")
    return html


DRAW_ID_RE = re.compile(r"lebanese-loto-results/draw-number/(\d+)\.php")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
# Main balls appear as two-digit numbers in the image filename.
MAIN_BALL_RE = re.compile(r"/images/lotto_balls_new/[^\"']*?_in_lebanon_(\d{2})\.gif")
# Bonus/complimentary ball appears in a different folder.
BONUS_BALL_RE = re.compile(r"/images/lotto_balls/[^\"']*?_in_lebanon_(\d{2})\.gif")


@dataclass(frozen=True)
class Draw:
    draw_id: int
    date: str
    nums: list[int]  # len 6
    bonus: int | None


def _parse_draw_page(draw_id: int) -> Draw:
    url = f"https://www.lebanon-lotto.com/lebanese-loto-results/draw-number/{draw_id}.php"
    html = _cached_get(url, cache_name=f"draw-{draw_id}.html")

    # Date: take first YYYY-MM-DD on the page (appears many times; first is usually correct).
    m = DATE_RE.search(html)
    if not m:
        raise ValueError(f"No date found for draw {draw_id}")
    date = m.group(1)

    # Main numbers: take first 6 occurrences from the 'lotto_balls_new' images.
    nums = [int(x) for x in MAIN_BALL_RE.findall(html)[:6]]
    if len(nums) != 6:
        raise ValueError(f"Expected 6 main numbers for draw {draw_id}, got {len(nums)}")

    # Bonus: sometimes present, sometimes '--'. We attempt to parse it.
    bonus = None
    bonus_candidates = [int(x) for x in BONUS_BALL_RE.findall(html)]
    # Heuristic: if exactly one 2-digit number shows up a bunch in bonus images on the page, take the most common.
    if bonus_candidates:
        from collections import Counter

        bonus = Counter(bonus_candidates).most_common(1)[0][0]

    return Draw(draw_id=draw_id, date=date, nums=nums, bonus=bonus)


def _list_draw_ids() -> list[int]:
    # The past results list page can be filtered by year. We crawl all available years.
    # It goes back to 2002 (relaunch).
    draw_ids: set[int] = set()
    for year in range(2002, 2027):
        print(f"[fetch] year {year} …")
        url = f"https://www.lebanon-lotto.com/past_results_list.php?pastyearsresults={year}"
        html = _cached_get(url, cache_name=f"past-{year}.html")
        for s in DRAW_ID_RE.findall(html):
            draw_ids.add(int(s))

    ids = sorted(draw_ids)
    print(f"[fetch] found {len(ids)} draw ids")
    return ids


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = DATA_DIR / "lebanon_loto_draws.csv"

    draw_ids = _list_draw_ids()
    if not draw_ids:
        raise RuntimeError("No draw ids found. Source HTML likely changed.")

    # Optional: limit for faster iteration/debugging
    import os

    max_draws = os.environ.get("MAX_DRAWS")
    if max_draws:
        draw_ids = draw_ids[-int(max_draws) :]
        print(f"[fetch] limiting to last {len(draw_ids)} draws (MAX_DRAWS={max_draws})")

    rows: list[Draw] = []
    errors: list[str] = []

    total = len(draw_ids)
    for i, did in enumerate(draw_ids, start=1):
        try:
            rows.append(_parse_draw_page(did))
        except Exception as e:
            errors.append(f"draw {did}: {e}")

        if i % 25 == 0 or i == total:
            print(f"[fetch] parsed {i}/{total} draws")

    # Write CSV
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["draw_id", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"])
        for d in rows:
            w.writerow([
                d.draw_id,
                d.date,
                *d.nums,
                "" if d.bonus is None else d.bonus,
            ])

    print(f"wrote {out_csv} ({len(rows)} rows)")
    if errors:
        err_path = DATA_DIR / "lebanon_loto_errors.txt"
        err_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"warnings: {len(errors)} draws failed to parse (see {err_path})")


if __name__ == "__main__":
    main()
