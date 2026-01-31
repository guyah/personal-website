"""Matplotlib plotting helpers for analysis assets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


_DPI = 100


def _prep_fig(w: int, h: int) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w / _DPI, h / _DPI), dpi=_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def _finalize(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg")
    plt.close(fig)


def text_panel(
    *,
    title: str,
    lines: Iterable[tuple[str, int, str] | str],
    out: Path,
    w: int = 900,
    h: int = 220,
    footer: tuple[str, int, str] | str | None = None,
) -> None:
    fig, ax = _prep_fig(w, h)
    ax.axis("off")

    ax.text(0.5, 0.9, title, ha="center", va="top", fontsize=14, color="#111", transform=ax.transAxes)

    lines_list = list(lines)
    top = 0.68
    step = 0.12 if len(lines_list) <= 4 else 0.1
    for i, item in enumerate(lines_list):
        if isinstance(item, tuple):
            text, size, color = item
        else:
            text, size, color = item, 12, "#111"
        ax.text(0.04, top - i * step, text, ha="left", va="top", fontsize=size, color=color, transform=ax.transAxes)

    if footer is not None:
        if isinstance(footer, tuple):
            text, size, color = footer
        else:
            text, size, color = footer, 11, "#666"
        ax.text(0.04, 0.08, text, ha="left", va="bottom", fontsize=size, color=color, transform=ax.transAxes)

    _finalize(fig, out)


def bar_chart(
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
    x = np.asarray(x)
    y = np.asarray(y)
    fig, ax = _prep_fig(w, h)

    if x_labels:
        positions = np.arange(len(x_labels))
        colors = ["#2c7fb8"] * len(positions)
        if highlight_x is not None:
            matches = np.where(x == highlight_x)[0]
            if len(matches) > 0:
                colors[int(matches[0])] = "#d7301f"
        ax.bar(positions, y, color=colors, alpha=0.9)
        ticks = list(range(0, len(x_labels), max(1, label_every)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([x_labels[i] for i in ticks], fontsize=10)
    else:
        colors = ["#2c7fb8"] * len(x)
        if highlight_x is not None:
            matches = np.where(x == highlight_x)[0]
            if len(matches) > 0:
                colors[int(matches[0])] = "#d7301f"
        ax.bar(x, y, color=colors, alpha=0.9, width=0.9)
        if label_every > 1:
            ticks = x[:: max(1, label_every)]
            ax.set_xticks(ticks)

    ax.axhline(0, color="#111", linewidth=1, alpha=0.6)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", color="#eee")

    fig.tight_layout()
    _finalize(fig, out)


def histogram(
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
    fig, ax = _prep_fig(w, h)

    ax.hist(values, bins=bins, color="#41b6c4", alpha=0.9, edgecolor="white")
    if vline is not None:
        ax.axvline(vline, color="#d7301f", linewidth=2)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", color="#eee")

    fig.tight_layout()
    _finalize(fig, out)


def heatmap(
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
    fig, ax = _prep_fig(w, h)

    im = ax.imshow(mat, cmap="viridis", aspect="auto", origin="upper")

    if x_labels:
        xticks = np.arange(0, len(x_labels), max(1, label_every))
        ax.set_xticks(xticks)
        ax.set_xticklabels([x_labels[i] for i in xticks], fontsize=8)
    else:
        ax.set_xticks([])

    if y_labels:
        yticks = np.arange(0, len(y_labels), max(1, label_every))
        ax.set_yticks(yticks)
        ax.set_yticklabels([y_labels[i] for i in yticks], fontsize=8)
    else:
        ax.set_yticks([])

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    fig.tight_layout()
    _finalize(fig, out)
