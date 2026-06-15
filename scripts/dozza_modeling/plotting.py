"""Utility di plotting per figure Dozza pronte per il paper."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


PAPER_FIGURE_FORMATS = ("png", "pdf")


def configure_paper_plots() -> None:
    """Imposta default Matplotlib sobri per report e paper."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.grid": False,
        }
    )


def save_figure(
    fig: plt.Figure,
    output_path: Path | str,
    dpi: int = 300,
    formats: Iterable[str] = PAPER_FIGURE_FORMATS,
) -> Path:
    """Salva una figura in PNG e PDF con lo stesso nome base."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    requested_format = path.suffix.lstrip(".").lower() or "png"
    ordered_formats = []
    for fmt in (requested_format, *formats):
        fmt = fmt.lower().lstrip(".")
        if fmt and fmt not in ordered_formats:
            ordered_formats.append(fmt)
    for fmt in ordered_formats:
        save_path = stem.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if fmt in {"png", "jpg", "jpeg", "tif", "tiff", "webp"}:
            kwargs["dpi"] = dpi
        fig.savefig(save_path, **kwargs)
    return path
