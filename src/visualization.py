from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def _prepare_path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def compute_observed_expansion_mask(built_base: np.ndarray, built_next: np.ndarray) -> np.ndarray:
    """Observed expansion is transition: non-built at base year -> built at next year."""
    if built_base.shape != built_next.shape:
        raise ValueError("built_base and built_next must have the same shape.")
    return (built_base == 0) & (built_next == 1)


def save_observed_expansion_map(
    built_base: np.ndarray,
    built_next: np.ndarray,
    path: str | Path,
    title: str = "Observed Urban Expansion (2010-2024)",
) -> None:
    """Save binary observed expansion map for a period."""
    out = _prepare_path(path)

    observed = compute_observed_expansion_mask(built_base, built_next)
    valid = np.isfinite(built_base) & np.isfinite(built_next)
    display = np.where(valid, observed.astype(np.float32), np.nan)

    cmap = matplotlib.colors.ListedColormap(["#e6e6e6", "#111111"])
    cmap.set_bad(color="#ffffff")

    fig, ax = plt.subplots(figsize=(8.8, 6.8), facecolor="white", constrained_layout=True)
    ax.imshow(display, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    legend_items = [
        Patch(facecolor="#111111", edgecolor="none", label="Observed expansion"),
        Patch(facecolor="#e6e6e6", edgecolor="none", label="No expansion"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=True, fontsize=9)

    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_training_concept_figure(path: str | Path) -> None:
    """LEGACY helper: not used by clean run_pipeline.py outputs."""
    out = _prepare_path(path)

    fig, ax = plt.subplots(figsize=(9.2, 5.8), facecolor="white", constrained_layout=True)
    ax.axis("off")

    ax.text(
        0.5,
        0.93,
        "Urban Expansion Transition Model (Training: 2000 -> 2010)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    box = dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#666666")
    ax.text(
        0.18,
        0.62,
        "Candidate Space\nBuilt_2000 == 0",
        ha="center",
        va="center",
        fontsize=12,
        bbox=box,
    )
    ax.text(
        0.50,
        0.62,
        "Target Y\n1: non-built(2000) -> built(2010)\n0: non-built(2000) -> non-built(2010)",
        ha="center",
        va="center",
        fontsize=11,
        bbox=box,
    )
    ax.text(
        0.82,
        0.62,
        "Predictors X\n1) distance_to_built_2000\n2) distance_to_roads",
        ha="center",
        va="center",
        fontsize=11,
        bbox=box,
    )

    arrow = dict(arrowstyle="->", lw=1.8, color="#333333")
    ax.annotate("", xy=(0.39, 0.62), xytext=(0.27, 0.62), arrowprops=arrow)
    ax.annotate("", xy=(0.71, 0.62), xytext=(0.61, 0.62), arrowprops=arrow)

    ax.text(
        0.5,
        0.28,
        "Important: Built_2000 is used only to define candidate pixels,\n"
        "not as a predictor feature.",
        ha="center",
        va="center",
        fontsize=11,
        color="#222222",
    )

    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_final_validation_confusion_map(
    pred_prob_map: np.ndarray,
    actual_change_mask: np.ndarray,
    candidate_mask: np.ndarray,
    built_mask: np.ndarray,
    threshold: float,
    path: str | Path,
    title: str = "Validation Confusion Map (2010 -> 2024)",
) -> None:
    """Save one validation confusion map (TP/FP/FN/TN plus excluded built area)."""
    out = _prepare_path(path)

    candidate = candidate_mask.astype(bool)
    built = built_mask.astype(bool)
    actual = actual_change_mask.astype(bool)
    pred_positive = candidate & np.isfinite(pred_prob_map) & (pred_prob_map >= threshold)

    tp = candidate & pred_positive & actual
    fp = candidate & pred_positive & (~actual)
    fn = candidate & (~pred_positive) & actual
    tn = candidate & (~pred_positive) & (~actual)

    rgb = np.ones((*candidate.shape, 3), dtype=np.float32)
    rgb[built] = np.array([0.23, 0.23, 0.23])
    rgb[tn] = np.array([0.92, 0.92, 0.92])
    rgb[fp] = np.array([0.90, 0.34, 0.17])
    rgb[fn] = np.array([0.25, 0.48, 0.85])
    rgb[tp] = np.array([0.14, 0.64, 0.25])

    fig, ax = plt.subplots(figsize=(8.8, 6.8), facecolor="white", constrained_layout=True)
    ax.imshow(rgb)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    legend_items = [
        Patch(facecolor="#24a440", edgecolor="none", label="True Positive"),
        Patch(facecolor="#e65a2b", edgecolor="none", label="False Positive"),
        Patch(facecolor="#3f7ad9", edgecolor="none", label="False Negative"),
        Patch(facecolor="#ebebeb", edgecolor="none", label="True Negative"),
        Patch(facecolor="#3b3b3b", edgecolor="none", label="Built in base year (excluded)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=True, fontsize=9)

    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_final_prediction_probability_map(
    pred_prob_map: np.ndarray,
    candidate_mask: np.ndarray,
    built_mask: np.ndarray,
    path: str | Path,
    title: str = "Final Prediction Probability Map",
) -> None:
    """Save one final prediction probability map for future expansion."""
    out = _prepare_path(path)

    fig, ax = plt.subplots(figsize=(8.8, 6.8), facecolor="white", constrained_layout=True)

    built_layer = np.where(built_mask, 1.0, np.nan)
    built_cmap = matplotlib.colors.ListedColormap(["#3b3b3b"])
    built_cmap.set_bad(alpha=0.0)
    ax.imshow(built_layer, cmap=built_cmap, vmin=0, vmax=1)

    prob_layer = np.where(candidate_mask, pred_prob_map, np.nan)
    prob_cmap = plt.get_cmap("YlOrRd").copy()
    prob_cmap.set_bad(alpha=0.0)
    im = ax.imshow(prob_layer, cmap=prob_cmap, vmin=0.0, vmax=1.0)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Predicted expansion probability")
    legend_items = [
        Patch(facecolor="#3b3b3b", edgecolor="none", label="Built in 2024 (excluded)")
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=True, fontsize=9)

    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_final_high_risk_map(
    high_risk_mask: np.ndarray,
    candidate_mask: np.ndarray,
    built_mask: np.ndarray,
    path: str | Path,
    title: str = "Final High-Risk Map",
) -> None:
    """Save one high-risk hotspot map."""
    out = _prepare_path(path)

    candidate = candidate_mask.astype(bool)
    built = built_mask.astype(bool)
    high_risk = high_risk_mask.astype(bool)

    rgb = np.ones((*candidate.shape, 3), dtype=np.float32)
    rgb[built] = np.array([0.23, 0.23, 0.23])
    rgb[candidate] = np.array([0.92, 0.92, 0.92])
    rgb[high_risk] = np.array([0.85, 0.16, 0.16])

    fig, ax = plt.subplots(figsize=(8.8, 6.8), facecolor="white", constrained_layout=True)
    ax.imshow(rgb)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    legend_items = [
        Patch(facecolor="#d92a2a", edgecolor="none", label="High-risk hotspot"),
        Patch(facecolor="#ebebeb", edgecolor="none", label="Other candidate non-built"),
        Patch(facecolor="#3b3b3b", edgecolor="none", label="Built in 2024 (excluded)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=True, fontsize=9)

    fig.savefig(out, dpi=220)
    plt.close(fig)
