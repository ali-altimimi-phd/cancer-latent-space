from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .latent_config import COMPARISON_MAP, PLOT_LABEL_MAP, STATE_CANCER, STATE_NORMAL
from .latent_tables import (
    add_standard_latent_labels,
    build_latent_dataframe,
    build_latent_working_table,
    get_filter_regime,
    get_latent_model_id,
    latent_columns,
)


def safe_cov(X: np.ndarray) -> np.ndarray:
    """Return sample-space covariance for latent coordinates."""
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    if X.shape[0] < 2:
        return np.full((X.shape[1], X.shape[1]), np.nan)
    return np.cov(X, rowvar=False)


def eigvals_sorted(cov: np.ndarray) -> np.ndarray:
    """Return non-negative covariance eigenvalues in descending order."""
    if np.isnan(cov).all():
        return np.array([np.nan])
    vals = np.linalg.eigvalsh(cov)
    vals = np.clip(vals, 0, None)
    return np.sort(vals)[::-1]


def participation_ratio(eigs: np.ndarray) -> float:
    """Effective dimensionality from covariance eigenvalues."""
    if np.isnan(eigs).all():
        return np.nan
    s1 = np.sum(eigs)
    s2 = np.sum(eigs**2)
    if s2 <= 0:
        return np.nan
    return float((s1**2) / s2)


def eig_entropy(eigs: np.ndarray) -> float:
    """Spectral entropy of normalized latent covariance eigenvalues."""
    if np.isnan(eigs).all():
        return np.nan
    total = np.sum(eigs)
    if total <= 0:
        return np.nan
    p = eigs / total
    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    return float(-(p * np.log(p)).sum())


def top_eig_fraction(eigs: np.ndarray) -> float:
    """Fraction of variance carried by the leading covariance eigenvalue."""
    if np.isnan(eigs).all():
        return np.nan
    total = np.sum(eigs)
    if total <= 0:
        return np.nan
    return float(eigs[0] / total)


def centroid_distances(X: np.ndarray) -> np.ndarray:
    """Euclidean distance of each sample from its class centroid."""
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    if X.shape[0] == 0:
        return np.array([np.nan])
    c = X.mean(axis=0)
    return np.linalg.norm(X - c, axis=1)


def comparison_definitions() -> dict[str, tuple[str, str]]:
    """Invert COMPARISON_MAP into comparison -> (normal tissue, tumor disease)."""
    inverted = {
        comparison: tissue_disease
        for tissue_disease, comparison in COMPARISON_MAP.items()
    }
    comparison_order = list(PLOT_LABEL_MAP.keys())
    return {
        comparison: inverted[comparison]
        for comparison in comparison_order
        if comparison in inverted
    }


def compute_latent_comparison_metrics(
    work: pd.DataFrame,
    config: Any | None = None,
) -> pd.DataFrame:
    """Compute one row per matched normal-tumor latent comparison."""
    latent_cols = latent_columns(work)
    if not latent_cols:
        raise ValueError("No latent coordinate columns found.")

    comparison_map = comparison_definitions()
    results = []

    chip = getattr(config, "chip_id", None) if config is not None else None
    filter_regime = get_filter_regime(config) if config is not None else None
    latent_model_id = get_latent_model_id(config) if config is not None else None

    if chip is None and "chip" in work.columns:
        chip = work["chip"].dropna().iloc[0] if work["chip"].notna().any() else None
    if filter_regime is None and "filter_regime" in work.columns:
        filter_regime = work["filter_regime"].dropna().iloc[0] if work["filter_regime"].notna().any() else None
    if latent_model_id is None and "latent_model_id" in work.columns:
        latent_model_id = work["latent_model_id"].dropna().iloc[0] if work["latent_model_id"].notna().any() else None

    for comp, (normal_tissue, tumor_disease) in comparison_map.items():
        df_normal = work.loc[
            (work["state"] == STATE_NORMAL) &
            (work["tissue"] == normal_tissue)
        ]
        df_tumor = work.loc[
            (work["state"] == STATE_CANCER) &
            (work["tissue"] == normal_tissue) &
            (work["disease"] == tumor_disease)
        ]

        n_normal = len(df_normal)
        n_tumor = len(df_tumor)

        if n_normal < 2 or n_tumor < 2:
            continue

        X_normal = df_normal[latent_cols].to_numpy(dtype=float)
        X_tumor = df_tumor[latent_cols].to_numpy(dtype=float)

        eigs_normal = eigvals_sorted(safe_cov(X_normal))
        eigs_tumor = eigvals_sorted(safe_cov(X_tumor))

        pr_normal = participation_ratio(eigs_normal)
        pr_tumor = participation_ratio(eigs_tumor)

        ent_normal = eig_entropy(eigs_normal)
        ent_tumor = eig_entropy(eigs_tumor)

        anisotropy_normal = top_eig_fraction(eigs_normal)
        anisotropy_tumor = top_eig_fraction(eigs_tumor)

        radius_normal = float(np.nanmean(centroid_distances(X_normal)))
        radius_tumor = float(np.nanmean(centroid_distances(X_tumor)))

        centroid_normal = X_normal.mean(axis=0)
        centroid_tumor = X_tumor.mean(axis=0)
        centroid_distance = float(np.linalg.norm(centroid_tumor - centroid_normal))

        results.append(
            {
                "chip": chip,
                "filter_regime": filter_regime,
                "comparison": comp,
                "group": PLOT_LABEL_MAP.get(comp, comp),
                "latent_model_id": latent_model_id,
                "normal_tissue": normal_tissue,
                "tumor_disease": tumor_disease,
                "n_normal": n_normal,
                "n_tumor": n_tumor,
                "pr_normal": pr_normal,
                "pr_tumor": pr_tumor,
                "pr_delta": pr_tumor - pr_normal,
                "eig_entropy_normal": ent_normal,
                "eig_entropy_tumor": ent_tumor,
                "eig_entropy_delta": ent_tumor - ent_normal,
                "anisotropy_normal": anisotropy_normal,
                "anisotropy_tumor": anisotropy_tumor,
                "anisotropy_delta": anisotropy_tumor - anisotropy_normal,
                "radius_normal": radius_normal,
                "radius_tumor": radius_tumor,
                "radius_delta": radius_tumor - radius_normal,
                "centroid_distance": centroid_distance,
                "plot_label": PLOT_LABEL_MAP.get(comp, comp),
                "normal_code": comp.split("/")[0],
                "tumor_code": comp.split("/")[1] if "/" in comp else comp,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(["chip", "filter_regime", "comparison"], na_position="last")
        .reset_index(drop=True)
    )


def build_and_save_latent_metrics(config: Any, logger: Any | None = None) -> pd.DataFrame:
    """Production replacement for Notebook 5."""
    config.ensure_directories()

    df = build_latent_dataframe(config)
    df = add_standard_latent_labels(df)
    work = build_latent_working_table(df)
    metrics = compute_latent_comparison_metrics(work, config=config)

    latent_tables_dir = Path(config.latent_tables_dir)
    latent_tables_dir.mkdir(parents=True, exist_ok=True)

    work_path = latent_tables_dir / "latent_sample_coordinates.csv"
    metrics_path = latent_tables_dir / "latent_comparison_metrics.csv"

    work.to_csv(work_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    if getattr(config, "make_plots", True):
        save_latent_structural_change_plot(
            metrics,
            Path(config.latent_plots_dir) / "latent_structural_change_space.png",
        )

    if logger is not None:
        logger.info("Saved latent sample coordinates: %s", work_path)
        logger.info("Saved latent comparison metrics: %s", metrics_path)

    return metrics


def save_latent_structural_change_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    """Save the latent participation-ratio vs anisotropy-delta plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if metrics.empty:
        raise ValueError("No latent comparison metrics available to plot.")

    plt.figure(figsize=(8, 6))
    plt.axhline(0, alpha=0.5)
    plt.axvline(0, alpha=0.5)

    x = metrics["pr_delta"]
    y = metrics["anisotropy_delta"]
    labels = metrics["plot_label"]

    plt.scatter(x, y, s=40)

    for xi, yi, lab in zip(x, y, labels):
        plt.text(xi, yi, lab, fontsize=8)

    plt.xlabel("Δ Participation Ratio")
    plt.ylabel("Δ Leading-Eigenvalue Fraction")
    plt.title("Latent structural change across matched comparisons")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
