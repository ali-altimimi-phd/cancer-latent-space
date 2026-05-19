from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loading import load_latent_array, load_metadata
from .latent_config import COMPARISON_MAP, PLOT_LABEL_MAP, STATE_CANCER, STATE_NORMAL


def latent_columns(df: pd.DataFrame) -> list[str]:
    """Return latent coordinate columns in z1, z2, ... order."""
    return sorted(
        [c for c in df.columns if c.startswith("z")],
        key=lambda c: int(c[1:]) if c[1:].isdigit() else c,
    )


def _get_config_attr(config: Any, name: str, default: Any = None) -> Any:
    """Small compatibility helper while LatentConfig is being refactored."""
    return getattr(config, name, default)


def get_latent_model_id(config: Any) -> str:
    """Construct or retrieve a stable latent model identifier."""
    explicit = _get_config_attr(config, "latent_model_id", None)
    if explicit:
        return str(explicit)

    chip = _get_config_attr(config, "chip_id", "unknown_chip")
    regime = _get_config_attr(config, "filter_regime", None)
    if regime is None:
        regime = _get_config_attr(config, "filter_method", "unknown_regime")
    top_n = _get_config_attr(config, "top_n", "unknown_topn")
    latent_dim = _get_config_attr(config, "latent_dim", "unknown_dim")
    return f"{chip}_{regime}_top{top_n}_vae_z{latent_dim}"


def get_filter_regime(config: Any) -> str:
    """Return canonical filter regime, accepting old config objects during transition."""
    regime = _get_config_attr(config, "filter_regime", None)
    if regime is not None:
        return str(regime)
    old = _get_config_attr(config, "filter_method", None)
    if old == "variance":
        return "variance_global"
    if old is not None:
        return str(old)
    return "unknown_regime"


def build_latent_dataframe(
    config: Any,
    latent_path: Path | None = None,
    metadata_path: Path | None = None,
    require_full_contract: bool = True,
) -> pd.DataFrame:
    """Load latent coordinates and metadata, assert row alignment, and return joined table."""
    latent_path = Path(latent_path or config.latent_path)
    metadata_path = Path(metadata_path or config.metadata_aligned_path)

    latent = load_latent_array(latent_path)
    metadata = load_metadata(
        metadata_path=metadata_path,
        require_full_contract=require_full_contract,
    )

    if latent.shape[0] != metadata.shape[0]:
        raise ValueError(
            f"Latent/metadata sample mismatch: {latent.shape[0]} latent rows vs "
            f"{metadata.shape[0]} metadata rows"
        )

    latent_df = pd.DataFrame(latent, index=metadata.index)
    latent_df.columns = [f"z{i + 1}" for i in range(latent_df.shape[1])]
    latent_df.index.name = "sample_id"

    if not latent_df.index.equals(metadata.index):
        raise ValueError("Latent rows and metadata rows are not aligned.")

    if not metadata["sample_id"].astype(str).equals(metadata.index.to_series().astype(str)):
        raise ValueError("sample_id column does not match metadata index.")

    out = latent_df.join(metadata)

    # Synthesis-facing provenance fields.
    out["chip"] = _get_config_attr(config, "chip_id", pd.NA)
    out["filter_regime"] = get_filter_regime(config)
    out["latent_model_id"] = get_latent_model_id(config)

    return out


def add_standard_latent_labels(
    df: pd.DataFrame,
    comparison_map: dict[tuple[str, str], str] = COMPARISON_MAP,
    plot_label_map: dict[str, str] = PLOT_LABEL_MAP,
) -> pd.DataFrame:
    """Add tissue/disease/state/comparison/short_label/plot_label columns."""
    out = df.copy()
    out["tissue"] = out["tissue_clean"]
    out["disease"] = out["disease_clean"]
    out["state"] = out["condition"]

    cancer_mask = out["state"].eq(STATE_CANCER)
    out["comparison"] = pd.NA
    out.loc[cancer_mask, "comparison"] = (
        out.loc[cancer_mask, "tissue"].astype(str)
        + "/"
        + out.loc[cancer_mask, "disease"].astype(str)
    )

    out["short_label"] = pd.NA
    out.loc[cancer_mask, "short_label"] = [
        comparison_map.get((t, d))
        for t, d in zip(out.loc[cancer_mask, "tissue"], out.loc[cancer_mask, "disease"])
    ]

    out["plot_label"] = pd.NA
    out.loc[cancer_mask, "plot_label"] = out.loc[cancer_mask, "short_label"].map(plot_label_map)

    return out


def find_unmapped_latent_labels(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return unmapped project/plot labels without notebook-only display side effects."""
    cancer = df[df["state"].eq(STATE_CANCER)].copy()

    missing_project = (
        cancer[cancer["short_label"].isna()][["tissue", "disease"]]
        .drop_duplicates()
        .sort_values(["tissue", "disease"])
        .reset_index(drop=True)
    )

    missing_plot = (
        cancer[cancer["plot_label"].isna()][["short_label", "tissue", "disease"]]
        .drop_duplicates()
        .sort_values(["short_label", "tissue", "disease"], na_position="last")
        .reset_index(drop=True)
    )

    return {
        "missing_project_labels": missing_project,
        "missing_plot_labels": missing_plot,
    }


def build_latent_working_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return standardized sample-level latent table consumed by downstream metrics."""
    base_cols = [
        "sample_id",
        "chip",
        "filter_regime",
        "latent_model_id",
        "geo_accession",
        "title",
        "platform_id",
        "tissue",
        "disease",
        "state",
        "comparison",
        "short_label",
        "plot_label",
        "tissue_label",
    ]
    available_base_cols = [c for c in base_cols if c in df.columns]
    return df[available_base_cols + latent_columns(df)].copy()


def build_and_save_latent_working_table(config: Any, logger: Any | None = None) -> pd.DataFrame:
    """Production replacement for Notebook 4."""
    config.ensure_directories()

    df = build_latent_dataframe(config)
    df = add_standard_latent_labels(df)
    work = build_latent_working_table(df)

    latent_tables_dir = Path(config.latent_tables_dir)
    latent_tables_dir.mkdir(parents=True, exist_ok=True)

    work_path = latent_tables_dir / "latent_sample_coordinates.csv"
    work.to_csv(work_path, index=False)

    # Compatibility name matching old Notebook 4 product.
    working_path = latent_tables_dir / "latent_working_table.csv"
    work.to_csv(working_path, index=False)

    unmapped = find_unmapped_latent_labels(df)
    unmapped["missing_project_labels"].to_csv(
        latent_tables_dir / "latent_unmapped_project_labels.csv",
        index=False,
    )
    unmapped["missing_plot_labels"].to_csv(
        latent_tables_dir / "latent_unmapped_plot_labels.csv",
        index=False,
    )

    if getattr(config, "make_plots", True):
        save_latent_centroid_plot(work, Path(config.latent_plots_dir) / "latent_space_centroids.png")

    if logger is not None:
        logger.info("Saved latent sample table: %s", work_path)
        logger.info("Saved latent working table: %s", working_path)

    return work


def save_latent_centroid_plot(work: pd.DataFrame, output_path: Path) -> None:
    """Save a static z1/z2 cancer centroid plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "z1" not in work.columns or "z2" not in work.columns:
        raise ValueError("Cannot plot latent centroids because z1/z2 columns are missing.")

    plot_df = work[
        work["state"].eq(STATE_CANCER) & work["plot_label"].notna()
    ].copy()

    if plot_df.empty:
        raise ValueError("No cancer samples with plot_label available for centroid plot.")

    plt.figure(figsize=(8, 6))
    plt.scatter(plot_df["z1"], plot_df["z2"], s=15, alpha=0.3)

    centroids = plot_df.groupby("plot_label")[["z1", "z2"]].mean()
    plt.scatter(centroids["z1"], centroids["z2"], s=80)

    for label, row in centroids.iterrows():
        plt.text(row["z1"], row["z2"], label, fontsize=9)

    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent space: cancer samples with class centroids")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
