from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REQUIRED_METADATA_COLUMNS = [
    "sample_id",
    "geo_accession",
    "title",
    "platform_id",
    "disease_clean",
    "tissue_clean",
    "condition",
    "tissue_label",
]

MINIMAL_METADATA_COLUMNS = [
    "sample_id",
    "geo_accession",
    "condition",
    "tissue_label",
]

COMPARISON_MAP = {
    ("Bladder", "bladder transitional cell carcinoma"): "BLAD/TCC",
    ("Breast", "breast adenocarcinoma"): "BR/BRAD",
    ("Colon", "colorectal adenocarcinoma"): "COL/COADREAD",
    ("Kidney", "renal cell carcinoma"): "KID/RCC",
    ("Lung", "lung adenocarcinoma"): "LU/LUAD",
    ("Ovary", "ovarian adenocarcinoma"): "OV/OVAD",
    ("Pancreas", "pancreatic adenocarcinoma"): "PA/PAAD",
    ("Prostate", "prostate adenocarcinoma"): "PR/PRAD",
    ("Uterus", "uterine adenocarcinoma"): "UT/EAC",
    ("Brain", "glioblastoma"): "Brain/GBM",
    ("Brain", "medulloblastoma"): "Brain/MB",
    ("Lymphoid Tissue", "Follicular lymphoma"): "GC/FL",
    ("Lymphoid Tissue", "large B-cell lymphoma"): "GC/LBCL",
    ("Blood", "acute myeloid leukemia"): "PB/AML",
    ("Bone Marrow", "B-cell ALL"): "PB/B-ALL",
    ("Bone Marrow", "T-cell ALL"): "PB/T-ALL",
}

PLOT_LABEL_MAP = {
    "BLAD/TCC": "TCC",
    "BR/BRAD": "BRAD",
    "COL/COADREAD": "COADREAD",
    "KID/RCC": "RCC",
    "LU/LUAD": "LUAD",
    "OV/OVAD": "OVAD",
    "PA/PAAD": "PAAD",
    "PR/PRAD": "PRAD",
    "UT/EAC": "EAC",
    "Brain/GBM": "GBM",
    "Brain/MB": "MB",
    "GC/FL": "FL",
    "GC/LBCL": "LBCL",
    "PB/AML": "AML",
    "PB/B-ALL": "B-ALL",
    "PB/T-ALL": "T-ALL",
}

STATE_CANCER = "cancer"
STATE_NORMAL = "normal"


def find_project_root(start_dir: Path) -> Path:
    """Locate the global-cancer-complexity project root."""
    start_dir = start_dir.resolve()
    candidates = [start_dir] + list(start_dir.parents)
    for candidate in candidates:
        if (candidate / "data" / "global_cancer").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate global-cancer-complexity project root from: "
        f"{start_dir}"
    )


@dataclass(frozen=True)
class LatentConfig:
    """Canonical paths and tunable parameters for latent-space production scripts."""

    project_dir: Path
    chip_id: str = "hu35ksuba"
    filter_method: str = "variance"
    top_n: int = 3000
    random_seed: int = 42
    test_size_total: float = 0.30
    val_fraction_of_temp: float = 0.50
    batch_size: int = 32
    latent_dim: int = 10
    beta: float = 0.01
    learning_rate: float = 1e-3
    epochs: int = 50
    make_plots: bool = True

    @classmethod
    def from_project_root(
        cls,
        project_dir: str | Path | None = None,
        start_dir: str | Path | None = None,
        **kwargs,
    ) -> "LatentConfig":
        if project_dir is None:
            project_dir = find_project_root(Path(start_dir or Path.cwd()))
        return cls(project_dir=Path(project_dir).resolve(), **kwargs)

    @property
    def ml_input_dir(self) -> Path:
        return self.project_dir / "data" / "global_cancer" / "processed" / "ml_inputs"

    @property
    def processed_dir(self) -> Path:
        return self.project_dir / "data" / "global_cancer" / "processed"

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "output" / "global_cancer"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def training_plots_dir(self) -> Path:
        return self.plots_dir / "training"

    @property
    def latent_plots_dir(self) -> Path:
        return self.plots_dir / "latent"

    @property
    def latent_tables_dir(self) -> Path:
        return self.tables_dir / "latent"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models" / "latent"

    @property
    def expression_filename(self) -> str:
        return f"{self.chip_id}_expr_top{self.top_n}_{self.filter_method}.csv"

    @property
    def metadata_filename(self) -> str:
        return f"{self.chip_id}_metadata_aligned.csv"

    @property
    def features_filename(self) -> str:
        return f"{self.chip_id}_top{self.top_n}_{self.filter_method}_features.csv"

    @property
    def expression_path(self) -> Path:
        return self.ml_input_dir / self.expression_filename

    @property
    def metadata_path(self) -> Path:
        return self.ml_input_dir / self.metadata_filename

    @property
    def features_path(self) -> Path:
        return self.ml_input_dir / self.features_filename

    @property
    def x_train_path(self) -> Path:
        return self.processed_dir / "X_train.npy"

    @property
    def x_val_path(self) -> Path:
        return self.processed_dir / "X_val.npy"

    @property
    def x_test_path(self) -> Path:
        return self.processed_dir / "X_test.npy"

    @property
    def ids_train_path(self) -> Path:
        return self.processed_dir / "ids_train.npy"

    @property
    def ids_val_path(self) -> Path:
        return self.processed_dir / "ids_val.npy"

    @property
    def ids_test_path(self) -> Path:
        return self.processed_dir / "ids_test.npy"

    @property
    def x_scaled_all_path(self) -> Path:
        return self.processed_dir / "X_scaled_all.npy"

    @property
    def sample_ids_all_path(self) -> Path:
        return self.processed_dir / "sample_ids_all.npy"

    @property
    def feature_ids_path(self) -> Path:
        return self.processed_dir / "feature_ids.npy"

    @property
    def metadata_aligned_path(self) -> Path:
        return self.processed_dir / "metadata_aligned.csv"

    @property
    def expression_aligned_path(self) -> Path:
        return self.processed_dir / "expression_aligned.csv"

    @property
    def scaler_stats_path(self) -> Path:
        return self.processed_dir / "scaler_stats.csv"

    @property
    def latent_path(self) -> Path:
        return self.processed_dir / "latent.npy"

    @property
    def latent_coordinates_path(self) -> Path:
        return self.processed_dir / "latent_coordinates.csv"

    @property
    def training_history_path(self) -> Path:
        return self.latent_tables_dir / "vae_training_history.csv"

    @property
    def training_summary_path(self) -> Path:
        return self.latent_tables_dir / "vae_training_summary.csv"

    @property
    def model_state_dict_path(self) -> Path:
        return self.models_dir / "first_vae_state_dict.pt"

    @property
    def model_checkpoint_path(self) -> Path:
        return self.models_dir / "first_vae_full_checkpoint.pt"

    def ensure_directories(self) -> None:
        for path in [
            self.processed_dir,
            self.training_plots_dir,
            self.latent_plots_dir,
            self.latent_tables_dir,
            self.models_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
