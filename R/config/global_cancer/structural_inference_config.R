# ------------------------------------------------------------------------------
# File: structural_inference_config.R
# Purpose: Configuration for the Global Cancer Structural Inference pipeline
# Role: Defines chips, filter regimes, engine toggles, paths, and logging
# Pipeline: Structural Inference
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

# ==============================================================================
# 1. Study identifier
# ==============================================================================

study_name <- "global_cancer"


# ==============================================================================
# 2. Structural inference pipeline toggles
# ==============================================================================


# TRUE / FALSE

build_matrix_maps <- FALSE
run_probe_selection <- FALSE

run_mp_engine         <- FALSE
run_complexity_engine <- FALSE
run_entropy_engine    <- FALSE
run_latent_engine     <- FALSE
build_structural_phenotype_table <- FALSE
build_structural_synthesis_plots <- FALSE
ingest_structural_results_to_duckdb <- TRUE
validate_structural_pipeline_outputs <- FALSE

# ==============================================================================
# 3. Chips and structural engines
# ==============================================================================

# Chips to include in structural inference.
chips <- c("hu35ksuba", "hu6800")

# Structural engines intended for the full framework.
engines <- c("complexity", "entropy", "mp_spectral", "latent")


# ==============================================================================
# 4. Probe-selection regimes
# ==============================================================================

# Canonical structural filter regimes.
#
# limma
#   Comparison-specific differential-expression filter.
#
# variance_comparison
#   Comparison-specific high-variance filter.
#
# variance_global
#   Chip-level global high-variance filter reused across comparisons.

filter_regimes <- c(
  "limma",
  "variance_comparison",
  "variance_global"
)

# Backward-compatible alias during transition.
selection_regimes <- filter_regimes


# ---- Limma regime -------------------------------------------------------------

limma_logfc_cutoff <- 0.33
limma_pval_cutoff  <- 0.05


# ---- Variance regimes ---------------------------------------------------------

variance_top_n <- 3000
variance_threshold <- 0.75

# Prefer explicit variance scopes over variance_mode alone.
variance_selection_mode <- "top_n"

# Backward-compatible alias during refactor.
variance_mode <- variance_selection_mode

# Canonical regime metadata table.
filter_regime_tbl <- tibble::tribble(
  ~filter_regime,          ~filter_method, ~filter_scope,  ~filter_n,        ~variance_mode,
  "limma",                 "limma",        "comparison",   NA_integer_,      NA_character_,
  "variance_comparison",   "variance",     "comparison",   variance_top_n,   variance_selection_mode,
  "variance_global",       "variance",     "global",       variance_top_n,   variance_selection_mode
)

# ==============================================================================
# 5. Structural admissibility thresholds
# ==============================================================================

# These thresholds define when a normal/tumor comparison is eligible for
# structural inference.
#
# They are shared across structural engines so that skipped comparisons reflect
# a common pipeline-level admissibility rule rather than engine-specific defaults.

# Minimum number of samples required in each condition group
# normal/tumor before computing structural summaries.
min_samples_per_condition <- 5

# Minimum number of selected probes required for structural computation.
#
# This applies to MP, complexity, and entropy engines. Although some spectral
# operations can be computed with fewer probes, very small probe sets should not
# be interpreted as stable structural descriptors.
min_selected_probes <- 5


# ==============================================================================
# 6. Structural engine inference settings
# ==============================================================================

# These settings control optional inferential layers within structural engines.
# Core structural descriptors are always computed when the corresponding engine
# is enabled. Permutation and bootstrap procedures provide optional inferential
# overlays for evaluating descriptor stability and statistical extremeness.
#
# Recommended workflow:
#
#   - run descriptor-first structural summaries routinely
#   - enable permutation/bootstrap selectively for:
#       * exploratory validation
#       * biological enrichment reporting
#       * targeted methodological studies
#       * publication-oriented analyses
#
# Structural resampling diagnostics were performed using comparison-level
# structural outputs across:
#
#   - both Affymetrix platforms
#   - both structural filter regimes
#   - multiple resampling depths
#
# Diagnostic findings guided the current default recommendations below.


# ---- Shared resampling defaults ----------------------------------------------

# Base seed used by structural engine resampling routines.
#
# Individual engines may internally derive additional seeds from this base
# value to preserve reproducibility across permutation/bootstrap workflows.

structural_resampling_seed <- 20260510L

# Default resampling dimension.
#
# "sample"
#   Resample or permute biological samples/sample labels.
#
# "probe"
#   Resample selected probes/features.
#
# Routine normal/tumor structural inference should generally use "sample"
# because the primary inferential target is phenotype-level organization rather
# than probe-selection uncertainty.

structural_resampling_unit <- "sample"


# ---- Complexity engine inference ---------------------------------------------
#
# Structural resampling diagnostics indicated that:
#
#   - permutation-based inference for kappa_delta is numerically stable
#   - permutation runtimes remain computationally tractable at n_perm = 100
#   - bootstrap confidence intervals for raw kappa-based statistics were highly
#     unstable due to ill-conditioned resampled matrices and extreme condition
#     number sensitivity
#
# Complexity metric used by optional inferential layers.
#
# Current recommendation:
#   "kappa"
#
# Historical option:
#   "all"
#
# "all" preserves legacy behavior by attempting inferential calculations for
# multiple complexity descriptors simultaneously. This is currently discouraged
# for routine workflows because different descriptors may exhibit substantially
# different numerical behavior under resampling.
#
# Matrix orientation for covariance/eigenspectrum-based complexity descriptors.
#
# "sample" computes covariance among samples across selected probes:
#
#   cov(mat)
#
# where rows correspond to probes/features and columns correspond to samples.
#
# This orientation is consistent with the entropy engine and with the broader
# structural inference framework.
#
# Bootstrap unit for complexity confidence intervals.
#
# "sample" resamples biological samples within phenotype groups and reflects
# phenotype-level uncertainty estimation.
#
# Complexity bootstrap is currently disabled by default because raw kappa-based
# bootstrap confidence intervals exhibited severe numerical instability during
# structural resampling diagnostics.
#
# Permutation unit for complexity testing.
#
# "sample_label" performs phenotype-label exchangeability testing by permuting
# normal/tumor sample labels while preserving matrix structure.
#
# Structural diagnostics indicated that permutation-based inference for
# kappa_delta is stable and suitable for exploratory and reporting workflows.

run_complexity_permutation <- TRUE
complexity_n_perm <- 100
complexity_permutation_metric <- "kappa"
complexity_permutation_unit <- "sample_label"

run_complexity_bootstrap <- FALSE
complexity_n_boot <- 0
complexity_bootstrap_metric <- "kappa"
complexity_bootstrap_unit <- "sample"

complexity_covariance_space <- "sample"


# ---- Entropy engine inference ------------------------------------------------
#
# Structural resampling diagnostics indicated that:
#
#   - permutation-based inference for spectral entropy is numerically stable
#   - bootstrap confidence intervals for spectral entropy are interpretable and
#     substantially more stable than kappa-based complexity bootstraps
#   - n_perm = 100 and n_boot = 100 provide a reasonable balance between
#     runtime and inferential stability for exploratory and reporting workflows
#
# Entropy metric used by optional inferential layers.
#
# Current recommendation:
#   "spectral"
#
# Historical options:
#   "shannon"
#   "all"
#
# Earlier implementations primarily emphasized Shannon entropy. The current
# framework prioritizes spectral entropy because it aligns more directly with
# covariance-spectrum organization shared across MP spectral analysis, latent
# geometry, and structural organization metrics.
#
# Matrix orientation for covariance/eigenspectrum-based entropy descriptors.
#
# "sample" computes covariance among samples across selected probes:
#
#   cov(mat)
#
# where rows correspond to probes/features and columns correspond to samples.
#
# "probe" computes covariance among probes across samples:
#
#   cov(t(mat))
#
# The "sample" orientation is currently preferred for:
#
#   - consistency with the structural inference framework
#   - improved runtime characteristics
#   - stable eigenspectrum estimation
#   - comparability with latent-space and MP spectral analyses
#
# Bootstrap unit for entropy confidence intervals.
#
# "sample" resamples biological samples within phenotype groups and reflects
# phenotype-level uncertainty estimation.
#
# Structural diagnostics indicated that entropy bootstrap intervals are
# numerically stable and suitable for exploratory and reporting workflows.
#
# Permutation unit for entropy testing.
#
# "sample_label" performs phenotype-label exchangeability testing by permuting
# normal/tumor labels while preserving matrix structure.
#
# Structural diagnostics indicated that permutation-based inference for
# spectral entropy is stable and computationally tractable at n_perm = 100.

run_entropy_permutation <- TRUE
entropy_n_perm <- 100
entropy_permutation_metric <- "spectral"

run_entropy_bootstrap <- TRUE
entropy_n_boot <- 100
entropy_bootstrap_metric <- "spectral"

entropy_covariance_space <- "sample"

entropy_bootstrap_unit <- "sample"
entropy_permutation_unit <- "sample_label"


# ---- MP engine inference ------------------------------------------------------
#
# MP spectral analysis is currently descriptive only.
#
# Future work may introduce:
#
#   - MP-specific null-model simulations
#   - eigenspectrum permutation procedures
#   - covariance randomization tests
#   - Tracy–Widom-style significance approximations
#
# For the current framework, MP metrics are treated as structural descriptors
# without inferential resampling layers.

run_mp_resampling <- FALSE
mp_n_resamples <- 0


# ==============================================================================
# 7. Gene-set interpretation settings
# ==============================================================================

gene_set_modes <- c("GO_BP", "GO_MF", "KEGG", "MSIGDB")

min_gene_set_probes <- list(
  GO_BP  = 5,
  GO_MF  = 5,
  KEGG   = 10,
  MSIGDB = 10
)

# These are retained for later biological interpretation, but the structural
# inference layer should not depend on GO/KEGG/Hallmark interpretation.
# Biological interpretation should consume structural phenotype outputs later.
quantile_cutoff <- 0.75
min_probes      <- 5
gene_set_min_mapped_probes <- 5 # rename min_probes in future

# Gene-set-level filtering parameters
quantile_cutoff <- 0.75


# ==============================================================================
# 8. Canonical input paths
# ==============================================================================

# Canonical preprocessing artifacts now live under data/global_cancer/processed/.
processed_dir <- here::here("data", study_name, "processed")
processed_rdata_dir <- file.path(processed_dir, "RData")
processed_ml_input_dir <- file.path(processed_dir, "ml_inputs")

eset_path <- file.path(
  processed_rdata_dir,
  paste0(study_name, "_eset_list.RData")
)

annotations_path <- file.path(
  processed_rdata_dir,
  "annotations",
  "full_chip_annotations.rds"
)


# ==============================================================================
# 9. Structural inference output paths
# ==============================================================================

# Keep the existing project structure stable by adding a thin, isolated output
# layer rather than moving old analysis/reporting artifacts.
structural_output_dir <- here::here(
  "output",
  study_name,
  "structural_inference"
)

structural_rdata_dir <- file.path(structural_output_dir, "RData")


# ---- Structural inference reusable input artifacts ---------------------------
#
# Canonical reusable artifacts consumed by downstream structural-inference
# stages. These paths allow selected stages to be restarted safely without
# assuming required objects are already present in memory.

# ---- Expression matrices and comparison maps ---------------------------------

structural_expression_matrices_dir <- file.path(
  structural_rdata_dir,
  "expression_matrices"
)

matrices_path <- file.path(
  structural_expression_matrices_dir,
  "global_cancer_matrix_maps.RData"
)

# ---- Filtered probe sets -----------------------------------------------------

structural_filtered_probes_dir <- file.path(
  structural_rdata_dir,
  "filtered_probes"
)

# ---- Shared input-loading helper ---------------------------------------------

load_pipeline_inputs_helper <- here::here(
  "R",
  "helpers",
  "load_pipeline_inputs.R"
)

# ---- Structural inference output directories ---------------------------------

structural_tables_dir <- file.path(
  structural_output_dir,
  "tables"
)

structural_plots_dir <- file.path(
  structural_output_dir,
  "plots"
)

structural_reports_dir <- file.path(
  structural_output_dir,
  "reports"
)

# ---- Structural synthesis outputs --------------------------------------------

structural_synthesis_table_dir <- file.path(
  structural_tables_dir,
  "synthesis"
)

structural_synthesis_rdata_dir <- file.path(
  structural_rdata_dir,
  "synthesis"
)

structural_synthesis_plot_dir <- file.path(
  structural_plots_dir,
  "synthesis"
)

# ---- Structural synthesis tables ---------------------------------------------

structural_phenotype_long_csv <- file.path(
  structural_synthesis_table_dir,
  "structural_phenotype_long.csv"
)

structural_phenotype_wide_csv <- file.path(
  structural_synthesis_table_dir,
  "structural_phenotype_wide.csv"
)

structural_phenotype_heatmap_long_csv <- file.path(
  structural_synthesis_table_dir,
  "structural_phenotype_heatmap_long.csv"
)

structural_phenotype_summary_csv <- file.path(
  structural_synthesis_table_dir,
  "structural_phenotype_summary.csv"
)

# ---- Structural synthesis RData ----------------------------------------------

structural_phenotype_tables_rds <- file.path(
  structural_synthesis_rdata_dir,
  "structural_phenotype_tables.rds"
)

# ---- Structural metadata/manifests -------------------------------------------

structural_filter_manifest_path <- file.path(
  structural_tables_dir,
  "structural_filter_manifest.csv"
)


# ==============================================================================
# 10. Warehouse paths
# ==============================================================================


warehouse_dir <- here::here(
  "output",
  study_name,
  "warehouse"
)

warehouse_db_path <- file.path(
  warehouse_dir,
  "global_cancer_results.duckdb"
)


# ==============================================================================
# 11. Logging paths
# ==============================================================================

logs_dir <- here::here("output", study_name, "logs")
structural_logs_dir <- file.path(logs_dir, "structural_inference")

structural_pipeline_logfile <- file.path(
  structural_logs_dir,
  "structural_inference_pipeline_log.txt"
)


# ==============================================================================
# 12. Latent-space engine configuration
# ==============================================================================

# Latent space is treated as a structural engine with a Python backend.
#
# R owns:
#
#   - orchestration
#   - configuration
#   - provenance
#   - synthesis integration
#
# Python acts as a computational backend only.

run_latent_python_scripts <- TRUE


# ---- Python runtime ----------------------------------------------------------

latent_python_exe <- Sys.getenv(
  "LATENT_PYTHON_EXE",
  unset = "C:/Users/drziy/anaconda3/envs/ml/python.exe"
)

latent_script_dir <- here::here(
  "R",
  "engines",
  "latent",
  "python"
)


# ---- Canonical latent structural regime --------------------------------------

latent_chip_id <- "hu35ksuba"

latent_filter_regime <- "variance_global"

latent_top_n <- 3000


# ---- Latent model parameters -------------------------------------------------

latent_dim <- 10

latent_epochs <- 50

latent_batch_size <- 32

latent_beta <- 0.01

latent_learning_rate <- 1e-3

latent_random_seed <- 42


# ---- Dataset splitting -------------------------------------------------------

latent_test_size_total <- 0.30

latent_val_fraction_of_temp <- 0.50


# ---- Output controls ---------------------------------------------------------

latent_make_plots <- TRUE


# ---- Latent provenance -------------------------------------------------------

latent_model_version <- "v1"

latent_model_id <- paste(
  latent_chip_id,
  latent_filter_regime,
  paste0("top", latent_top_n),
  "vae",
  latent_model_version,
  sep = "_"
)


# ---- Latent structural outputs -----------------------------------------------

latent_tables_dir <- file.path(
  structural_tables_dir,
  "latent"
)

latent_plots_dir <- file.path(
  structural_plots_dir,
  "latent"
)

latent_rdata_dir <- file.path(
  structural_rdata_dir,
  "latent"
)


# ---- Production Python execution order ---------------------------------------

latent_python_scripts <- c(
  "run_latent_preprocessing.py",
  "run_latent_training.py",
  "run_latent_tables.py",
  "run_latent_metrics.py"
)


# ==============================================================================
# 14. Validation helpers
# ==============================================================================

valid_chips <- c("hu35ksuba", "hu6800")

valid_filter_regimes <- c(
  "limma",
  "variance_comparison",
  "variance_global"
)

valid_selection_regimes <- valid_filter_regimes

valid_variance_modes <- c("top_n", "threshold")
valid_filter_scopes  <- c("comparison", "global")

valid_structural_engines <- c(
  "mp_spectral",
  "complexity",
  "entropy",
  "latent"
)


# ==============================================================================
# 15. Ensure required directories exist
# ==============================================================================

dirs_to_create <- c(
  structural_output_dir,
  structural_rdata_dir,
  
  structural_filtered_probes_dir,
  structural_expression_matrices_dir,
  
  structural_tables_dir,
  structural_plots_dir,
  structural_reports_dir,
  structural_logs_dir,
  
  structural_synthesis_table_dir,
  structural_synthesis_rdata_dir,
  structural_synthesis_plot_dir,
  
  latent_tables_dir,
  latent_plots_dir,
  latent_rdata_dir
)

invisible(lapply(
  dirs_to_create,
  dir.create,
  recursive = TRUE,
  showWarnings = FALSE
))