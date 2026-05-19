# ------------------------------------------------------------------------------
# File: biological_enrichment_config.R
# Purpose: Configuration for biological enrichment pipeline
# Pipeline: Biological enrichment
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------


# ==============================================================================
# 1. Study configuration
# ==============================================================================

study_name <- "global_cancer"


# ==============================================================================
# 2. Biological enrichment pipeline toggles
# ==============================================================================

# TRUE / FALSE

run_pairwise           <- FALSE
run_aggregator         <- FALSE
run_comparison_summary <- FALSE


# ==============================================================================
# 3. Structural-inference inputs consumed by biological layer
# ==============================================================================

chips <- c(
  "hu35ksuba",
  "hu6800"
)

# Concrete filtered-probe datasets produced by the structural inference pipeline.
# These labels must correspond to files already present in:
#
# output/<study_name>/structural_inference/RData/filtered_probes/
#
# The biological enrichment layer does not define feature-selection logic.
# It consumes these already-created filtered probe sets.
filter_regimes <- c(
  "limma",
  "variance_top3000"
)

# # Analytical selection methods used by biological pairwise wrapper
# selection_regimes <- c(
#   "limma",
#   "variance"
# )

# ---- Project output roots ----------------------------------------------------

project_output_root <- here::here(
  "output",
  study_name
)

structural_inference_output_dir <- file.path(
  project_output_root,
  "structural_inference"
)

biological_enrichment_output_dir <- file.path(
  project_output_root,
  "biological_enrichment"
)

# ---- Structural inference reusable inputs -----------------------------------

structural_input_root <- file.path(
  structural_inference_output_dir,
  "RData"
)

matrices_path <- file.path(
  structural_input_root,
  "expression_matrices",
  "global_cancer_matrix_maps.RData"
)

structural_filtered_probes_dir <- file.path(
  structural_input_root,
  "filtered_probes"
)

# ---- Shared helper paths -----------------------------------------------------

load_pipeline_inputs_helper <- here::here(
  "R",
  "helpers",
  "load_pipeline_inputs.R"
)


# ==============================================================================
# 3A. Biological enrichment inferential settings
# ==============================================================================

# These settings control optional inferential layers within the biological
# enrichment pipeline.
#
# Biological enrichment operates downstream of the structural inference layer
# and evaluates structural descriptors within:
#
#   - GO Biological Process terms
#   - GO Molecular Function terms
#   - KEGG pathways
#   - MSIGDB signatures
#
# Core biological structural descriptors are always computed when the
# corresponding engine is enabled. Permutation and bootstrap procedures provide
# optional inferential overlays for evaluating pathway-level structural
# organization and enrichment stability.
#
# Recommended workflow:
#
#   - run descriptor-first biological enrichment routinely for rapid exploratory
#     analyses
#   - enable permutation/bootstrap selectively for:
#       * pathway-level significance estimation
#       * report generation
#       * comparative biological interpretation
#       * publication-oriented analyses
#
# Structural resampling diagnostics were performed at the comparison level using
# full-chip structural outputs across:
#
#   - both Affymetrix platforms
#   - both structural filter regimes
#   - multiple resampling depths
#
# These diagnostics guided the current default inferential recommendations for
# the biological enrichment layer.
#
# These biological-layer defaults intentionally mirror the structural diagnostic
# conclusions, but they should be re-evaluated after the first full biological
# enrichment inference rerun because gene-set-level matrices may have different
# numerical behavior from full-comparison structural matrices.
#
# IMPORTANT VALIDATION NOTE:
# Confirm that the biological pairwise runner actually passes the parameters
# below into the complexity and entropy engines. In particular, verify that the
# runner consumes:
#
#   - biological_resampling_seed
#   - complexity_permutation_unit
#   - complexity_bootstrap_unit
#   - complexity_covariance_space
#   - entropy_permutation_unit
#   - entropy_bootstrap_unit
#   - entropy_covariance_space
#
# If any of these parameters are defined here but not passed into the relevant
# engine calls, the config will be internally documented but operationally
# incomplete.
#
# Because biological enrichment performs large-scale gene-set-level analyses,
# inferential layers substantially increase runtime and output size relative to
# descriptor-only workflows.


# ---- Shared biological resampling defaults -----------------------------------

# Base seed used by biological enrichment resampling routines.
#
# Individual engines may internally derive additional seeds from this base value
# to preserve reproducibility across permutation/bootstrap workflows.

biological_resampling_seed <- 20260510L

# Default biological resampling dimension.
#
# "sample"
#   Resample or permute biological samples/sample labels.
#
# "probe"
#   Resample probes/features within gene sets.
#
# Routine normal/tumor biological enrichment inference should generally use
# "sample" because the primary inferential target is phenotype-level structural
# organization rather than probe-selection uncertainty.

biological_resampling_unit <- "sample"


# ---- Biological complexity inference -----------------------------------------
#
# Structural resampling diagnostics indicated that:
#
#   - permutation-based inference for kappa_delta is numerically stable
#   - permutation runtimes remain computationally tractable at n_perm = 100
#   - bootstrap confidence intervals for raw kappa-based statistics were highly
#     unstable due to ill-conditioned resampled matrices and extreme condition
#     number sensitivity
#
# Complexity metric used by optional biological inferential layers.
#
# Current recommendation:
#   "kappa"
#
# Historical option:
#   "all"
#
# "all" preserves legacy behavior by attempting inferential calculations for
# multiple complexity descriptors simultaneously. This is currently discouraged
# for routine biological workflows because different descriptors may exhibit
# substantially different numerical behavior under resampling.
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
# Bootstrap unit for biological complexity confidence intervals.
#
# "sample" resamples biological samples within phenotype groups and reflects
# phenotype-level uncertainty estimation.
#
# Complexity bootstrap is currently disabled by default because raw kappa-based
# bootstrap confidence intervals exhibited severe numerical instability during
# structural resampling diagnostics.
#
# Permutation unit for biological complexity testing.
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


# ---- Biological entropy inference --------------------------------------------
#
# Structural resampling diagnostics indicated that:
#
#   - permutation-based inference for spectral entropy is numerically stable
#   - bootstrap confidence intervals for spectral entropy are interpretable and
#     substantially more stable than kappa-based complexity bootstraps
#   - n_perm = 100 and n_boot = 100 provide a reasonable balance between
#     runtime and inferential stability for exploratory/reporting workflows
#
# Entropy metric used by optional biological inferential layers.
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
# Bootstrap unit for biological entropy confidence intervals.
#
# "sample" resamples biological samples within phenotype groups and reflects
# phenotype-level uncertainty estimation.
#
# Structural diagnostics indicated that entropy bootstrap intervals are
# numerically stable and suitable for exploratory and reporting workflows.
#
# Permutation unit for biological entropy testing.
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


# ==============================================================================
# 4. Pairwise comparison and gene-set settings
# ==============================================================================

geo_chip_map <- list(
  hu35ksuba = "GPL98",
  hu6800    = "GPL80"
)

biological_engines <- c(
  "complexity",
  "entropy"
)

# gene_set_modes <- c(
#   "GO_BP",
#   "GO_MF",
#   "KEGG",
#   "MSIGDB"
# )


# gene_set_modes <- c(
#   "GO_BP"
# )

gene_set_modes <- c(
  "GO_MF"
)

# gene_set_modes <- c(
#   "KEGG",
#   "MSIGDB"
# )

# Minimum number of usable probes required after intersecting:
#
#   chip expression probes
#   ∩ filtered probes for the current comparison/regime
#   ∩ gene-set annotation probes
#
# These thresholds are biological-layer inclusion rules only.
# They do not perform structural feature selection.

# Gene-set-level filtering parameters.
#
# quantile_cutoff is retained for compatibility with earlier biological
# enrichment helpers. The primary current inclusion rule is
# min_gene_set_probes below.

quantile_cutoff <- 0.75

# Probe-threshold rationale:
#
# Minimum probe thresholds were selected empirically from chip-specific
# gene-set probe-count distributions derived from the integrated annotation
# warehouse.
#
# GO terms were retained at a minimum of 5 probes because both GO Biological
# Process (GO_BP) and GO Molecular Function (GO_MF) exhibited low mapped-probe
# support across both Affymetrix platforms:
#
#   - GO_BP median: 3 probes on both hu35ksuba and hu6800
#   - GO_MF median: 2 probes on both hu35ksuba and hu6800
#
# At a threshold of 5 probes, approximately 30–38% of GO terms remained
# eligible for analysis. Higher thresholds, such as 8–10 probes, would exclude
# the large majority of GO terms and substantially reduce ontology coverage.
#
# KEGG pathways exhibited larger mapped-probe distributions:
#
#   - hu35ksuba median: 18 probes
#   - hu6800 median: 41.5 probes
#
# Approximately 70–86% of KEGG pathways retained at least 10 probes, supporting
# a more stringent threshold for pathway-level analyses.
#
# MSIGDB collections showed the largest mapped-probe support:
#
#   - minimum: 14–15 probes
#   - hu35ksuba median: 67.5 probes
#   - hu6800 median: 128 probes
#
# Therefore, a minimum threshold of 10 probes retains essentially all MSIGDB
# signatures while preserving stronger statistical support for covariance-based
# structural descriptors.

min_gene_set_probes <- list(
  GO_BP  = 5,
  GO_MF  = 5,
  KEGG   = 10,
  MSIGDB = 10
)


# ==============================================================================
# 5. Annotation resources
# ==============================================================================

annotations_path <- here::here(
  "data",
  study_name,
  "processed",
  "RData",
  "annotations",
  "full_chip_annotations.rds"
)

load_biological_annotations_helper <- here::here(
  "R",
  "helpers",
  "biological_enrichment",
  "load_biological_annotations.R"
)


# ==============================================================================
# 6. Output and logging paths
# ==============================================================================

logs_dir <- file.path(
  project_output_root,
  "logs"
)

biological_enrichment_logs_dir <- file.path(
  logs_dir,
  "biological_enrichment"
)

biological_enrichment_rdata_dir <- file.path(
  biological_enrichment_output_dir,
  "RData"
)

biological_enrichment_tables_dir <- file.path(
  biological_enrichment_output_dir,
  "tables"
)

biological_enrichment_reports_dir <- file.path(
  biological_enrichment_output_dir,
  "reports"
)

biological_enrichment_logfile <- file.path(
  biological_enrichment_logs_dir,
  "biological_enrichment_pipeline_log.txt"
)


# ==============================================================================
# 7. Ensure required directories exist
# ==============================================================================

dir.create(
  biological_enrichment_logs_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  biological_enrichment_output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  biological_enrichment_rdata_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  biological_enrichment_tables_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  biological_enrichment_reports_dir,
  recursive = TRUE,
  showWarnings = FALSE
)
