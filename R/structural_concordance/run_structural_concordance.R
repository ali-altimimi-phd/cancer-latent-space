# ==============================================================================
# File: R/structural_concordance/run_structural_concordance.R
# Purpose: Driver for the structural concordance layer
# Project: Global Cancer Complexity / Structural Inference
# ------------------------------------------------------------------------------
# Notes:
#   - Run this after the structural synthesis script and DuckDB ingestion have
#     created vw_structural_phenotype_wide_with_latent_overlay.
#   - This driver creates canonical concordance, quadrant assignment,
#     cross-engine correlation, and quadrant stability tables.
# ==============================================================================

suppressPackageStartupMessages({
  library(here)
})

source(here::here(
  "R",
  "structural_concordance",
  "build_structural_concordance_tables.R"
))

tables <- build_structural_concordance_tables(
  db_path = here::here(
    "output",
    "global_cancer",
    "warehouse",
    "global_cancer_results.duckdb"
  ),
  output_dir = here::here(
    "output",
    "global_cancer",
    "structural_inference",
    "tables",
    "concordance"
  ),
  synthesis_view = "vw_structural_phenotype_wide_with_latent_overlay",

  # Canonical four-quadrant axes.
  # Change x_metric here if the formal quadrant definition later uses
  # composite_kappa_delta or another complexity descriptor instead of
  # effective-rank delta.
  x_metric = "complexity__effrank_delta",
  y_metric = "mp__spectral_entropy_delta",

  center_method = "zero",
  boundary_tolerance = 0,
  correlation_method = "spearman",
  min_complete = 5,
  materialize_to_duckdb = TRUE,
  write_csv = TRUE,
  write_rds = TRUE
)

message("✅ Structural concordance tables available in:")
message("   output/global_cancer/structural_inference/tables/concordance")
