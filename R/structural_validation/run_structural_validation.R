#' Run Structural Validation Layer
#'
#' Builds cross-engine calibration and discordance mapping tables from the
#' structural phenotype warehouse. This layer should be run after structural
#' synthesis and before biological interpretation.
#'
#' Inputs:
#'   - DuckDB warehouse with vw_structural_phenotype_wide_with_latent_overlay
#'
#' Outputs:
#'   - cross_engine_metric_correlations
#'   - cross_engine_directional_concordance
#'   - cross_engine_discordance_archetypes
#'   - validation summary views
#'   - CSV exports under output/global_cancer/structural_inference/tables/validation

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(dplyr)
  library(readr)
  library(here)
})

# ==============================================================================
# Configuration
# ==============================================================================

warehouse_path <- here::here(
  "output/global_cancer/warehouse/global_cancer_results.duckdb"
)

validation_output_dir <- here::here(
  "output/global_cancer/structural_inference/tables/validation"
)

dir.create(validation_output_dir, recursive = TRUE, showWarnings = FALSE)

# Near-zero tolerance for directional sign calls.
#
# Recommended workflow:
#   1. Start with "fixed" and eps_fixed = 1e-8 for debugging.
#   2. Re-run with "scaled" once all columns are confirmed.
#
# The scaled threshold reduces false discordance caused by tiny numerical deltas.

direction_eps_mode <- "fixed"  # "fixed" or "scaled"
eps_fixed          <- 1e-8
eps_scaled_factor  <- 0.10

# Whether latent anisotropy should be inverted for directional concordance.
#
# Rationale:
#   Increasing anisotropy usually means stronger directional constraint.
#   If the other engines define positive values as expansion/dispersion, then
#   latent anisotropy should be multiplied by -1 before sign classification.

invert_latent_anisotropy_for_direction <- TRUE

# ==============================================================================
# Connect
# ==============================================================================

con <- DBI::dbConnect(duckdb::duckdb(), warehouse_path)
on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)

message("Structural validation layer started.")
message("Warehouse: ", warehouse_path)

# ==============================================================================
# Source stages
# ==============================================================================

source(here::here("R/structural_validation/01_build_cross_engine_calibration.R"))
source(here::here("R/structural_validation/02_build_engine_discordance_map.R"))
source(here::here("R/structural_validation/03_summarize_discordance_archetypes.R"))

# ==============================================================================
# Run stages
# ==============================================================================

calibration_results <- build_cross_engine_calibration(
  con = con,
  output_dir = validation_output_dir
)

concordance_results <- build_engine_discordance_map(
  con = con,
  output_dir = validation_output_dir,
  eps_mode = direction_eps_mode,
  eps_fixed = eps_fixed,
  eps_scaled_factor = eps_scaled_factor,
  invert_latent_anisotropy = invert_latent_anisotropy_for_direction
)

archetype_results <- summarize_discordance_archetypes(
  con = con,
  output_dir = validation_output_dir
)

message("Structural validation layer completed.")
message("CSV outputs written to: ", validation_output_dir)
