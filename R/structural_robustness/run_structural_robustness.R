# ==============================================================================
# Run Structural Robustness Profiling
# ==============================================================================
#
# Purpose:
#   Build the structural robustness layer after synthesis, concordance, and
#   validation.
#
# Pipeline position:
#
#   structural_synthesis
#     -> structural_concordance
#     -> structural_validation
#     -> structural_robustness
#
# Outputs:
#   CSV:
#     structural_robustness_boundary_assignments.csv
#     structural_robustness_summary.csv
#     structural_robustness_trajectories.csv
#
#   RDS:
#     structural_robustness_results.rds
#
#   DuckDB:
#     structural_robustness_boundary_assignments
#     structural_robustness_summary
#     structural_robustness_trajectories
#
# ==============================================================================


suppressPackageStartupMessages({
  library(here)
  library(dplyr)
  library(readr)
  library(duckdb)
  library(DBI)
})


# ==============================================================================
# 1. Configuration
# ==============================================================================

study_name <- "global_cancer"

warehouse_path <- here::here(
  "output",
  study_name,
  "warehouse",
  "global_cancer_results.duckdb"
)

source_view <- "vw_structural_phenotype_wide_with_latent_overlay"

output_dir <- here::here(
  "output",
  study_name,
  "structural_inference",
  "tables",
  "robustness"
)

rds_output_dir <- here::here(
  "output",
  study_name,
  "structural_inference",
  "RData",
  "robustness"
)

scale_within <- "chip_filter"

write_to_duckdb <- TRUE


# ==============================================================================
# 2. Source helpers
# ==============================================================================

source(here::here(
  "R",
  "structural_robustness",
  "helpers",
  "structural_robustness_metrics.R"
))

source(here::here(
  "R",
  "structural_robustness",
  "helpers",
  "structural_robustness_duckdb.R"
))


# ==============================================================================
# 3. Prepare output directories
# ==============================================================================

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(rds_output_dir, recursive = TRUE, showWarnings = FALSE)


# ==============================================================================
# 4. Load input data from DuckDB
# ==============================================================================

message("🔌 Connecting to DuckDB warehouse...")

con <- connect_structural_warehouse(warehouse_path)

message("📥 Loading structural phenotype data...")

robustness_input <- load_structural_robustness_input(
  con = con,
  source_view = source_view
)


# ==============================================================================
# 5. Boundary-distance profiling
# ==============================================================================

message("📐 Computing boundary-distance metrics...")

boundary_assignments <- robustness_input %>%
  add_boundary_distance_metrics(
    x_col = "complexity_delta",
    y_col = "spectral_entropy_delta",
    scale_within = scale_within
  )


# ==============================================================================
# 6. Margin profile
# ==============================================================================

message("🧭 Summarising margin-aware profiles...")

margin_summary <- boundary_assignments %>%
  summarise_margin_profile()


# ==============================================================================
# 7. Instability decomposition
# ==============================================================================

message("🧩 Decomposing chip/filter/sign instability...")

instability_summary <- boundary_assignments %>%
  compute_instability_decomposition()


# ==============================================================================
# 8. Cross-regime trajectories
# ==============================================================================

message("🧬 Computing cross-regime structural trajectories...")

trajectory_summary <- boundary_assignments %>%
  compute_cross_regime_trajectories()


# ==============================================================================
# 9. Structural confidence scoring
# ==============================================================================

message("🏷️ Assigning margin-aware robustness classes...")

robustness_summary <- margin_summary %>%
  dplyr::left_join(instability_summary, by = "comparison") %>%
  dplyr::left_join(trajectory_summary, by = "comparison") %>%
  assign_margin_aware_class()


# ==============================================================================
# 10. Write outputs
# ==============================================================================

message("💾 Writing robustness outputs...")

readr::write_csv(
  boundary_assignments,
  file.path(output_dir, "structural_robustness_boundary_assignments.csv")
)

readr::write_csv(
  robustness_summary,
  file.path(output_dir, "structural_robustness_summary.csv")
)

readr::write_csv(
  trajectory_summary,
  file.path(output_dir, "structural_robustness_trajectories.csv")
)

saveRDS(
  list(
    boundary_assignments = boundary_assignments,
    robustness_summary = robustness_summary,
    trajectory_summary = trajectory_summary
  ),
  file.path(rds_output_dir, "structural_robustness_results.rds")
)


# ==============================================================================
# 11. Optional DuckDB materialization
# ==============================================================================

if (isTRUE(write_to_duckdb)) {
  message("🦆 Writing robustness tables to DuckDB...")
  
  write_structural_robustness_tables(
    con = con,
    boundary_assignments = boundary_assignments,
    robustness_summary = robustness_summary,
    trajectory_summary = trajectory_summary,
    overwrite = TRUE
  )
}


# ==============================================================================
# 12. Disconnect
# ==============================================================================

disconnect_structural_warehouse(con)

message("✅ Structural robustness profiling complete.")