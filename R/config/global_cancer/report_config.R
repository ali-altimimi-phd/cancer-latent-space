# ------------------------------------------------------------------------------
# File: report_config.R
# Purpose:
# Role: Report configuration file
# Pipeline: Report
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

# ==============================================================================
# 1. Study identifier
# ==============================================================================

# Used to define directory structure for multiple data sources
study_name <- "global_cancer"


# ==============================================================================
# 2. Report pipeline toggles
# ==============================================================================

run_summary_exports           <- TRUE # Toggle to enable or disable
run_pairwise_summary_html     <- TRUE
run_go_clustering             <- TRUE
run_comparison_qmd_generation <- TRUE
run_quarto                    <- FALSE

# Visualization code temporarily removed from pipeline
run_visualization <- FALSE


# ==============================================================================
# 3. Report parameters
# ==============================================================================

geo_chip_map <- list(
  hu35ksuba = "GPL98",
  hu6800    = "GPL80"
)

# Other choices: GO_MF and GO_UNSPECIFIED
go_mode <- "GO_BP"
go_similarity_cutoff <- 0.5

# ==============================================================================
# 4. Core output and logging paths
# ==============================================================================

logs_dir            <- here::here("output", study_name, "logs")
data_dir            <- here::here("output", study_name, "RData")
filtered_probes_dir <- here::here(data_dir, "filtered_probes")
aggregate_dir       <- here::here(data_dir, "aggregated")
summaries_dir       <- here::here(data_dir, "summaries")
reports_dir         <- here::here("output", study_name, "reports")
plots_dir           <- here::here("output", study_name, "plots")

reports_pipeline_logfile <- here::here(
  logs_dir,
  "reports",
  "reports_pipeline_log.txt"
)


# ==============================================================================
# 5. Input object paths
# ==============================================================================

eset_path <- here::here(
  data_dir,
  paste0(study_name, "_eset_list.RData")
)

annotations_path <- here::here(
  data_dir,
  "annotations",
  "full_chip_annotations.rds"
)

matrices_path <- here::here(
  data_dir,
  "expression_matrices",
  "global_cancer_matrix_maps.RData"
)

res_hu35ksuba_path <- here::here(
  filtered_probes_dir,
  "filtered_probes_hu35ksuba.rds"
)

res_hu6800_path <- here::here(
  filtered_probes_dir,
  "filtered_probes_hu6800.rds"
)

complexity_df_path <- here::here(
  aggregate_dir,
  "complexity_aggregated_results.rds"
)

entropy_df_path <- here::here(
  aggregate_dir,
  "entropy_aggregated_results.rds"
)

summaries_df_path <- here::here(
  summaries_dir,
  "summaries_combined_df.rds"
)


# ==============================================================================
# 6. Ensure required directories exist
# ==============================================================================

dir.create(dirname(reports_pipeline_logfile), recursive = TRUE, showWarnings = FALSE)
dir.create(reports_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(aggregate_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(summaries_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(filtered_probes_dir, recursive = TRUE, showWarnings = FALSE)
