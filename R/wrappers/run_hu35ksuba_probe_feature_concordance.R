# ------------------------------------------------------------------------------
# File: run_hu35ksuba_probe_feature_concordance.R
# Purpose: Run hu35ksuba probe feature-space concordance analysis and export
#   Quarto-ready resources.
# Role: Wrapper / executable runner
# Pipeline: Reporting / Feature-space concordance
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

if (!requireNamespace("here", quietly = TRUE)) {
  stop("Package 'here' is required.", call. = FALSE)
}

source(here::here("R", "feature-space", "generate_probe_feature_concordance.R"))
source(here::here("R", "feature-space", "export_probe_feature_concordance_to_quarto.R"))

chip <- "hu35ksuba"

result <- generate_probe_feature_concordance(
  filtered_probes_path = here::here(
    "output", "global_cancer", "RData", "filtered_probes",
    "filtered_probes_hu35ksuba.rds"
  ),
  latent_features_path = here::here(
    "data", "global_cancer", "processed", "ml_inputs",
    "hu35ksuba_top3000_variance_features.csv"
  ),
  chip = chip,
  output_table_dir = here::here("output", "global_cancer", "tables", "probe_concordance"),
  output_plot_dir = here::here("output", "global_cancer", "plots", "probe_concordance"),
  plot_utils_path = here::here("R", "helpers", "plot_utils.R"),
  
  annotation_path = here::here(
    "output", "global_cancer", "RData", "annotations",
    "full_chip_annotations.rds"
  ),
  write_annotated_rds = FALSE
)

exported <- export_probe_feature_concordance_to_quarto(
  concordance_tbl = result$concordance_table,
  group_summary_tbl = result$group_summary,
  chip = chip,
  source_table_dir = here::here("output", "global_cancer", "tables", "probe_concordance"),
  source_plot_dir = here::here("output", "global_cancer", "plots", "probe_concordance"),
  quarto_table_dir = here::here("quarto", "resources", "tables", "probe_concordance"),
  quarto_plot_dir = here::here("quarto", "resources", "plots", "probe_concordance"),
  plot_utils_path = here::here("R", "helpers", "plot_utils.R")
)

invisible(list(result = result, exported = exported))
