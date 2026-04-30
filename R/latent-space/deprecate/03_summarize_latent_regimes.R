#' Summarize Latent Geometric Regimes
#'
#' @description
#' Loads the latent taxonomy table produced by
#' `02_build_latent_taxonomy_table.R`, computes regime-level summary tables,
#' generates a compact narrative-ready summary, and exports the resulting tables
#' for downstream interpretation.
#'
#' @details
#' This script is the third stage of the postprocessing pipeline. It is designed
#' to be pipeline-safe: all required inputs are loaded explicitly from disk and
#' all outputs are written explicitly to disk.
#'
#' @section Input:
#' \itemize{
#'   \item `output/tables/postprocessing/latent_taxonomy_table.csv`
#' }
#'
#' @section Outputs:
#' \itemize{
#'   \item `output/tables/postprocessing/latent_regime_summary_table.csv`
#'   \item `output/tables/postprocessing/latent_regime_by_transformation_table.csv`
#'   \item `output/tables/postprocessing/latent_regime_summary_narrative.csv`
#' }
#'
#' @keywords internal
#' @noRd

suppressPackageStartupMessages({
  library(here)
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
})

source(here::here("R/helpers/pipeline_logger.R"))

log_dir <- here::here("projects", "cancer-latent-space", "output", "logs")
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

if (!exists("logger")) {
  logger <- start_log(logfile = file.path(log_dir, "03_summarize_latent_regimes.log"))
}

library(tidyverse)
library(here)

cat("[03] Summarizing latent regimes\n")

in_path <- here("projects/cancer-latent-space/output/tables/postprocessing/latent_taxonomy_table.csv")

table_dir <- here("projects/cancer-latent-space/output/tables/postprocessing")
plot_dir  <- here("projects/cancer-latent-space/output/plots/postprocessing")

dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

df <- read_csv(in_path)

# ---- Summary counts ----
summary_counts <- df %>%
  count(separation_class, regime)

write_csv(summary_counts, file.path(table_dir, "regime_counts.csv"))

# ---- Cross-tab ----
cross_tab <- df %>%
  count(complexity_direction, separation_class)

write_csv(cross_tab, file.path(table_dir, "complexity_vs_separation.csv"))

# ---- Plot ----
p <- ggplot(df, aes(x = separation_class, fill = complexity_direction)) +
  geom_bar(position = "dodge") +
  theme_minimal() +
  labs(title = "Separation vs Complexity Direction")

ggsave(
  filename = file.path(plot_dir, "separation_vs_complexity.png"),
  plot = p,
  width = 8,
  height = 5
)

cat("[03] Saved summaries and plots\n")