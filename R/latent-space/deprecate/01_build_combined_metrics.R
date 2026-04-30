#' Build Combined Dissertation and Latent Metrics Table
#'
#' @description
#' Loads cleaned dissertation-era complexity and entropy outputs together with
#' Notebook 5 latent comparison metrics, constructs a base master table for the
#' selected analysis slice, and exports a combined table for downstream latent
#' taxonomy and summary scripts.
#'
#' @details
#' This script is the first stage of the post-Notebook-5 postprocessing pipeline.
#' It is designed to be deterministic and pipeline-safe: all required inputs are
#' loaded explicitly from disk, outputs are written to disk, and no reliance is
#' placed on objects already existing in memory.
#'
#' The script currently uses the base dissertation slice defined by:
#' \itemize{
#'   \item `mode == "ALL"`
#'   \item `gene_set_name == "ALL"`
#'   \item `chip == "hu35ksuba"`
#' }
#'
#' @section Inputs:
#' \itemize{
#'   \item `output/global_cancer/reports/complexity_cleaned.csv`
#'   \item `output/global_cancer/reports/entropy_cleaned.csv`
#'   \item `projects/cancer-latent-space/output/tables/latent/latent_comparison_metrics.csv`
#' }
#'
#' @section Outputs:
#' \itemize{
#'   \item `projects/cancer-latent-space/output/tables/postprocessing/dissertation_master.csv`
#'   \item `projects/cancer-latent-space/output/tables/postprocessing/master_combined_metrics.csv`
#'   \item `projects/cancer-latent-space/output/tables/postprocessing/master_combined_metrics.rds`
#'   \item `projects/cancer-latent-space/output/logs/01_build_combined_metrics.log`
#' }
#'
#' @seealso [dplyr::left_join()]
#' @keywords internal
#' @noRd

suppressPackageStartupMessages({
  library(here)
  library(readr)
  library(dplyr)
  library(tidyverse)
})

source(here::here("R/helpers/pipeline_logger.R"))

log_dir <- here::here("projects", "cancer-latent-space", "output", "logs")
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

if (!exists("logger")) {
  logger <- start_log(logfile = file.path(log_dir, "01_build_combined_metrics.log"))
}

cat("[01] Building combined metrics table\n")

out_dir <- here("projects/cancer-latent-space/output/tables/postprocessing")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Load data ----
latent_n5 <- read_csv(here("projects/cancer-latent-space/output/tables/latent/latent_comparison_metrics.csv"))
latent_n6 <- read_csv(here("projects/cancer-latent-space/output/tables/latent/notebook_06/final_notebook_06_group_summary.csv"))

entropy <- read_csv(here("output/global_cancer/reports/entropy_cleaned.csv"))
complexity <- read_csv(here("output/global_cancer/reports/complexity_cleaned.csv"))

# ---- Clean keys ----
clean_key <- function(x) {
  x %>%
    as.character() %>%
    trimws() %>%
    toupper() %>%
    gsub(" ", "", ., fixed = TRUE)
}

latent_n5$comparison  <- clean_key(latent_n5$comparison)
latent_n6$comparison  <- clean_key(latent_n6$comparison)
entropy$comparison    <- clean_key(entropy$comparison)
complexity$comparison <- clean_key(complexity$comparison)

# ---- Reduce entropy/complexity to ONE row per comparison ----
entropy_summary <- entropy %>%
  filter(
    toupper(mode) == "ALL",
    toupper(gene_set_normalized) == "ALL",
    toupper(gene_set_name) == "ALL"
  ) %>%
  select(any_of(c(
    "comparison",
    "shannon_1", "shannon_2", "shannon_delta",
    "spectral_1", "spectral_2", "spectral_delta",
    "direction", "p_perm"
  ))) %>%
  distinct(comparison, .keep_all = TRUE)

if ("direction" %in% names(entropy_summary)) {
  entropy_summary <- entropy_summary %>%
    rename(entropy_direction = direction)
}

if ("p_perm" %in% names(entropy_summary)) {
  entropy_summary <- entropy_summary %>%
    rename(entropy_p_perm = p_perm)
}

complexity_summary <- complexity %>%
  filter(
    toupper(mode) == "ALL",
    toupper(gene_set_normalized) == "ALL",
    toupper(gene_set_name) == "ALL"
  ) %>%
  select(any_of(c(
    "comparison",
    "svd κ 1", "svd κ 2", "δ svd κ",
    "effrank_1", "effrank_2",
    "κ_composite_1", "κ_composite_2",
    "direction", "p_perm"
  ))) %>%
  distinct(comparison, .keep_all = TRUE)

if ("direction" %in% names(complexity_summary)) {
  complexity_summary <- complexity_summary %>%
    rename(complexity_direction_legacy = direction)
}

if ("p_perm" %in% names(complexity_summary)) {
  complexity_summary <- complexity_summary %>%
    rename(complexity_p_perm = p_perm)
}

# Optional QC
cat("[01] entropy_summary rows   :", nrow(entropy_summary), "\n")
cat("[01] complexity_summary rows:", nrow(complexity_summary), "\n")

# ---- Merge ----
combined <- latent_n5 %>%
  left_join(latent_n6, by = "comparison", suffix = c("_n5", "_n6")) %>%
  left_join(entropy_summary, by = "comparison") %>%
  left_join(complexity_summary, by = "comparison")

# ---- Basic checks ----
cat("[01] Rows:", nrow(combined), "\n")
cat("[01] Unique comparisons:", n_distinct(combined$comparison), "\n")

# ---- Save ----
write_csv(combined, file.path(out_dir, "master_combined_metrics.csv"))

cat("[01] Saved master_combined_metrics.csv\n")

logger$log("Stage 01 completed successfully.", section = "PIPELINE")