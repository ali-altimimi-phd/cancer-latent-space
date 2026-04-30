#' Build a Latent Taxonomy Table from Combined Metrics
#'
#' @description
#' Loads the combined dissertation + latent metrics table produced by
#' `01_build_combined_metrics.R`, assigns each comparison to a geometric regime,
#' classifies movement versus reorganization patterns, adds secondary shape
#' modifiers, and exports a clean taxonomy table for downstream summaries.
#'
#' @details
#' The taxonomy is based primarily on the joint sign pattern of:
#' \itemize{
#'   \item `effrank_delta`
#'   \item `pr_delta`
#' }
#'
#' Secondary descriptors are derived from:
#' \itemize{
#'   \item `kappa_delta`
#'   \item `anisotropy_delta`
#'   \item `centroid_distance`
#' }
#'
#' Transformation type is defined using median splits on:
#' \itemize{
#'   \item `centroid_distance`
#'   \item `abs(effrank_delta)`
#' }
#'
#' @section Input:
#' \itemize{
#'   \item `output/tables/postprocessing/master_combined_metrics.csv`
#' }
#'
#' @section Output:
#' \itemize{
#'   \item `output/tables/postprocessing/latent_taxonomy_table.csv`
#'   \item `output/tables/postprocessing/latent_taxonomy_table.rds`
#' }
#'
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
  logger <- start_log(logfile = file.path(log_dir, "02_build_latent_taxonomy_table.log"))
}

cat("[02] Building latent taxonomy table\n")

in_path <- here("projects/cancer-latent-space/output/tables/postprocessing/master_combined_metrics.csv")
out_dir <- here("projects/cancer-latent-space/output/tables/postprocessing")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read_csv(in_path, show_col_types = FALSE)

# old function: too strict at the top and too permissive in the middle
# classify_sep <- function(sil, nn) {
#   if (is.na(sil) || is.na(nn)) return("unknown")
#   if (sil > 0.5 && nn < 0.2) return("well-separated")
#   if (sil > 0.25 && nn < 0.4) return("moderate")
#   if (sil > 0) return("weak")
#   return("overlap")
# }

# new function:
sil_q <- quantile(df$mean_silhouette, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)
nn_q  <- quantile(df$mean_same_group_neighbor_fraction, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)

classify_sep <- function(sil, nn) {
  if (is.na(sil) || is.na(nn)) return("unknown")
  
  # strongest: high global separation AND local cohesion
  if (sil >= sil_q[[3]] && nn >= nn_q[[3]]) {
    return("relatively_separated")
  }
  
  # moderate: above-median silhouette OR above-median cohesion
  if (sil >= sil_q[[2]] || nn >= nn_q[[2]]) {
    return("moderately_structured")
  }
  
  # weak: above lower quartile in either dimension
  if (sil >= sil_q[[1]] || nn >= nn_q[[1]]) {
    return("weak_structure")
  }
  
  # lowest: below both thresholds
  return("overlap")
}

# diagnostics
cat("sil_q: ", sil_q, "\n")
cat("nn_q: ",nn_q, "\n")

# Use an available complexity delta column.
delta_col <- NULL
if ("effrank_delta" %in% names(df)) {
  delta_col <- "effrank_delta"
} else if ("δ svd κ" %in% names(df)) {
  delta_col <- "δ svd κ"
} else if ("pr_delta_n5" %in% names(df)) {
  delta_col <- "pr_delta_n5"
} else {
  stop("No usable complexity delta column found. Expected one of: effrank_delta, δ svd κ, pr_delta_n5")
}

cat("[02] Using complexity delta column:", delta_col, "\n")

df <- df %>%
  mutate(
    separation_class = mapply(
      classify_sep,
      mean_silhouette,
      mean_same_group_neighbor_fraction
    ),
    complexity_delta_used = .data[[delta_col]],
    complexity_direction = case_when(
      is.na(complexity_delta_used) ~ "unknown",
      complexity_delta_used > 0 ~ "increase",
      complexity_delta_used < 0 ~ "decrease",
      TRUE ~ "no_change"
    ),
    regime = case_when(
      complexity_direction == "increase" & separation_class == "well-separated" ~ "structured_divergence",
      complexity_direction == "increase" & separation_class == "overlap" ~ "noisy_expansion",
      complexity_direction == "decrease" & separation_class == "well-separated" ~ "constrained_specialization",
      complexity_direction == "decrease" & separation_class == "overlap" ~ "degenerate_overlap",
      TRUE ~ "mixed"
    )
  )

write_csv(df, file.path(out_dir, "latent_taxonomy_table.csv"))

cat("[02] Saved latent_taxonomy_table.csv\n")
logger$log("Stage 02 completed successfully.", section = "PIPELINE")