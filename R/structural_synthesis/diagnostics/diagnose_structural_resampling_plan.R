# ==============================================================================
# File: diagnose_structural_resampling_plan.R
# Purpose: Build a resampling diagnostic grid from structural complexity/entropy
#   result RDS files.
# Role: Standalone diagnostic planning utility
# Project: Global Cancer Structural Inference Framework
# ==============================================================================

suppressPackageStartupMessages({
  library(here)
  library(dplyr)
  library(readr)
  library(tibble)
  library(tidyr)
  library(DBI)
  library(duckdb)
})

# ---- Paths --------------------------------------------------------------------

structural_rdata_dir <- here::here(
  "output",
  "global_cancer",
  "structural_inference",
  "RData"
)

diagnostic_dir <- here::here(
  "output",
  "global_cancer",
  "diagnostics",
  "structural_resampling"
)

duckdb_path <- here::here(
  "output",
  "global_cancer",
  "warehouse",
  "global_cancer_results.duckdb"
)

dir.create(diagnostic_dir, recursive = TRUE, showWarnings = FALSE)

complexity_path <- file.path(structural_rdata_dir, "complexity_results.rds")
entropy_path    <- file.path(structural_rdata_dir, "entropy_results.rds")

# ---- Diagnostic settings -------------------------------------------------------

resampling_grid <- c(25, 50, 100, 250)

# ---- Load structural result tables --------------------------------------------

complexity_tbl <- readRDS(complexity_path)
entropy_tbl    <- readRDS(entropy_path)

if (is.list(complexity_tbl) && "summary" %in% names(complexity_tbl)) {
  complexity_tbl <- complexity_tbl$summary
}

if (is.list(entropy_tbl) && "summary" %in% names(entropy_tbl)) {
  entropy_tbl <- entropy_tbl$summary
}

complexity_tbl <- as_tibble(complexity_tbl)
entropy_tbl    <- as_tibble(entropy_tbl)

# ---- Standardize structural diagnostic frame ----------------------------------

structural_complexity_frame <- complexity_tbl |>
  transmute(
    engine = "complexity",
    chip = chip,
    filter_regime = filter_regime,
    group = group,
    comparison = comparison,
    n_features = dplyr::coalesce(
      as.integer(n_features),
      as.integer(n_shared_probes)
    ),
    primary_metric = "kappa_delta",
    primary_delta = as.numeric(kappa_delta),
    abs_primary_delta = abs(as.numeric(kappa_delta))
  )

structural_entropy_frame <- entropy_tbl |>
  transmute(
    engine = "entropy",
    chip = chip,
    filter_regime = filter_regime,
    group = group,
    comparison = comparison,
    n_features = dplyr::coalesce(
      as.integer(n_features),
      as.integer(n_shared_probes)
    ),
    primary_metric = "spectral_delta",
    primary_delta = as.numeric(spectral_delta),
    abs_primary_delta = abs(as.numeric(spectral_delta))
  )

structural_resampling_manifest <- bind_rows(
  structural_complexity_frame,
  structural_entropy_frame
) |>
  filter(
    !is.na(chip),
    !is.na(filter_regime),
    !is.na(group),
    !is.na(comparison),
    !is.na(primary_delta)
  ) |>
  group_by(engine, chip, filter_regime) |>
  mutate(
    q25_abs_delta = quantile(abs_primary_delta, 0.25, na.rm = TRUE),
    q50_abs_delta = quantile(abs_primary_delta, 0.50, na.rm = TRUE),
    q75_abs_delta = quantile(abs_primary_delta, 0.75, na.rm = TRUE),
    effect_size_class = case_when(
      abs_primary_delta <= q25_abs_delta ~ "small",
      abs_primary_delta <= q50_abs_delta ~ "medium_low",
      abs_primary_delta <= q75_abs_delta ~ "medium_high",
      TRUE ~ "large"
    )
  ) |>
  ungroup() |>
  arrange(engine, chip, filter_regime, group, comparison) |>
  mutate(diagnostic_id = row_number()) |>
  select(
    diagnostic_id,
    engine,
    chip,
    filter_regime,
    group,
    comparison,
    n_features,
    primary_metric,
    primary_delta,
    abs_primary_delta,
    effect_size_class
  )

structural_resampling_grid <- tidyr::crossing(
  structural_resampling_manifest,
  n_resamples = resampling_grid
) |>
  mutate(
    run_permutation = TRUE,
    run_bootstrap = TRUE,
    permutation_metric = case_when(
      engine == "complexity" ~ "kappa",
      engine == "entropy" ~ "spectral",
      TRUE ~ NA_character_
    ),
    bootstrap_metric = case_when(
      engine == "complexity" ~ "kappa",
      engine == "entropy" ~ "spectral",
      TRUE ~ NA_character_
    ),
    permutation_unit = "sample_label",
    bootstrap_unit = "sample",
    covariance_space = "sample",
    seed = 20260514L + diagnostic_id * 1000L + n_resamples
  )

structural_resampling_workload_summary <- structural_resampling_grid |>
  group_by(engine, chip, filter_regime, n_resamples) |>
  summarise(
    n_diagnostic_units = n(),
    median_n_features = median(n_features, na.rm = TRUE),
    q75_n_features = quantile(n_features, 0.75, na.rm = TRUE),
    max_n_features = max(n_features, na.rm = TRUE),
    .groups = "drop"
  )

structural_resampling_config_candidates <- tibble(
  parameter = c(
    "run_structural_complexity_permutation",
    "run_structural_complexity_bootstrap",
    "run_structural_entropy_permutation",
    "run_structural_entropy_bootstrap",
    "complexity_permutation_metric",
    "complexity_bootstrap_metric",
    "entropy_permutation_metric",
    "entropy_bootstrap_metric",
    "complexity_n_perm",
    "complexity_n_boot",
    "entropy_n_perm",
    "entropy_n_boot",
    "permutation_unit",
    "bootstrap_unit",
    "covariance_space"
  ),
  exploratory_candidate = c(
    "TRUE", "TRUE", "TRUE", "TRUE",
    "kappa", "kappa", "spectral", "spectral",
    "100", "100", "100", "100",
    "sample_label", "sample", "sample"
  ),
  stability_candidate = c(
    "TRUE", "TRUE", "TRUE", "TRUE",
    "kappa", "kappa", "spectral", "spectral",
    "250", "250", "250", "250",
    "sample_label", "sample", "sample"
  )
)

# ---- Write CSV outputs ---------------------------------------------------------

write_csv(
  structural_resampling_manifest,
  file.path(diagnostic_dir, "structural_resampling_manifest.csv")
)

write_csv(
  structural_resampling_grid,
  file.path(diagnostic_dir, "structural_resampling_grid.csv")
)

write_csv(
  structural_resampling_workload_summary,
  file.path(diagnostic_dir, "structural_resampling_workload_summary.csv")
)

write_csv(
  structural_resampling_config_candidates,
  file.path(diagnostic_dir, "structural_resampling_config_candidates.csv")
)

# ---- Write DuckDB diagnostic tables -------------------------------------------

con <- DBI::dbConnect(
  duckdb::duckdb(),
  dbdir = duckdb_path,
  read_only = FALSE
)

on.exit({
  try(DBI::dbDisconnect(con, shutdown = TRUE), silent = TRUE)
}, add = TRUE)

DBI::dbWriteTable(
  con,
  "diagnostic_structural_resampling_manifest",
  structural_resampling_manifest,
  overwrite = TRUE
)

DBI::dbWriteTable(
  con,
  "diagnostic_structural_resampling_grid",
  structural_resampling_grid,
  overwrite = TRUE
)

DBI::dbWriteTable(
  con,
  "diagnostic_structural_resampling_workload_summary",
  structural_resampling_workload_summary,
  overwrite = TRUE
)

DBI::dbWriteTable(
  con,
  "diagnostic_structural_resampling_config_candidates",
  structural_resampling_config_candidates,
  overwrite = TRUE
)

DBI::dbExecute(
  con,
  "
  CREATE OR REPLACE VIEW vw_diagnostic_structural_resampling_inventory AS
  SELECT
      'diagnostic_structural_resampling_manifest' AS table_name,
      COUNT(*) AS n_rows
  FROM diagnostic_structural_resampling_manifest

  UNION ALL

  SELECT
      'diagnostic_structural_resampling_grid' AS table_name,
      COUNT(*) AS n_rows
  FROM diagnostic_structural_resampling_grid

  UNION ALL

  SELECT
      'diagnostic_structural_resampling_workload_summary' AS table_name,
      COUNT(*) AS n_rows
  FROM diagnostic_structural_resampling_workload_summary

  UNION ALL

  SELECT
      'diagnostic_structural_resampling_config_candidates' AS table_name,
      COUNT(*) AS n_rows
  FROM diagnostic_structural_resampling_config_candidates
  "
)

message("✅ Wrote structural resampling diagnostics to: ", diagnostic_dir)

print(
  DBI::dbGetQuery(
    con,
    "SELECT * FROM vw_diagnostic_structural_resampling_inventory"
  )
)

# ==============================================================================
# Notes
#
# This script uses structural comparison-level result tables:
#
#   - complexity_results.rds
#   - entropy_results.rds
#
# These are small comparison-level tibbles, typically 53 rows per engine, and are
# more appropriate for broad resampling-parameter calibration than the much larger
# biological enrichment outputs.
#
# This script does not execute resampling. It builds the manifest/grid used to
# plan structural convergence diagnostics and to justify candidate settings such
# as n_perm = 100 and n_boot = 100.
#
# Biological enrichment DuckDB diagnostic tables from the earlier approach are
# not required by this refactored structural-RDS workflow.
#
# ==============================================================================