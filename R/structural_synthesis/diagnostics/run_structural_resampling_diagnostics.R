# ==============================================================================
# File: run_structural_resampling_diagnostics.R
# Purpose: Execute structural resampling diagnostics from structural grid
# Role: Standalone diagnostic execution utility
# Project: Global Cancer Structural Inference Framework
# ==============================================================================

suppressPackageStartupMessages({
  library(here)
  library(DBI)
  library(duckdb)
  library(dplyr)
  library(readr)
  library(tibble)
  library(purrr)
})

# ---- Source configuration ------------------------------------------------------

source(
  here::here("R/config/global_cancer/structural_inference_config.R"),
  local = FALSE
)

# ---- Paths --------------------------------------------------------------------

duckdb_path <- here::here(
  "output",
  study_name,
  "warehouse",
  "global_cancer_results.duckdb"
)

diagnostic_dir <- here::here(
  "output",
  study_name,
  "diagnostics",
  "structural_resampling"
)

dir.create(diagnostic_dir, recursive = TRUE, showWarnings = FALSE)

message("📂 Project root: ", here::here())
message("🦆 DuckDB path: ", duckdb_path)
message("📂 Diagnostic output dir: ", diagnostic_dir)

# ---- Source project code -------------------------------------------------------

source(here::here("R/helpers/pipeline_logger.R"), local = FALSE)
source(load_pipeline_inputs_helper, local = FALSE)

source(here::here("R/helpers/structural/build_selection_regime_labels.R"), local = FALSE)
source(here::here("R/helpers/structural/resolve_comparison_matrices.R"), local = FALSE)

source(here::here("R/engines/complexity/core_complexity_metrics.R"), local = FALSE)
source(here::here("R/engines/complexity/statistical_complexity_helpers.R"), local = FALSE)
source(here::here("R/engines/complexity/compute_single_complexity.R"), local = FALSE)
source(here::here("R/engines/complexity/compare_pair_complexity.R"), local = FALSE)
source(here::here("R/engines/complexity/run_pairwise_complexity.R"), local = FALSE)
source(here::here("R/wrappers/run_complexity_structural_engine.R"), local = FALSE)

source(here::here("R/engines/entropy/core_entropy_metrics.R"), local = FALSE)
source(here::here("R/engines/entropy/statistical_entropy_helpers.R"), local = FALSE)
source(here::here("R/engines/entropy/compute_single_entropy.R"), local = FALSE)
source(here::here("R/engines/entropy/compare_pair_entropy.R"), local = FALSE)
source(here::here("R/engines/entropy/run_pairwise_entropy.R"), local = FALSE)
source(here::here("R/wrappers/run_entropy_structural_engine.R"), local = FALSE)

# ---- Settings -----------------------------------------------------------------

# Use small value for testing; set to Inf for full 424-row diagnostic.
max_diagnostic_rows <- Inf

diagnostic_engine_filter <- NULL
diagnostic_chip_filter <- NULL
diagnostic_filter_regime_filter <- NULL
diagnostic_n_resamples_filter <- NULL

# ---- Local helpers -------------------------------------------------------------

elapsed_seconds <- function(start_time) {
  as.numeric(difftime(Sys.time(), start_time, units = "secs"))
}

safe_numeric <- function(x) {
  suppressWarnings(as.numeric(x))
}

get_scalar <- function(tbl, candidates, default = NA_real_) {
  hit <- intersect(candidates, names(tbl))
  
  if (length(hit) == 0) {
    return(default)
  }
  
  val <- tbl[[hit[[1]]]]
  
  if (length(val) == 0) {
    return(default)
  }
  
  val[[1]]
}

extract_selected_probes_local <- function(x) {
  if (is.character(x)) {
    return(x)
  }
  
  if (!is.list(x)) {
    return(character())
  }
  
  preferred_names <- c(
    "selected_probes",
    "filtered_probes",
    "probe_ids",
    "probes",
    "probe_id",
    "ids",
    "features",
    "feature_ids",
    "selected_features"
  )
  
  nm <- names(x)
  
  if (!is.null(nm)) {
    for (candidate_name in intersect(preferred_names, nm)) {
      candidate <- x[[candidate_name]]
      
      if (is.character(candidate) || is.factor(candidate)) {
        return(as.character(candidate))
      }
      
      if (is.data.frame(candidate)) {
        for (col in names(candidate)) {
          if (is.character(candidate[[col]]) || is.factor(candidate[[col]])) {
            return(as.character(candidate[[col]]))
          }
        }
      }
    }
  }
  
  character()
}

make_diagnostic_logger <- function() {
  list(
    log = function(message, section = NULL) {
      if (is.null(section)) {
        base::message(message)
      } else {
        base::message(sprintf("[%s] %s", section, message))
      }
    }
  )
}

# ---- Resolve filter regimes ----------------------------------------------------

filter_regimes <- build_selection_regime_labels(
  selection_regimes = selection_regimes,
  variance_mode = variance_mode,
  variance_top_n = variance_top_n,
  variance_threshold = variance_threshold
)

message("🧪 Filter regimes: ", paste(filter_regimes, collapse = ", "))

# ---- Connect to DuckDB ---------------------------------------------------------

con <- DBI::dbConnect(
  duckdb::duckdb(),
  dbdir = duckdb_path,
  read_only = FALSE
)

on.exit({
  try(DBI::dbDisconnect(con, shutdown = TRUE), silent = TRUE)
}, add = TRUE)

# ---- Load diagnostic grid ------------------------------------------------------

diagnostic_grid <- DBI::dbGetQuery(
  con,
  "
  SELECT *
  FROM diagnostic_structural_resampling_grid
  ORDER BY diagnostic_id, n_resamples
  "
) |>
  as_tibble()

if (!is.null(diagnostic_engine_filter)) {
  diagnostic_grid <- diagnostic_grid |>
    filter(engine %in% diagnostic_engine_filter)
}

if (!is.null(diagnostic_chip_filter)) {
  diagnostic_grid <- diagnostic_grid |>
    filter(chip %in% diagnostic_chip_filter)
}

if (!is.null(diagnostic_filter_regime_filter)) {
  diagnostic_grid <- diagnostic_grid |>
    filter(filter_regime %in% diagnostic_filter_regime_filter)
}

if (!is.null(diagnostic_n_resamples_filter)) {
  diagnostic_grid <- diagnostic_grid |>
    filter(n_resamples %in% diagnostic_n_resamples_filter)
}

if (is.finite(max_diagnostic_rows)) {
  diagnostic_grid <- diagnostic_grid |>
    slice_head(n = max_diagnostic_rows)
}

if (nrow(diagnostic_grid) == 0) {
  stop("Diagnostic grid is empty after filtering.", call. = FALSE)
}

message("🧪 Structural diagnostic rows selected: ", nrow(diagnostic_grid))

# ---- Load structural inputs ----------------------------------------------------

diagnostic_logger <- make_diagnostic_logger()

load_pipeline_inputs(
  matrices_path = matrices_path,
  filtered_probes_dir = structural_filtered_probes_dir,
  chips = chips,
  filter_regimes = filter_regimes,
  require_matrix_maps = TRUE,
  require_filtered_probes = FALSE,
  logger = diagnostic_logger,
  overwrite = TRUE
)

matrix_lookup <- list()
comparison_lookup <- list()

if (exists("matrices_hu35ksuba", inherits = TRUE)) {
  matrix_lookup[["hu35ksuba"]] <- get("matrices_hu35ksuba", inherits = TRUE)
}

if (exists("matrices_hu6800", inherits = TRUE)) {
  matrix_lookup[["hu6800"]] <- get("matrices_hu6800", inherits = TRUE)
}

if (exists("comparison_map_hu35ksuba", inherits = TRUE)) {
  comparison_lookup[["hu35ksuba"]] <- get("comparison_map_hu35ksuba", inherits = TRUE)
}

if (exists("comparison_map_hu6800", inherits = TRUE)) {
  comparison_lookup[["hu6800"]] <- get("comparison_map_hu6800", inherits = TRUE)
}

missing_matrix_chips <- setdiff(chips, names(matrix_lookup))
missing_map_chips <- setdiff(chips, names(comparison_lookup))

if (length(missing_matrix_chips) > 0) {
  stop(
    "Missing matrix lists for chip(s): ",
    paste(missing_matrix_chips, collapse = ", "),
    call. = FALSE
  )
}

if (length(missing_map_chips) > 0) {
  stop(
    "Missing comparison maps for chip(s): ",
    paste(missing_map_chips, collapse = ", "),
    call. = FALSE
  )
}

# ---- Load selected probes for one row -----------------------------------------

load_selected_probes_for_row <- function(row) {
  filtered_path <- file.path(
    structural_filtered_probes_dir,
    sprintf(
      "filtered_probes_%s_%s.rds",
      row$chip,
      row$filter_regime
    )
  )
  
  if (!file.exists(filtered_path)) {
    stop("Filtered-probe file not found: ", filtered_path, call. = FALSE)
  }
  
  filtered_i <- readRDS(filtered_path)
  
  group_obj <- filtered_i[[row$group]]
  
  if (is.null(group_obj)) {
    stop(
      "Group not found in filtered probes: ",
      row$group,
      " | file: ",
      filtered_path,
      call. = FALSE
    )
  }
  
  comparison_obj <- group_obj[[row$comparison]]
  
  if (is.null(comparison_obj)) {
    stop(
      "Comparison not found in filtered probes: ",
      row$comparison,
      " | group: ",
      row$group,
      call. = FALSE
    )
  }
  
  selected_probes <- extract_selected_probes_local(comparison_obj)
  
  if (length(selected_probes) < min_selected_probes) {
    stop(
      "Fewer than min_selected_probes for ",
      row$chip,
      " / ",
      row$filter_regime,
      " / ",
      row$group,
      " / ",
      row$comparison,
      call. = FALSE
    )
  }
  
  selected_probes
}

# ---- Run one diagnostic row ----------------------------------------------------

run_single_structural_diagnostic <- function(row) {
  matrices_i <- matrix_lookup[[row$chip]]
  comparison_map_i <- comparison_lookup[[row$chip]]
  
  if (is.null(matrices_i)) {
    stop("Missing matrix lookup for chip: ", row$chip, call. = FALSE)
  }
  
  if (is.null(comparison_map_i)) {
    stop("Missing comparison lookup for chip: ", row$chip, call. = FALSE)
  }
  
  comparison_map_group <- comparison_map_i[[row$group]]
  
  if (is.null(comparison_map_group)) {
    stop("Missing comparison map group: ", row$group, call. = FALSE)
  }
  
  comparison_labels <- comparison_map_group[[row$comparison]]
  
  if (is.null(comparison_labels)) {
    stop("Missing comparison labels for: ", row$comparison, call. = FALSE)
  }
  
  selected_probes <- load_selected_probes_for_row(row)
  
  if (identical(row$engine, "complexity")) {
    return(
      compare_pair_complexity(
        matrices_i = matrices_i,
        comparison_labels = comparison_labels,
        selected_probes = selected_probes,
        comparison = row$comparison,
        group = row$group,
        chip = row$chip,
        filter_regime = row$filter_regime,
        complexity_fn = get_svd_kappa,
        
        run_permutation = TRUE,
        n_perm = as.integer(row$n_resamples),
        permutation_metric = row$permutation_metric,
        permutation_unit = row$permutation_unit,
        
        run_bootstrap = TRUE,
        n_boot = as.integer(row$n_resamples),
        bootstrap_metric = row$bootstrap_metric,
        bootstrap_unit = row$bootstrap_unit,
        
        covariance_space = row$covariance_space,
        seed = as.integer(row$seed)
      )
    )
  }
  
  if (identical(row$engine, "entropy")) {
    return(
      compare_pair_entropy(
        matrices_i = matrices_i,
        comparison_labels = comparison_labels,
        selected_probes = selected_probes,
        comparison = row$comparison,
        group = row$group,
        chip = row$chip,
        filter_regime = row$filter_regime,
        entropy_fn = compute_shannon_entropy,
        
        run_permutation = TRUE,
        n_perm = as.integer(row$n_resamples),
        
        run_bootstrap = TRUE,
        n_boot = as.integer(row$n_resamples),
        
        permutation_metric = row$permutation_metric,
        bootstrap_metric = row$bootstrap_metric,
        covariance_space = row$covariance_space,
        
        permutation_unit = row$permutation_unit,
        bootstrap_unit = row$bootstrap_unit,
        
        seed = as.integer(row$seed)
      )
    )
  }
  
  stop("Unknown engine: ", row$engine, call. = FALSE)
}

# ---- Summaries for row result --------------------------------------------------

summarise_success <- function(result_tbl, row, runtime_seconds) {
  result_tbl <- as_tibble(result_tbl)
  
  tibble(
    diagnostic_id = row$diagnostic_id,
    engine = row$engine,
    chip = row$chip,
    filter_regime = row$filter_regime,
    group = row$group,
    comparison = row$comparison,
    effect_size_class = row$effect_size_class,
    
    n_resamples = as.integer(row$n_resamples),
    n_features_expected = as.integer(row$n_features),
    primary_metric = row$primary_metric,
    observed_primary_delta = safe_numeric(row$primary_delta),
    abs_observed_primary_delta = safe_numeric(row$abs_primary_delta),
    
    permutation_metric = row$permutation_metric,
    bootstrap_metric = row$bootstrap_metric,
    permutation_unit = row$permutation_unit,
    bootstrap_unit = row$bootstrap_unit,
    covariance_space = row$covariance_space,
    seed = as.integer(row$seed),
    
    runtime_seconds = runtime_seconds,
    
    p_perm = safe_numeric(
      get_scalar(
        result_tbl,
        c("p_perm", "p_perm_kappa", "p_perm_spectral", "p_perm_shannon")
      )
    ),
    
    p_perm_shannon = safe_numeric(
      get_scalar(result_tbl, c("p_perm_shannon"))
    ),
    
    p_perm_spectral = safe_numeric(
      get_scalar(result_tbl, c("p_perm_spectral"))
    ),
    
    ci_normal_low = safe_numeric(
      get_scalar(
        result_tbl,
        c(
          "ci_normal_low",
          "kappa_boot_normal_ci_lower",
          "spectral_boot_normal_ci_lower",
          "shannon_boot_normal_ci_lower"
        )
      )
    ),
    
    ci_normal_high = safe_numeric(
      get_scalar(
        result_tbl,
        c(
          "ci_normal_high",
          "kappa_boot_normal_ci_upper",
          "spectral_boot_normal_ci_upper",
          "shannon_boot_normal_ci_upper"
        )
      )
    ),
    
    ci_tumor_low = safe_numeric(
      get_scalar(
        result_tbl,
        c(
          "ci_tumor_low",
          "kappa_boot_tumor_ci_lower",
          "spectral_boot_tumor_ci_lower",
          "shannon_boot_tumor_ci_lower"
        )
      )
    ),
    
    ci_tumor_high = safe_numeric(
      get_scalar(
        result_tbl,
        c(
          "ci_tumor_high",
          "kappa_boot_tumor_ci_upper",
          "spectral_boot_tumor_ci_upper",
          "shannon_boot_tumor_ci_upper"
        )
      )
    ),
    
    ci_normal_width = ci_normal_high - ci_normal_low,
    ci_tumor_width = ci_tumor_high - ci_tumor_low,
    
    status = "ok",
    error_message = NA_character_
  )
}

summarise_error <- function(row, runtime_seconds, error_message) {
  tibble(
    diagnostic_id = row$diagnostic_id,
    engine = row$engine,
    chip = row$chip,
    filter_regime = row$filter_regime,
    group = row$group,
    comparison = row$comparison,
    effect_size_class = row$effect_size_class,
    
    n_resamples = as.integer(row$n_resamples),
    n_features_expected = as.integer(row$n_features),
    primary_metric = row$primary_metric,
    observed_primary_delta = safe_numeric(row$primary_delta),
    abs_observed_primary_delta = safe_numeric(row$abs_primary_delta),
    
    permutation_metric = row$permutation_metric,
    bootstrap_metric = row$bootstrap_metric,
    permutation_unit = row$permutation_unit,
    bootstrap_unit = row$bootstrap_unit,
    covariance_space = row$covariance_space,
    seed = as.integer(row$seed),
    
    runtime_seconds = runtime_seconds,
    
    p_perm = NA_real_,
    p_perm_shannon = NA_real_,
    p_perm_spectral = NA_real_,
    ci_normal_low = NA_real_,
    ci_normal_high = NA_real_,
    ci_tumor_low = NA_real_,
    ci_tumor_high = NA_real_,
    ci_normal_width = NA_real_,
    ci_tumor_width = NA_real_,
    
    status = "error",
    error_message = error_message
  )
}

# ---- Execute diagnostics -------------------------------------------------------

diagnostic_results <- vector("list", nrow(diagnostic_grid))

for (i in seq_len(nrow(diagnostic_grid))) {
  row <- diagnostic_grid[i, ]
  
  message(
    sprintf(
      "[%d/%d] %s | %s | %s | %s | %s | n=%s",
      i,
      nrow(diagnostic_grid),
      row$engine,
      row$chip,
      row$filter_regime,
      row$group,
      row$comparison,
      row$n_resamples
    )
  )
  
  start_time <- Sys.time()
  
  diagnostic_results[[i]] <- tryCatch(
    {
      result_tbl <- run_single_structural_diagnostic(row)
      runtime_seconds <- elapsed_seconds(start_time)
      
      summarise_success(
        result_tbl = result_tbl,
        row = row,
        runtime_seconds = runtime_seconds
      )
    },
    error = function(e) {
      runtime_seconds <- elapsed_seconds(start_time)
      
      summarise_error(
        row = row,
        runtime_seconds = runtime_seconds,
        error_message = conditionMessage(e)
      )
    }
  )
}

diagnostic_results_tbl <- bind_rows(diagnostic_results)

# ---- Summaries ----------------------------------------------------------------

runtime_summary <- diagnostic_results_tbl |>
  group_by(engine, chip, filter_regime, n_resamples) |>
  summarise(
    n_runs = n(),
    n_ok = sum(status == "ok", na.rm = TRUE),
    n_error = sum(status == "error", na.rm = TRUE),
    median_runtime_seconds = median(runtime_seconds, na.rm = TRUE),
    q75_runtime_seconds = quantile(runtime_seconds, 0.75, na.rm = TRUE),
    total_runtime_seconds = sum(runtime_seconds, na.rm = TRUE),
    runtime_per_resample_median = median(runtime_seconds / n_resamples, na.rm = TRUE),
    .groups = "drop"
  )

convergence_summary <- diagnostic_results_tbl |>
  filter(status == "ok") |>
  group_by(engine, chip, filter_regime, effect_size_class, n_resamples) |>
  summarise(
    n_runs = n(),
    median_p_perm = median(p_perm, na.rm = TRUE),
    q25_p_perm = quantile(p_perm, 0.25, na.rm = TRUE),
    q75_p_perm = quantile(p_perm, 0.75, na.rm = TRUE),
    median_ci_normal_width = median(ci_normal_width, na.rm = TRUE),
    median_ci_tumor_width = median(ci_tumor_width, na.rm = TRUE),
    .groups = "drop"
  )

# ---- Write CSV outputs ---------------------------------------------------------

write_csv(
  diagnostic_results_tbl,
  file.path(diagnostic_dir, "structural_resampling_diagnostic_results.csv")
)

write_csv(
  runtime_summary,
  file.path(diagnostic_dir, "structural_resampling_runtime_summary.csv")
)

write_csv(
  convergence_summary,
  file.path(diagnostic_dir, "structural_resampling_convergence_summary.csv")
)

# ---- Write DuckDB outputs ------------------------------------------------------

DBI::dbWriteTable(
  con,
  "diagnostic_structural_resampling_results",
  diagnostic_results_tbl,
  overwrite = TRUE
)

DBI::dbWriteTable(
  con,
  "diagnostic_structural_resampling_runtime_summary",
  runtime_summary,
  overwrite = TRUE
)

DBI::dbWriteTable(
  con,
  "diagnostic_structural_resampling_convergence_summary",
  convergence_summary,
  overwrite = TRUE
)

DBI::dbExecute(
  con,
  "
  CREATE OR REPLACE VIEW vw_diagnostic_structural_resampling_execution_inventory AS
  SELECT
      'diagnostic_structural_resampling_results' AS table_name,
      COUNT(*) AS n_rows
  FROM diagnostic_structural_resampling_results

  UNION ALL

  SELECT
      'diagnostic_structural_resampling_runtime_summary' AS table_name,
      COUNT(*) AS n_rows
  FROM diagnostic_structural_resampling_runtime_summary

  UNION ALL

  SELECT
      'diagnostic_structural_resampling_convergence_summary' AS table_name,
      COUNT(*) AS n_rows
  FROM diagnostic_structural_resampling_convergence_summary
  "
)

message("✅ Structural resampling diagnostics completed.")

print(
  DBI::dbGetQuery(
    con,
    "SELECT * FROM vw_diagnostic_structural_resampling_execution_inventory"
  )
)

# ==============================================================================
# Notes
#
# This script executes structural comparison-level resampling diagnostics using
# the same matrix maps and filtered-probe files consumed by the structural
# inference pipeline.
#
# It reads:
#
#   - diagnostic_structural_resampling_grid
#
# from DuckDB and writes:
#
#   - diagnostic_structural_resampling_results
#   - diagnostic_structural_resampling_runtime_summary
#   - diagnostic_structural_resampling_convergence_summary
#
# plus CSV copies under:
#
#   output/global_cancer/diagnostics/structural_resampling/
#
# The diagnostic unit is:
#
#   engine × chip × filter_regime × group × comparison × n_resamples
#
# Use max_diagnostic_rows <- 16L for smoke testing and Inf for the full grid.
#
# ==============================================================================