# ==============================================================================
# File: run_complexity_structural_engine.R
# Purpose: Run complexity analysis for structural inference
# Role: Pure wrapper function for structural inference pipeline
# ==============================================================================

#' Run complexity structural engine
#'
#' This function assumes all dependencies have already been sourced by the
#' top-level pipeline. It does not source files, load globals, or save outputs.
#'
#' @return Output returned by run_pairwise_complexity().
run_complexity_structural_engine <- function(matrix_lookup,
                                             comparison_lookup,
                                             filtered_probes_dir,
                                             chips,
                                             filter_regime_tbl,
                                             run_permutation = FALSE,
                                             n_perm = 0,
                                             permutation_metric = "all",
                                             permutation_unit = "sample_label",
                                             run_bootstrap = FALSE,
                                             n_boot = 0,
                                             bootstrap_metric = "all",
                                             bootstrap_unit = "sample",
                                             covariance_space = "sample",
                                             seed = NULL) {
  
  required_cols <- c(
    "filter_regime",
    "filter_method",
    "filter_scope",
    "filter_n",
    "variance_mode"
  )
  
  missing_cols <- setdiff(required_cols, names(filter_regime_tbl))
  
  if (length(missing_cols) > 0) {
    stop(
      "filter_regime_tbl is missing required columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  
  filter_regimes <- unique(filter_regime_tbl$filter_regime)
  
  required_probe_files <- as.vector(
    outer(
      chips,
      filter_regimes,
      FUN = function(chip, regime) {
        sprintf("filtered_probes_%s_%s.rds", chip, regime)
      }
    )
  )
  
  missing_probe_files <- required_probe_files[
    !file.exists(file.path(filtered_probes_dir, required_probe_files))
  ]
  
  if (length(missing_probe_files) > 0) {
    stop(
      sprintf(
        "Missing filtered probe files required for complexity analysis: %s",
        paste(missing_probe_files, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  
  run_pairwise_complexity(
    matrix_lookup = matrix_lookup,
    comparison_lookup = comparison_lookup,
    filtered_probes_dir = filtered_probes_dir,
    chips = chips,
    filter_regimes = filter_regimes,
    
    run_permutation = run_permutation,
    n_perm = n_perm,
    permutation_metric = permutation_metric,
    permutation_unit = permutation_unit,
    
    run_bootstrap = run_bootstrap,
    n_boot = n_boot,
    bootstrap_metric = bootstrap_metric,
    bootstrap_unit = bootstrap_unit,
    
    covariance_space = covariance_space,
    seed = seed
  )
}