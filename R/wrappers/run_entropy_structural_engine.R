# ==============================================================================
# File: run_entropy_structural_engine.R
# Purpose: Run entropy analysis for structural inference
# Role: Pure wrapper function for structural inference pipeline
# ==============================================================================

#' Run entropy structural engine
#'
#' This function assumes all dependencies have already been sourced by the
#' top-level pipeline. It does not source files, load globals, or save outputs.
#'
#' @return Output returned by run_pairwise_entropy().
run_entropy_structural_engine <- function(matrix_lookup,
                                          comparison_lookup,
                                          filtered_probes_dir,
                                          chips,
                                          filter_regime_tbl,
                                          run_permutation = FALSE,
                                          n_perm = 0,
                                          run_bootstrap = FALSE,
                                          n_boot = 0,
                                          permutation_metric = "shannon",
                                          bootstrap_metric = "shannon",
                                          covariance_space = "sample",
                                          permutation_unit = "sample_label",
                                          bootstrap_unit = "sample",
                                          seed = NULL) {
  
  stopifnot(is.logical(run_permutation), length(run_permutation) == 1)
  stopifnot(is.logical(run_bootstrap), length(run_bootstrap) == 1)
  
  if (!covariance_space %in% c("sample", "probe")) {
    stop(
      sprintf(
        "Invalid covariance_space: %s. Valid values are: sample, probe",
        covariance_space
      ),
      call. = FALSE
    )
  }
  
  if (!permutation_metric %in% c("shannon", "spectral")) {
    stop(
      sprintf(
        "Invalid permutation_metric: %s. Valid values are: shannon, spectral",
        permutation_metric
      ),
      call. = FALSE
    )
  }
  
  if (!bootstrap_metric %in% c("shannon", "spectral")) {
    stop(
      sprintf(
        "Invalid bootstrap_metric: %s. Valid values are: shannon, spectral",
        bootstrap_metric
      ),
      call. = FALSE
    )
  }
  
  if (!permutation_unit %in% c("sample_label")) {
    stop(
      sprintf(
        "Invalid permutation_unit: %s. Currently supported value: sample_label",
        permutation_unit
      ),
      call. = FALSE
    )
  }
  
  if (!bootstrap_unit %in% c("sample", "probe")) {
    stop(
      sprintf(
        "Invalid bootstrap_unit: %s. Valid values are: sample, probe",
        bootstrap_unit
      ),
      call. = FALSE
    )
  }
  
  if (!is.numeric(n_perm) || length(n_perm) != 1 || n_perm < 0) {
    stop("n_perm must be a single non-negative numeric value.", call. = FALSE)
  }
  
  if (!is.numeric(n_boot) || length(n_boot) != 1 || n_boot < 0) {
    stop("n_boot must be a single non-negative numeric value.", call. = FALSE)
  }
  
  n_perm <- as.integer(n_perm)
  n_boot <- as.integer(n_boot)
  
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
        "Missing filtered probe files required for entropy analysis: %s",
        paste(missing_probe_files, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  
  run_pairwise_entropy(
    matrix_lookup = matrix_lookup,
    comparison_lookup = comparison_lookup,
    filtered_probes_dir = filtered_probes_dir,
    chips = chips,
    filter_regimes = filter_regimes,
    
    run_permutation = run_permutation,
    n_perm = n_perm,
    
    run_bootstrap = run_bootstrap,
    n_boot = n_boot,
    
    permutation_metric = permutation_metric,
    bootstrap_metric = bootstrap_metric,
    covariance_space = covariance_space,
    
    permutation_unit = permutation_unit,
    bootstrap_unit = bootstrap_unit,
    
    seed = seed
  )
}