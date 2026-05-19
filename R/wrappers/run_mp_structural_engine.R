# ==============================================================================
# File: run_mp_structural_engine.R
# Purpose: Run Marchenko-Pastur spectral analysis for structural inference
# Role: Pure wrapper function for structural inference pipeline
# ==============================================================================

#' Run MP structural spectral engine
#'
#' This function assumes all dependencies have already been sourced by the
#' top-level pipeline. It does not source files, load globals, or save outputs.
#'
#' @return Output returned by run_pairwise_spectral().
run_mp_structural_engine <- function(matrix_lookup,
                                     comparison_lookup,
                                     filtered_probes_dir,
                                     chips,
                                     filter_regime_tbl,
                                     spectral_method = "marchenko_pastur",
                                     min_samples_per_condition = 5,
                                     min_selected_probes = 5,
                                     use_correlation = TRUE,
                                     standardize = TRUE) {
  
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
        "Missing filtered probe files required for MP spectral analysis: %s",
        paste(missing_probe_files, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  
  run_pairwise_spectral(
    matrix_lookup = matrix_lookup,
    comparison_lookup = comparison_lookup,
    filtered_probes_dir = filtered_probes_dir,
    chips = chips,
    filter_regimes = filter_regimes,
    method = spectral_method,
    min_samples_per_condition = min_samples_per_condition,
    min_selected_probes = min_selected_probes,
    use_correlation = use_correlation,
    standardize = standardize
  )
}