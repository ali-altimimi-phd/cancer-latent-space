# ==============================================================================
# Compute single spectral summary
# ==============================================================================
#
# Purpose:
#   Thin single-condition wrapper for spectral analyses. Currently delegates to
#   Marchenko-Pastur summary computation, but this file is intentionally named
#   generically to match the engine architecture and allow later methods.
#
# Dependencies:
#   R/engines/spectral/marchenko_pastur_helpers.R
#
# ==============================================================================

#' Compute a single-condition spectral summary
#'
#' @inheritParams compute_mp_summary
#' @param method Spectral method. Currently only "marchenko_pastur" is supported.
#' @param min_samples_per_condition Minimum samples required in this condition
#'   before spectral computation.
#'
#' @return Tibble of spectral metrics.
compute_single_spectral <- function(expr_matrix,
                                    comparison,
                                    group,
                                    chip,
                                    filter_regime,
                                    condition_label,
                                    matrix_key,
                                    method = "marchenko_pastur",
                                    min_samples_per_condition = 5,
                                    use_correlation = TRUE,
                                    standardize = TRUE) {
  
  if (!identical(method, "marchenko_pastur")) {
    stop(sprintf("Unsupported spectral method: %s", method), call. = FALSE)
  }
  
  compute_mp_summary(
    expr_matrix = expr_matrix,
    comparison = comparison,
    group = group,
    chip = chip,
    filter_regime = filter_regime,
    condition_label = condition_label,
    matrix_key = matrix_key,
    min_samples_per_condition = min_samples_per_condition,
    use_correlation = use_correlation,
    standardize = standardize
  )
}