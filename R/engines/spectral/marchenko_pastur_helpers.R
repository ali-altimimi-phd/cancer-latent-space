# ==============================================================================
# Marchenko-Pastur helpers
# ==============================================================================
#
# Purpose:
#   Marchenko-Pastur-specific utilities for random-matrix spectral comparison.
#
# Dependencies:
#   R/engines/spectral/core_spectral_metrics.R
#   R/engines/spectral/statistical_spectral_helpers.R
#
# ==============================================================================

#' Compute Marchenko-Pastur bulk edges
#'
#' @param n_samples Number of samples.
#' @param n_features Number of features/probes.
#' @param sigma2 Noise variance parameter. Defaults to 1, which is correct for a
#'   standardized correlation matrix. For covariance matrices, pass the estimated
#'   noise variance explicitly (e.g., median eigenvalue of the empirical spectrum).
#' @return List with q, lambda_minus, and lambda_plus.
mp_edges <- function(n_samples, n_features, sigma2 = 1) {
  q <- n_features / n_samples

  list(
    q = q,
    lambda_minus = sigma2 * (1 - sqrt(q))^2,
    lambda_plus  = sigma2 * (1 + sqrt(q))^2
  )
}

#' Compute Marchenko-Pastur summary for one condition matrix
#'
#' @param expr_matrix Numeric expression matrix, expected probes x samples.
#' @param comparison Comparison label/code.
#' @param group Cancer group, e.g. carcinomas.
#' @param chip Chip/platform identifier.
#' @param filter_regime Filtering method label.
#' @param condition_label Condition label, usually normal or tumor.
#' @param matrix_key Name of source expression matrix.
#' @param min_samples_per_condition Minimum samples required in this condition
#'   before MP/spectral computation.
#' @param use_correlation Logical; if TRUE use correlation, otherwise covariance.
#' @param standardize Logical; if TRUE standardize each probe across samples.
#' @return Tibble with MP and spectral summary metrics.
compute_mp_summary <- function(expr_matrix,
                               comparison,
                               group,
                               chip,
                               filter_regime,
                               condition_label,
                               matrix_key,
                               min_samples_per_condition = 5,
                               use_correlation = TRUE,
                               standardize = TRUE) {
  
  x <- prepare_expression_for_spectrum(
    expr_matrix = expr_matrix,
    standardize = standardize
  )
  
  n_features <- nrow(x)
  n_samples  <- ncol(x)
  
  if (n_features < 2) {
    stop(
      paste(
        "Internal error: compute_mp_summary() received fewer than two",
        "features after upstream selected-probe admissibility validation."
        ),
      call. = FALSE
    )
  }
  
  if (n_samples < min_samples_per_condition) {
    return(tibble::tibble(
      chip = chip,
      group = group,
      comparison = comparison,
      filter_regime = filter_regime,
      condition = condition_label,
      matrix_key = matrix_key,
      n_samples = n_samples,
      n_features = n_features,
      q = NA_real_,
      mp_lambda_minus = NA_real_,
      mp_lambda_plus = NA_real_,
      largest_eigenvalue = NA_real_,
      largest_eigenvalue_fraction = NA_real_,
      n_spikes = NA_integer_,
      spike_fraction = NA_real_,
      excess_spectral_mass = NA_real_,
      spectral_entropy = NA_real_,
      participation_ratio = NA_real_,
      status = "skipped_insufficient_samples"
    ))
  }

  spectral_matrix <- compute_spectral_matrix(
    x = x,
    use_correlation = use_correlation
  )

  vals <- safe_eigenvalues(spectral_matrix)
  vals <- sort(vals, decreasing = TRUE)

  # sigma2 = 1 is valid for correlation matrices (each variable has unit variance).
  # For covariance matrices (use_correlation = FALSE), the caller should supply an
  # empirical sigma2 estimate; using 1 will mis-place the MP bulk edge.
  if (!use_correlation) {
    warning(
      "MP bulk edges computed with sigma2 = 1, which is only exact for correlation ",
      "matrices. For covariance matrices, supply an empirical sigma2.",
      call. = FALSE
    )
  }

  edges <- mp_edges(
    n_samples = n_samples,
    n_features = n_features,
    sigma2 = 1
  )

  lambda_plus <- edges$lambda_plus
  spike_vals <- vals[vals > lambda_plus]

  tibble::tibble(
    chip = chip,
    group = group,
    comparison = comparison,
    filter_regime = filter_regime,
    condition = condition_label,
    matrix_key = matrix_key,
    n_samples = n_samples,
    n_features = n_features,
    q = edges$q,
    mp_lambda_minus = edges$lambda_minus,
    mp_lambda_plus = edges$lambda_plus,
    largest_eigenvalue = vals[1],
    largest_eigenvalue_fraction = vals[1] / sum(vals),
    n_spikes = length(spike_vals),
    spike_fraction = length(spike_vals) / length(vals),
    excess_spectral_mass = sum(spike_vals - lambda_plus),
    spectral_entropy = spectral_entropy(vals),
    participation_ratio = participation_ratio(vals),
    status = "ok"
  )
}
