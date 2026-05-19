# ==============================================================================
# Statistical spectral helpers
# ==============================================================================
#
# Purpose:
#   General preprocessing and matrix-construction helpers used by spectral
#   analyses.
#
# ==============================================================================

#' Prepare an expression matrix for spectral analysis
#'
#' @param expr_matrix Numeric expression matrix, expected probes x samples.
#' @param standardize Logical; if TRUE, standardize each probe across samples.
#' @return Numeric matrix after row filtering and optional standardization.
prepare_expression_for_spectrum <- function(expr_matrix, standardize = TRUE) {
  x <- as.matrix(expr_matrix)
  x <- x[complete.cases(x), , drop = FALSE]
  
  if (nrow(x) == 0L || ncol(x) == 0L) {
    stop(
      "Expression matrix is empty after complete-case filtering.",
      call. = FALSE
    )
  }
  
  if (standardize) {
    x <- t(scale(t(x)))
    x[!is.finite(x)] <- 0
  }
  
  x
}

#' Compute a covariance or correlation matrix for spectral analysis
#'
#' @param x Numeric expression matrix, probes x samples.
#' @param use_correlation Logical; if TRUE use correlation, otherwise covariance.
#' @return Numeric symmetric matrix.
compute_spectral_matrix <- function(x, use_correlation = TRUE) {
  if (use_correlation) {
    spectral_matrix <- stats::cor(t(x), use = "pairwise.complete.obs")
  } else {
    spectral_matrix <- stats::cov(t(x), use = "pairwise.complete.obs")
  }

  spectral_matrix[!is.finite(spectral_matrix)] <- 0
  spectral_matrix
}
