# ==============================================================================
# Core spectral metrics
# ==============================================================================
#
# Purpose:
#   General eigenvalue-spectrum utilities used by spectral analyses.
#   These functions are intentionally independent of Marchenko-Pastur logic and
#   can be reused for other spectral organization analyses.
#
# ==============================================================================

#' Safely compute eigenvalues for a symmetric matrix
#'
#' @param x Numeric symmetric matrix.
#' @return Numeric vector of eigenvalues.
safe_eigenvalues <- function(x) {
  vals <- eigen(x, symmetric = TRUE, only.values = TRUE)$values
  vals <- as.numeric(vals)
  vals[vals < 0 & vals > -1e-8] <- 0
  vals
}

#' Compute spectral entropy from eigenvalues
#'
#' @param vals Numeric vector of eigenvalues.
#' @return Spectral entropy, or NA_real_ if undefined.
spectral_entropy <- function(vals) {
  vals <- vals[is.finite(vals) & vals > 0]

  if (length(vals) == 0) {
    return(NA_real_)
  }

  p <- vals / sum(vals)
  -sum(p * log(p))
}

#' Compute participation ratio from eigenvalues
#'
#' @param vals Numeric vector of eigenvalues.
#' @return Participation ratio, or NA_real_ if undefined.
participation_ratio <- function(vals) {
  vals <- vals[is.finite(vals) & vals > 0]

  if (length(vals) == 0) {
    return(NA_real_)
  }

  sum(vals)^2 / sum(vals^2)
}

#' Compute leading eigenvalue fraction
#'
#' @param vals Numeric vector of eigenvalues.
#' @return Fraction of total spectral mass in the largest eigenvalue.
leading_eigenvalue_fraction <- function(vals) {
  vals <- vals[is.finite(vals) & vals > 0]

  if (length(vals) == 0 || sum(vals) == 0) {
    return(NA_real_)
  }

  max(vals) / sum(vals)
}
