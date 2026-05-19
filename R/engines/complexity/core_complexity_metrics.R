# ==============================================================================
# File: core_complexity_metrics.R
# Purpose: Define core transcriptomic complexity descriptor functions
# Role: Metric layer for the Global Cancer structural inference pipeline
# ==============================================================================

#' Core complexity metric functions
#'
#' These functions define structural descriptors used to quantify transcriptomic
#' complexity as matrix conditioning, spectral organization, effective
#' dimensionality, sparsity, and composite conditioning.
#'
#' Matrices are assumed to follow the structural inference convention:
#'
#'   rows    = probes/features
#'   columns = samples
#'
#' Therefore, covariance in sample space is computed as:
#'
#'   cov(mat)
#'
#' while covariance in probe space is computed as:
#'
#'   cov(t(mat))
#'
#' Default structural inference should use sample-space covariance for parity
#' with the entropy engine.
NULL

# ---- Internal helpers ---------------------------------------------------------

sanitize_complexity_matrix <- function(mat) {
  if (!is.matrix(mat)) {
    return(NULL)
  }
  
  mat <- mat[complete.cases(mat), , drop = FALSE]
  
  if (nrow(mat) < 2 || ncol(mat) < 2) {
    return(NULL)
  }
  
  mat
}

compute_complexity_covariance <- function(mat,
                                          covariance_space = "sample") {
  mat <- sanitize_complexity_matrix(mat)
  
  if (is.null(mat)) {
    return(NULL)
  }
  
  if (covariance_space == "sample") {
    return(cov(mat))
  }
  
  if (covariance_space == "probe") {
    return(cov(t(mat)))
  }
  
  stop(
    sprintf(
      "Invalid covariance_space: %s. Expected 'sample' or 'probe'.",
      covariance_space
    ),
    call. = FALSE
  )
}

# ---- Core metrics -------------------------------------------------------------

#' Compute covariance matrix condition number
#'
#' @param mat Numeric matrix with probes/features in rows and samples in columns.
#' @param covariance_space Covariance orientation: "sample" or "probe".
#'
#' @return Condition number of the covariance matrix.
get_cov_kappa <- function(mat,
                          covariance_space = "sample") {
  cov_matrix <- compute_complexity_covariance(
    mat = mat,
    covariance_space = covariance_space
  )
  
  if (is.null(cov_matrix)) {
    return(NA_real_)
  }
  
  round(pracma::cond(cov_matrix), 2)
}

#' Compute 2-norm condition number
#'
#' @param mat Numeric matrix.
#'
#' @return 2-norm condition number.
get_2norm_kappa <- function(mat) {
  mat <- sanitize_complexity_matrix(mat)
  
  if (is.null(mat)) {
    return(NA_real_)
  }
  
  round(kappa(mat), 2)
}

#' Compute SVD-based condition number
#'
#' @param mat Numeric matrix.
#'
#' @return Ratio of largest to smallest singular values.
get_svd_kappa <- function(mat) {
  mat <- sanitize_complexity_matrix(mat)
  
  if (is.null(mat)) {
    return(NA_real_)
  }
  
  svd_d <- svd(mat)$d
  
  if (length(svd_d) == 0) {
    return(NA_real_)
  }
  
  round(max(svd_d) / max(min(svd_d), .Machine$double.eps), 2)
}

#' Calculate effective rank from covariance eigenvalues
#'
#' @param mat Numeric matrix with probes/features in rows and samples in columns.
#' @param covariance_space Covariance orientation: "sample" or "probe".
#'
#' @return Effective rank as exp(entropy of normalized eigenvalues).
get_effective_rank <- function(mat,
                               covariance_space = "sample") {
  cov_matrix <- compute_complexity_covariance(
    mat = mat,
    covariance_space = covariance_space
  )
  
  if (is.null(cov_matrix)) {
    return(NA_real_)
  }
  
  eig_vals <- eigen(
    cov_matrix,
    symmetric = TRUE,
    only.values = TRUE
  )$values
  
  eig_vals <- eig_vals[eig_vals > 0]
  
  if (length(eig_vals) == 0 || sum(eig_vals) <= 0) {
    return(NA_real_)
  }
  
  p <- eig_vals / sum(eig_vals)
  h <- -sum(p * log(p))
  
  round(exp(h), 2)
}

#' Estimate matrix sparsity
#'
#' @param mat Numeric matrix.
#' @param threshold Threshold for considering values near zero.
#'
#' @return Proportion of near-zero elements.
get_matrix_sparsity <- function(mat,
                                threshold = 1e-5) {
  mat <- sanitize_complexity_matrix(mat)
  
  if (is.null(mat)) {
    return(NA_real_)
  }
  
  round(mean(abs(mat) < threshold), 3)
}

# NOTE:
# This function currently recomputes covariance- and SVD-based condition
# summaries internally even when individual components have already been
# computed upstream within compare_pair_complexity().
#
# This redundancy is acceptable for the current descriptor-focused structural
# workflow because runtime is now dominated by optional inferential layers
# rather than descriptor calculation itself.
#
# Future optimization:
#   refactor toward a shared eigenspectrum/SVD cache so condition-number
#   summaries and derived descriptors can reuse previously computed matrix
#   decompositions instead of recomputing them independently.
#
# This is especially relevant if complexity descriptors are later expanded
# or integrated into a unified structural phenotype engine shared across:
#
#   MP + complexity + entropy + latent geometry

#' Compute composite kappa from condition-number estimates
#'
#' @param mat Numeric matrix.
#' @param covariance_space Covariance orientation for covariance kappa.
#'
#' @return Mean of covariance, 2-norm, and SVD condition numbers.
compute_composite_kappa <- function(mat,
                                    covariance_space = "sample") {
  mat <- sanitize_complexity_matrix(mat)
  
  if (is.null(mat)) {
    return(NA_real_)
  }
  
  k1 <- get_cov_kappa(
    mat = mat,
    covariance_space = covariance_space
  )
  k2 <- get_2norm_kappa(mat)
  k3 <- get_svd_kappa(mat)
  
  round(mean(c(k1, k2, k3), na.rm = TRUE), 2)
}

#' Determine complexity direction
#'
#' @param diff Difference in complexity between tumor and normal.
#'
#' @return "gained", "lost", or NA.
complexity_gain <- function(diff) {
  if (is.na(diff)) {
    return(NA_character_)
  }
  
  if (diff > 0) {
    return("gained")
  }
  
  "lost"
}

#' Compute all core complexity descriptors for one matrix
#'
#' @param tissue Tissue or condition label.
#' @param mat Numeric matrix.
#' @param covariance_space Covariance orientation: "sample" or "probe".
#'
#' @return List of complexity metrics.
main_complexity <- function(tissue,
                            mat,
                            covariance_space = "sample") {
  mat_checked <- sanitize_complexity_matrix(mat)
  
  if (is.null(mat_checked)) {
    return(list(
      tissue = tissue,
      samples = NA_integer_,
      kappa1 = NA_real_,
      kappa2 = NA_real_,
      kappa3 = NA_real_,
      eff_rank = NA_real_,
      sparsity = NA_real_,
      comp_kappa = NA_real_
    ))
  }
  
  list(
    tissue = tissue,
    samples = ncol(mat_checked),
    kappa1 = get_cov_kappa(
      mat = mat_checked,
      covariance_space = covariance_space
    ),
    kappa2 = get_2norm_kappa(mat_checked),
    kappa3 = get_svd_kappa(mat_checked),
    eff_rank = get_effective_rank(
      mat = mat_checked,
      covariance_space = covariance_space
    ),
    sparsity = get_matrix_sparsity(mat_checked),
    comp_kappa = compute_composite_kappa(
      mat = mat_checked,
      covariance_space = covariance_space
    )
  )
}