# ==============================================================================
# Core entropy metric functions
# ==============================================================================
#
# Matrix orientation convention:
#   mat is expected to be probes/features x samples.
#
# For spectral entropy:
#   covariance_space = "sample" computes sample-space covariance:
#     cov(mat)
#
#   covariance_space = "probe" computes probe-space covariance:
#     cov(t(mat))
#
# The structural inference default should be covariance_space = "sample".
# ==============================================================================


# ------------------------------------------------------------------------------
# Shannon entropy
# ------------------------------------------------------------------------------

compute_shannon_entropy <- function(mat) {
  if (!is.matrix(mat) || nrow(mat) == 0 || ncol(mat) == 0) {
    return(NA_real_)
  }
  
  mat <- mat[complete.cases(mat), , drop = FALSE]
  
  if (nrow(mat) == 0 || ncol(mat) == 0) {
    return(NA_real_)
  }
  
  round(DescTools::Entropy(as.vector(mat), base = 2), 3)
}


# ------------------------------------------------------------------------------
# Spectral entropy
# ------------------------------------------------------------------------------

compute_spectral_entropy <- function(mat,
                                     covariance_space = "sample") {
  if (!covariance_space %in% c("sample", "probe")) {
    stop(
      sprintf(
        "Invalid covariance_space: %s. Valid values are: sample, probe",
        covariance_space
      ),
      call. = FALSE
    )
  }
  
  if (!is.matrix(mat) || nrow(mat) == 0 || ncol(mat) == 0) {
    return(NA_real_)
  }
  
  mat <- mat[complete.cases(mat), , drop = FALSE]
  
  if (nrow(mat) < 2 || ncol(mat) < 2) {
    return(NA_real_)
  }
  
  cov_mat <- tryCatch(
    {
      if (identical(covariance_space, "sample")) {
        # mat is probes x samples.
        # cov(mat) computes covariance among samples.
        stats::cov(mat, use = "pairwise.complete.obs")
      } else {
        # mat is probes x samples.
        # cov(t(mat)) computes covariance among probes.
        stats::cov(t(mat), use = "pairwise.complete.obs")
      }
    },
    error = function(e) NULL
  )
  
  if (is.null(cov_mat)) {
    return(NA_real_)
  }
  
  eig_vals <- tryCatch(
    eigen(cov_mat, symmetric = TRUE, only.values = TRUE)$values,
    error = function(e) NULL
  )
  
  if (is.null(eig_vals)) {
    return(NA_real_)
  }
  
  eig_vals <- pmax(Re(eig_vals), 0)
  eig_vals <- eig_vals[eig_vals > 0]
  
  if (length(eig_vals) == 0 || sum(eig_vals) <= 0) {
    return(NA_real_)
  }
  
  p <- eig_vals / sum(eig_vals)
  entropy <- -sum(p * log2(p + .Machine$double.eps))
  
  round(entropy, 3)
}


# ------------------------------------------------------------------------------
# Entropy direction labels
# ------------------------------------------------------------------------------

entropy_direction <- function(delta) {
  if (is.na(delta)) {
    return(NA_character_)
  }
  
  dplyr::case_when(
    delta < -0.5 ~ "strongly anti-chaotic",
    delta < -0.1 ~ "mildly anti-chaotic",
    delta <  0.1 ~ "neutral",
    delta <  0.5 ~ "mildly chaotic",
    TRUE         ~ "strongly chaotic"
  )
}


# ------------------------------------------------------------------------------
# Single-tissue entropy helper
# ------------------------------------------------------------------------------

main_entropy <- function(tissue,
                         mat,
                         covariance_space = "sample") {
  if (is.null(mat) || !is.matrix(mat)) {
    return(list(
      tissue = tissue,
      samples = NA_integer_,
      shannon = NA_real_,
      spectral = NA_real_,
      covariance_space = covariance_space
    ))
  }
  
  list(
    tissue = tissue,
    samples = ncol(mat),
    shannon = compute_shannon_entropy(mat),
    spectral = compute_spectral_entropy(
      mat,
      covariance_space = covariance_space
    ),
    covariance_space = covariance_space
  )
}