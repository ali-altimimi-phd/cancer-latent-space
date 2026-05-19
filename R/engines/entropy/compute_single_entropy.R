# ==============================================================================
# Compute entropy metrics for individual matrices
# ==============================================================================

compute_matrix_entropy <- function(label,
                                   chip,
                                   mat,
                                   covariance_space = "sample") {
  stopifnot(is.matrix(mat))
  
  res <- main_entropy(
    tissue = label,
    mat = mat,
    covariance_space = covariance_space
  )
  
  tibble::tibble(
    label = label,
    chip = chip,
    samples = res$samples,
    shannon = res$shannon,
    spectral = res$spectral,
    covariance_space = res$covariance_space
  )
}