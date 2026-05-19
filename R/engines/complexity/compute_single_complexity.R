# ------------------------------------------------------------------------------
# Compute complexity metrics for a single expression matrix
# ------------------------------------------------------------------------------

compute_matrix_complexity <- function(label,
                                      chip,
                                      mat,
                                      covariance_space = "sample") {
  
  stopifnot(is.matrix(mat))
  
  res <- main_complexity(
    tissue = label,
    mat = mat,
    covariance_space = covariance_space
  )
  
  tibble::tibble(
    Label = label,
    Chip = chip,
    Samples = res$samples,
    
    `COV kappa` = res$kappa1,
    `TwoNorm kappa` = res$kappa2,
    `SVD kappa` = res$kappa3,
    
    `Eff Rank` = res$eff_rank,
    
    Sparsity = res$sparsity,
    
    `Composite κ` = res$comp_kappa
  )
}