# ==============================================================================
# Statistical helper functions for entropy analysis
# ==============================================================================
#
# These helpers implement optional inferential layers for the entropy engine.
#
# Matrix orientation convention:
#   mat is expected to be probes/features x samples.
#
# Bootstrap:
#   sample_dim = "col" resamples samples.
#   sample_dim = "row" resamples probes/features.
#
# Permutation:
#   currently supports sample-label permutation by column exchange.
# ==============================================================================


# ------------------------------------------------------------------------------
# Bootstrap entropy confidence interval
# ------------------------------------------------------------------------------

bootstrap_entropy_ci <- function(mat,
                                 entropy_fn = compute_shannon_entropy,
                                 n_boot = 0,
                                 level = 0.95,
                                 sample_dim = "col",
                                 seed = NULL) {
  if (!sample_dim %in% c("col", "row")) {
    stop(
      sprintf(
        "Invalid sample_dim: %s. Valid values are: col, row",
        sample_dim
      ),
      call. = FALSE
    )
  }
  
  if (!is.numeric(n_boot) || length(n_boot) != 1 || n_boot < 0) {
    stop("n_boot must be a single non-negative numeric value.", call. = FALSE)
  }
  
  n_boot <- as.integer(n_boot)
  
  if (n_boot == 0L) {
    return(list(
      mean = NA_real_,
      ci_lower = NA_real_,
      ci_upper = NA_real_,
      distribution = numeric(0)
    ))
  }
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  if (!is.matrix(mat) || nrow(mat) == 0 || ncol(mat) == 0) {
    return(list(
      mean = NA_real_,
      ci_lower = NA_real_,
      ci_upper = NA_real_,
      distribution = numeric(0)
    ))
  }
  
  stats <- replicate(n_boot, {
    if (identical(sample_dim, "col")) {
      mat_boot <- mat[, sample.int(ncol(mat), replace = TRUE), drop = FALSE]
    } else {
      mat_boot <- mat[sample.int(nrow(mat), replace = TRUE), , drop = FALSE]
    }
    
    entropy_fn(mat_boot)
  })
  
  ci <- stats::quantile(
    stats,
    probs = c((1 - level) / 2, 1 - (1 - level) / 2),
    na.rm = TRUE
  )
  
  list(
    mean = mean(stats, na.rm = TRUE),
    ci_lower = unname(ci[[1]]),
    ci_upper = unname(ci[[2]]),
    distribution = stats
  )
}


# ------------------------------------------------------------------------------
# Permutation test for paired normal/tumor entropy difference
# ------------------------------------------------------------------------------

permutation_test_entropy <- function(mat_normal,
                                     mat_cancer,
                                     entropy_fn = compute_shannon_entropy,
                                     n_perm = 0,
                                     seed = NULL) {
  if (!is.numeric(n_perm) || length(n_perm) != 1 || n_perm < 0) {
    stop("n_perm must be a single non-negative numeric value.", call. = FALSE)
  }
  
  n_perm <- as.integer(n_perm)
  
  if (n_perm == 0L) {
    return(list(
      observed = NA_real_,
      p_value = NA_real_,
      null_distribution = numeric(0)
    ))
  }
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  if (!is.matrix(mat_normal) || !is.matrix(mat_cancer)) {
    return(list(
      observed = NA_real_,
      p_value = NA_real_,
      null_distribution = numeric(0)
    ))
  }
  
  if (nrow(mat_normal) == 0 || nrow(mat_cancer) == 0 ||
      ncol(mat_normal) == 0 || ncol(mat_cancer) == 0) {
    return(list(
      observed = NA_real_,
      p_value = NA_real_,
      null_distribution = numeric(0)
    ))
  }
  
  if (nrow(mat_normal) != nrow(mat_cancer)) {
    stop(
      sprintf(
        "Permutation requires equal feature counts: normal=%d, cancer=%d",
        nrow(mat_normal),
        nrow(mat_cancer)
      ),
      call. = FALSE
    )
  }
  
  all_mat <- cbind(mat_normal, mat_cancer)
  
  labels <- c(
    rep("normal", ncol(mat_normal)),
    rep("cancer", ncol(mat_cancer))
  )
  
  observed_diff <- entropy_fn(mat_cancer) - entropy_fn(mat_normal)
  
  null_dist <- replicate(n_perm, {
    perm_labels <- sample(labels)
    
    mat_n <- all_mat[, perm_labels == "normal", drop = FALSE]
    mat_c <- all_mat[, perm_labels == "cancer", drop = FALSE]
    
    entropy_fn(mat_c) - entropy_fn(mat_n)
  })
  
  p_value <- mean(abs(null_dist) >= abs(observed_diff), na.rm = TRUE)
  
  list(
    observed = observed_diff,
    p_value = p_value,
    null_distribution = null_dist
  )
}


# ------------------------------------------------------------------------------
# Backward-compatible alias
# ------------------------------------------------------------------------------

# Prefer permutation_test_entropy() in all new structural code.
perm_test_entropy <- function(mat_1,
                              mat_2,
                              entropy_fn = compute_shannon_entropy,
                              n_perm = 0,
                              seed = NULL) {
  permutation_test_entropy(
    mat_normal = mat_1,
    mat_cancer = mat_2,
    entropy_fn = entropy_fn,
    n_perm = n_perm,
    seed = seed
  )
}