# ==============================================================================
# File: statistical_complexity_helpers.R
# Purpose: Statistical helper functions for transcriptomic complexity inference
# Role: Optional inferential layer for the structural complexity engine
# ==============================================================================

#' Statistical helper functions for complexity analysis
#'
#' These functions provide optional inferential support for transcriptomic
#' complexity analysis through bootstrap confidence intervals, permutation
#' testing, and related distributional utilities.
#'
#' Within the structural inference framework, the core complexity descriptors
#' are intended to operate as fast structural phenotypes. Bootstrap and
#' permutation procedures are therefore treated as optional secondary inference
#' layers rather than default pipeline behavior.
#'
#' Current default structural workflow:
#'
#'   descriptor computation only
#'
#' Optional targeted workflow:
#'
#'   descriptor + permutation/bootstrap validation
#'
#' Matrices are assumed to follow the structural inference convention:
#'
#'   rows    = probes/features
#'   columns = samples
NULL

# ---- Bootstrap confidence intervals ------------------------------------------

#' Bootstrap confidence interval for a complexity metric
#'
#' @param mat Numeric matrix with probes/features in rows and samples in columns.
#' @param complexity_fn Function used to compute the complexity statistic.
#' @param n_boot Integer; number of bootstrap replicates.
#' @param level Confidence level for interval estimation.
#' @param sample_dim Resampling dimension:
#'   "col" = sample-level resampling
#'   "row" = probe-level resampling
#' @param seed Optional integer seed for reproducibility.
#'
#' @return List containing bootstrap mean, confidence bounds,
#'   and bootstrap distribution.
bootstrap_kappa_ci <- function(mat,
                               complexity_fn = get_svd_kappa,
                               n_boot = 0,
                               level = 0.95,
                               sample_dim = "col",
                               seed = NULL) {
  
  if (n_boot <= 0) {
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
  
  stats <- replicate(n_boot, {
    
    if (sample_dim == "col") {
      mat_boot <- mat[
        ,
        sample(ncol(mat), replace = TRUE),
        drop = FALSE
      ]
    } else {
      mat_boot <- mat[
        sample(nrow(mat), replace = TRUE),
        ,
        drop = FALSE
      ]
    }
    
    complexity_fn(mat_boot)
  })
  
  ci <- quantile(
    stats,
    probs = c(
      (1 - level) / 2,
      1 - (1 - level) / 2
    ),
    na.rm = TRUE
  )
  
  list(
    mean = mean(stats, na.rm = TRUE),
    ci_lower = ci[[1]],
    ci_upper = ci[[2]],
    distribution = stats
  )
}

# ---- Permutation testing -----------------------------------------------------

#' Permutation test for complexity-score difference
#'
#' The permutation p-value is computed using a two-sided test:
#'
#'   mean(abs(null_distribution) >= abs(observed_difference))
#'
#' @param mat_normal Matrix of normal samples.
#' @param mat_cancer Matrix of cancer/tumor samples.
#' @param complexity_fn Function used to compute the complexity statistic.
#' @param n_perm Integer; number of permutation replicates.
#' @param seed Optional integer seed for reproducibility.
#'
#' @return List containing observed difference, permutation p-value,
#'   and null distribution.
permutation_test_complexity <- function(mat_normal,
                                        mat_cancer,
                                        complexity_fn = get_svd_kappa,
                                        n_perm = 0,
                                        seed = NULL) {
  
  observed_diff <- complexity_fn(mat_cancer) -
    complexity_fn(mat_normal)
  
  if (n_perm <= 0) {
    return(list(
      observed = observed_diff,
      p_value = NA_real_,
      null_distribution = numeric(0)
    ))
  }
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  all_mat <- cbind(mat_normal, mat_cancer)
  
  labels <- c(
    rep("normal", ncol(mat_normal)),
    rep("cancer", ncol(mat_cancer))
  )
  
  null_dist <- replicate(n_perm, {
    
    perm_labels <- sample(labels)
    
    mat_n <- all_mat[
      ,
      perm_labels == "normal",
      drop = FALSE
    ]
    
    mat_c <- all_mat[
      ,
      perm_labels == "cancer",
      drop = FALSE
    ]
    
    complexity_fn(mat_c) - complexity_fn(mat_n)
  })
  
  p_value <- mean(
    abs(null_dist) >= abs(observed_diff)
  )
  
  list(
    observed = observed_diff,
    p_value = p_value,
    null_distribution = null_dist
  )
}

# ---- Distribution comparison utilities ---------------------------------------

# Currently not used in the structural complexity engine.
#
# Retained for possible future workflows involving:
#
#   - sample-level complexity distributions
#   - latent-space complexity distributions
#   - cross-cancer distributional comparisons
#   - nonparametric structural-profile testing

#' Compare distributions of complexity scores
#'
#' @param norm_vals Numeric vector of normal-state scores.
#' @param cancer_vals Numeric vector of tumor-state scores.
#' @param method Statistical comparison method:
#'   "ks"     = Kolmogorov-Smirnov test
#'   "wilcox" = Wilcoxon rank-sum test
#'
#' @return P-value from the selected statistical test.
compare_kappa_distributions <- function(norm_vals,
                                        cancer_vals,
                                        method = "ks") {
  
  if (method == "ks") {
    test <- ks.test(norm_vals, cancer_vals)
  } else {
    test <- wilcox.test(norm_vals, cancer_vals)
  }
  
  test$p.value
}