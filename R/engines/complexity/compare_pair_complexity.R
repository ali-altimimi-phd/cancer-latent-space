# ==============================================================================
# File: compare_pair_complexity.R
# Purpose: Compare paired normal/tumor complexity summaries
# Role: Pair-level complexity descriptor and optional inference computation
# ==============================================================================

#' Compare paired normal/tumor complexity summaries
#'
#' This function computes core complexity descriptors for one normal/tumor
#' comparison. Permutation and bootstrap layers are optional and are only run
#' when explicitly enabled by the calling engine.
#'
#' Required functions in environment:
#'   - get_group_matrices()
#'   - get_svd_kappa()
#'   - complexity_gain()
#'   - permutation_test_complexity()
#'   - bootstrap_kappa_ci()
#'   - get_effective_rank()
#'   - get_matrix_sparsity()
#'   - compute_composite_kappa()
#'
#' @param matrices_i Named list of condition-specific expression matrices.
#' @param comparison_labels Character vector of length two: normal label, tumor label.
#' @param selected_probes Character vector of selected probe IDs.
#' @param comparison Comparison code.
#' @param group Cancer group.
#' @param chip Chip/platform identifier.
#' @param filter_regime Probe-selection regime label.
#' @param gene_set_mode Gene-set mode or analysis context. Defaults to "FULL"
#'   for whole filtered-probe comparisons.
#' @param gene_set_name Gene-set identifier or name. Defaults to "FULL" for
#'   whole filtered-probe comparisons.
#' @param complexity_fn Function used as the primary legacy complexity statistic.
#' @param run_permutation Logical; whether to run permutation testing.
#' @param n_perm Integer; number of permutation replicates.
#' @param permutation_metric Metric family used for permutation testing.
#' @param permutation_unit Unit for permutation, usually "sample_label".
#' @param run_bootstrap Logical; whether to run bootstrap confidence intervals.
#' @param n_boot Integer; number of bootstrap replicates.
#' @param bootstrap_metric Metric family used for bootstrap inference.
#' @param bootstrap_unit Unit for bootstrap, usually "sample".
#' @param covariance_space Covariance orientation for eigenspectrum descriptors.
#'   Currently retained for API parity with entropy.
#' @param seed Optional integer seed for resampling routines.
#'
#' @return Tibble with one paired complexity summary row.
compare_pair_complexity <- function(matrices_i,
                                    comparison_labels,
                                    selected_probes,
                                    comparison,
                                    group,
                                    chip,
                                    filter_regime,
                                    gene_set_mode = "FULL",
                                    gene_set_name = "FULL",
                                    complexity_fn = get_svd_kappa,
                                    run_permutation = FALSE,
                                    n_perm = 0,
                                    permutation_metric = "all",
                                    permutation_unit = "sample_label",
                                    run_bootstrap = FALSE,
                                    n_boot = 0,
                                    bootstrap_metric = "all",
                                    bootstrap_unit = "sample",
                                    covariance_space = "sample",
                                    seed = NULL) {
  
  # Currently retained for API parity with entropy.
  # Metric/unit-specific dispatch can be added after descriptor fast-path
  # validation if complexity inference is expanded beyond kappa.
  invisible(permutation_metric)
  invisible(permutation_unit)
  invisible(bootstrap_metric)
  invisible(bootstrap_unit)
  invisible(covariance_space)
  
  group_mats <- get_group_matrices(
    matrices_i = matrices_i,
    comparison_labels = comparison_labels,
    selected_probes = selected_probes
  )
  
  mat_normal <- group_mats$normal
  mat_tumor  <- group_mats$tumor
  
  kappa_normal <- complexity_fn(mat_normal)
  kappa_tumor  <- complexity_fn(mat_tumor)
  kappa_delta  <- round(kappa_tumor - kappa_normal, 3)
  direction    <- complexity_gain(kappa_delta)
  
  # ---- Optional inferential layers -------------------------------------------
  
  seed_perm <- if (!is.null(seed)) seed + 1 else NULL
  seed_boot_normal <- if (!is.null(seed)) seed + 2 else NULL
  seed_boot_tumor <- if (!is.null(seed)) seed + 3 else NULL
  
  perm <- NULL
  if (isTRUE(run_permutation) && n_perm > 0) {
    perm <- permutation_test_complexity(
      mat_normal = mat_normal,
      mat_cancer = mat_tumor,
      complexity_fn = complexity_fn,
      n_perm = n_perm,
      seed = seed_perm
    )
  }
  
  boot_normal <- NULL
  boot_tumor <- NULL
  if (isTRUE(run_bootstrap) && n_boot > 0) {
    boot_normal <- bootstrap_kappa_ci(
      mat = mat_normal,
      complexity_fn = complexity_fn,
      n_boot = n_boot,
      seed = seed_boot_normal
    )
    
    boot_tumor <- bootstrap_kappa_ci(
      mat = mat_tumor,
      complexity_fn = complexity_fn,
      n_boot = n_boot,
      seed = seed_boot_tumor
    )
  }
  
  # ---- Descriptor layer -------------------------------------------------------
  
  effrank_normal <- get_effective_rank(mat_normal, covariance_space = covariance_space)
  effrank_tumor  <- get_effective_rank(mat_tumor, covariance_space = covariance_space)
  
  sparsity_normal <- get_matrix_sparsity(mat_normal)
  sparsity_tumor  <- get_matrix_sparsity(mat_tumor)
  
  composite_kappa_normal <- compute_composite_kappa(mat_normal, covariance_space = covariance_space)
  composite_kappa_tumor  <- compute_composite_kappa(mat_tumor, covariance_space = covariance_space)
  
  tibble::tibble(
    chip = chip,
    group = group,
    comparison = comparison,
    filter_regime = filter_regime,
    engine = "complexity",
    gene_set_mode = gene_set_mode,
    gene_set_name = gene_set_name,
    
    matrix_key_normal = group_mats$normal_key,
    matrix_key_tumor = group_mats$tumor_key,
    n_shared_probes = group_mats$n_shared_probes,
    
    n_samples_normal = ncol(mat_normal),
    n_samples_tumor = ncol(mat_tumor),
    n_features = nrow(mat_normal),
    
    kappa_normal = kappa_normal,
    kappa_tumor = kappa_tumor,
    kappa_delta = kappa_delta,
    direction = direction,
    p_perm = if (!is.null(perm)) perm$p_value else NA_real_,
    
    ci_normal_low = if (!is.null(boot_normal)) boot_normal$ci_lower else NA_real_,
    ci_normal_high = if (!is.null(boot_normal)) boot_normal$ci_upper else NA_real_,
    ci_tumor_low = if (!is.null(boot_tumor)) boot_tumor$ci_lower else NA_real_,
    ci_tumor_high = if (!is.null(boot_tumor)) boot_tumor$ci_upper else NA_real_,
    
    effrank_normal = effrank_normal,
    effrank_tumor = effrank_tumor,
    effrank_delta = effrank_tumor - effrank_normal,
    
    sparsity_normal = sparsity_normal,
    sparsity_tumor = sparsity_tumor,
    sparsity_delta = sparsity_tumor - sparsity_normal,
    
    composite_kappa_normal = composite_kappa_normal,
    composite_kappa_tumor = composite_kappa_tumor,
    composite_kappa_delta = composite_kappa_tumor - composite_kappa_normal,
    
    status = "ok"
  )
}