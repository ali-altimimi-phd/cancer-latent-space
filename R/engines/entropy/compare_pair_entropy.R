# ==============================================================================
# Compare paired normal/tumor entropy summaries
# ==============================================================================
#
# Purpose:
#   Compute paired entropy summaries for one normal/tumor comparison within
#   the structural inference framework.
#
#   Observed entropy descriptors are always computed.
#   Permutation and bootstrap inference are optional and controlled by config.
#
# ==============================================================================

#' Compare paired normal/tumor entropy summaries
#'
#' This function computes core entropy descriptors for one normal/tumor
#' comparison. Permutation and bootstrap layers are optional and are only run
#' when explicitly enabled by the calling engine.
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
#' @param entropy_fn Function used as the primary entropy statistic.
#' @param run_permutation Logical; whether to run entropy permutation testing.
#' @param n_perm Integer; number of permutation replicates.
#' @param permutation_metric Entropy metric used for permutation testing.
#' @param permutation_unit Unit used for permutation testing, usually
#'   "sample_label".
#' @param run_bootstrap Logical; whether to run bootstrap confidence intervals.
#' @param n_boot Integer; number of bootstrap replicates.
#' @param bootstrap_metric Entropy metric used for bootstrap inference.
#' @param bootstrap_unit Unit used for bootstrap resampling, usually "sample".
#' @param covariance_space Covariance orientation used for spectral entropy
#'   descriptors. Typically "sample" or "probe".
#' @param seed Optional random seed for resampling procedures.
#'
#' @return Tibble with one paired entropy summary row.
compare_pair_entropy <- function(matrices_i,
                                 comparison_labels,
                                 selected_probes,
                                 comparison,
                                 group,
                                 chip,
                                 filter_regime,
                                 gene_set_mode = "FULL",
                                 gene_set_name = "FULL",
                                 entropy_fn = compute_shannon_entropy,
                                 run_permutation = FALSE,
                                 n_perm = 0,
                                 permutation_metric = "shannon",
                                 permutation_unit = "sample_label",
                                 run_bootstrap = FALSE,
                                 n_boot = 0,
                                 bootstrap_metric = "shannon",
                                 bootstrap_unit = "sample",
                                 covariance_space = "sample",
                                 seed = NULL) {
  
  if (!permutation_metric %in% c("shannon", "spectral")) {
    stop(
      sprintf(
        "Invalid permutation_metric: %s. Valid values are: shannon, spectral",
        permutation_metric
      ),
      call. = FALSE
    )
  }
  
  if (!bootstrap_metric %in% c("shannon", "spectral")) {
    stop(
      sprintf(
        "Invalid bootstrap_metric: %s. Valid values are: shannon, spectral",
        bootstrap_metric
      ),
      call. = FALSE
    )
  }
  
  if (!covariance_space %in% c("sample", "probe")) {
    stop(
      sprintf(
        "Invalid covariance_space: %s. Valid values are: sample, probe",
        covariance_space
      ),
      call. = FALSE
    )
  }
  
  if (!permutation_unit %in% c("sample_label")) {
    stop(
      sprintf(
        "Invalid permutation_unit: %s. Currently supported value: sample_label",
        permutation_unit
      ),
      call. = FALSE
    )
  }
  
  if (!bootstrap_unit %in% c("sample", "probe")) {
    stop(
      sprintf(
        "Invalid bootstrap_unit: %s. Valid values are: sample, probe",
        bootstrap_unit
      ),
      call. = FALSE
    )
  }
  
  group_mats <- get_group_matrices(
    matrices_i = matrices_i,
    comparison_labels = comparison_labels,
    selected_probes = selected_probes
  )
  
  mat_normal <- group_mats$normal
  mat_tumor  <- group_mats$tumor
  
  # ---- Shannon entropy --------------------------------------------------------
  
  shannon_normal <- compute_shannon_entropy(mat_normal)
  shannon_tumor  <- compute_shannon_entropy(mat_tumor)
  
  shannon_delta <- round(
    shannon_tumor - shannon_normal,
    3
  )
  
  shannon_direction <- entropy_direction(shannon_delta)
  
  # ---- Spectral entropy -------------------------------------------------------
  
  spectral_normal <- compute_spectral_entropy(
    mat_normal,
    covariance_space = covariance_space
  )
  
  spectral_tumor <- compute_spectral_entropy(
    mat_tumor,
    covariance_space = covariance_space
  )
  
  spectral_delta <- round(
    spectral_tumor - spectral_normal,
    3
  )
  
  spectral_direction <- entropy_direction(spectral_delta)
  
  # ---- Optional permutation testing ------------------------------------------
  
  seed_perm <- if (!is.null(seed)) seed + 1 else NULL
  
  p_perm_shannon <- NA_real_
  p_perm_spectral <- NA_real_
  
  if (isTRUE(run_permutation) && n_perm > 0) {
    
    if (identical(permutation_metric, "shannon")) {
      
      perm <- permutation_test_entropy(
        mat_normal = mat_normal,
        mat_cancer = mat_tumor,
        entropy_fn = compute_shannon_entropy,
        n_perm = n_perm,
        seed = seed_perm
      )
      
      p_perm_shannon <- perm$p_value
      
    } else if (identical(permutation_metric, "spectral")) {
      
      perm <- permutation_test_entropy(
        mat_normal = mat_normal,
        mat_cancer = mat_tumor,
        entropy_fn = function(x) {
          compute_spectral_entropy(
            x,
            covariance_space = covariance_space
          )
        },
        n_perm = n_perm,
        seed = seed_perm
      )
      
      p_perm_spectral <- perm$p_value
    }
  }
  
  # ---- Optional bootstrap intervals ------------------------------------------
  
  seed_boot_normal <- if (!is.null(seed)) seed + 101 else NULL
  seed_boot_tumor  <- if (!is.null(seed)) seed + 102 else NULL
  
  shannon_boot_normal_mean <- NA_real_
  shannon_boot_normal_ci_lower <- NA_real_
  shannon_boot_normal_ci_upper <- NA_real_
  
  shannon_boot_tumor_mean <- NA_real_
  shannon_boot_tumor_ci_lower <- NA_real_
  shannon_boot_tumor_ci_upper <- NA_real_
  
  spectral_boot_normal_mean <- NA_real_
  spectral_boot_normal_ci_lower <- NA_real_
  spectral_boot_normal_ci_upper <- NA_real_
  
  spectral_boot_tumor_mean <- NA_real_
  spectral_boot_tumor_ci_lower <- NA_real_
  spectral_boot_tumor_ci_upper <- NA_real_
  
  if (isTRUE(run_bootstrap) && n_boot > 0) {
    
    sample_dim <- if (identical(bootstrap_unit, "sample")) {
      "col"
    } else {
      "row"
    }
    
    if (identical(bootstrap_metric, "shannon")) {
      
      boot_normal <- bootstrap_entropy_ci(
        mat = mat_normal,
        entropy_fn = compute_shannon_entropy,
        n_boot = n_boot,
        sample_dim = sample_dim,
        seed = seed_boot_normal
      )
      
      boot_tumor <- bootstrap_entropy_ci(
        mat = mat_tumor,
        entropy_fn = compute_shannon_entropy,
        n_boot = n_boot,
        sample_dim = sample_dim,
        seed = seed_boot_tumor
      )
      
      shannon_boot_normal_mean <- boot_normal$mean
      shannon_boot_normal_ci_lower <- boot_normal$ci_lower
      shannon_boot_normal_ci_upper <- boot_normal$ci_upper
      
      shannon_boot_tumor_mean <- boot_tumor$mean
      shannon_boot_tumor_ci_lower <- boot_tumor$ci_lower
      shannon_boot_tumor_ci_upper <- boot_tumor$ci_upper
      
    } else if (identical(bootstrap_metric, "spectral")) {
      
      spectral_entropy_fn <- function(x) {
        compute_spectral_entropy(
          x,
          covariance_space = covariance_space
        )
      }
      
      boot_normal <- bootstrap_entropy_ci(
        mat = mat_normal,
        entropy_fn = spectral_entropy_fn,
        n_boot = n_boot,
        sample_dim = sample_dim,
        seed = seed_boot_normal
      )
      
      boot_tumor <- bootstrap_entropy_ci(
        mat = mat_tumor,
        entropy_fn = spectral_entropy_fn,
        n_boot = n_boot,
        sample_dim = sample_dim,
        seed = seed_boot_tumor
      )
      
      spectral_boot_normal_mean <- boot_normal$mean
      spectral_boot_normal_ci_lower <- boot_normal$ci_lower
      spectral_boot_normal_ci_upper <- boot_normal$ci_upper
      
      spectral_boot_tumor_mean <- boot_tumor$mean
      spectral_boot_tumor_ci_lower <- boot_tumor$ci_lower
      spectral_boot_tumor_ci_upper <- boot_tumor$ci_upper
    }
  }
  
  # ---- Output ----------------------------------------------------------------
  
  tibble::tibble(
    chip = chip,
    group = group,
    comparison = comparison,
    filter_regime = filter_regime,
    engine = "entropy",
    gene_set_mode = gene_set_mode,
    gene_set_name = gene_set_name,
    
    matrix_key_normal = group_mats$normal_key,
    matrix_key_tumor = group_mats$tumor_key,
    n_shared_probes = group_mats$n_shared_probes,
    
    n_samples_normal = ncol(mat_normal),
    n_samples_tumor = ncol(mat_tumor),
    n_features = nrow(mat_normal),
    
    entropy_covariance_space = covariance_space,
    
    shannon_normal = shannon_normal,
    shannon_tumor = shannon_tumor,
    shannon_delta = shannon_delta,
    shannon_direction = shannon_direction,
    
    spectral_normal = spectral_normal,
    spectral_tumor = spectral_tumor,
    spectral_delta = spectral_delta,
    spectral_direction = spectral_direction,
    
    run_permutation = isTRUE(run_permutation),
    permutation_metric = ifelse(isTRUE(run_permutation), permutation_metric, NA_character_),
    permutation_unit = ifelse(isTRUE(run_permutation), permutation_unit, NA_character_),
    n_perm = ifelse(isTRUE(run_permutation), n_perm, 0L),
    
    p_perm_shannon = p_perm_shannon,
    p_perm_spectral = p_perm_spectral,
    
    run_bootstrap = isTRUE(run_bootstrap),
    bootstrap_metric = ifelse(isTRUE(run_bootstrap), bootstrap_metric, NA_character_),
    bootstrap_unit = ifelse(isTRUE(run_bootstrap), bootstrap_unit, NA_character_),
    n_boot = ifelse(isTRUE(run_bootstrap), n_boot, 0L),
    
    shannon_boot_normal_mean = shannon_boot_normal_mean,
    shannon_boot_normal_ci_lower = shannon_boot_normal_ci_lower,
    shannon_boot_normal_ci_upper = shannon_boot_normal_ci_upper,
    
    shannon_boot_tumor_mean = shannon_boot_tumor_mean,
    shannon_boot_tumor_ci_lower = shannon_boot_tumor_ci_lower,
    shannon_boot_tumor_ci_upper = shannon_boot_tumor_ci_upper,
    
    spectral_boot_normal_mean = spectral_boot_normal_mean,
    spectral_boot_normal_ci_lower = spectral_boot_normal_ci_lower,
    spectral_boot_normal_ci_upper = spectral_boot_normal_ci_upper,
    
    spectral_boot_tumor_mean = spectral_boot_tumor_mean,
    spectral_boot_tumor_ci_lower = spectral_boot_tumor_ci_lower,
    spectral_boot_tumor_ci_upper = spectral_boot_tumor_ci_upper,
    
    status = "ok"
  )
}