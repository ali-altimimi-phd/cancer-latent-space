# ==============================================================================
# Run pairwise entropy analysis
# ==============================================================================
#
# Purpose:
#   Engine-level loop over chips, probe-selection regimes, cancer groups,
#   and comparisons.
#
#   This function performs computation only; loading and saving are handled by
#   the structural inference runner.
#
# ==============================================================================

#' Run pairwise entropy analysis across chips and probe-selection regimes
#'
#' @param matrix_lookup Named list of matrix lists by chip.
#' @param comparison_lookup Named list of comparison-map objects by chip.
#' @param filtered_probes_dir Directory containing filtered probe RDS files.
#' @param chips Character vector of chips/platforms to analyze.
#' @param filter_regimes Character vector of fully resolved probe-selection regimes.
#' @param entropy_fn Function used as the primary entropy statistic.
#' @param run_permutation Logical; whether to run entropy permutation testing.
#' @param n_perm Number of permutations for entropy testing.
#' @param run_bootstrap Logical; whether to run entropy bootstrap intervals.
#' @param n_boot Number of bootstrap replicates.
#' @param permutation_metric Entropy metric used for permutation testing.
#' @param bootstrap_metric Entropy metric used for bootstrap intervals.
#' @param covariance_space Space used for spectral entropy covariance: "sample" or "probe".
#' @param permutation_unit Unit used for permutation testing.
#' @param bootstrap_unit Unit used for bootstrap resampling.
#' @param seed Optional random seed.
#'
#' @return List with summary tibble.
run_pairwise_entropy <- function(matrix_lookup,
                                 comparison_lookup,
                                 filtered_probes_dir,
                                 chips,
                                 filter_regimes,
                                 entropy_fn = compute_shannon_entropy,
                                 run_permutation = FALSE,
                                 n_perm = 0,
                                 run_bootstrap = FALSE,
                                 n_boot = 0,
                                 permutation_metric = "shannon",
                                 bootstrap_metric = "shannon",
                                 covariance_space = "sample",
                                 permutation_unit = "sample_label",
                                 bootstrap_unit = "sample",
                                 seed = NULL) {
  
  all_results <- list()
  
  for (chip in chips) {
    
    message(sprintf("Processing chip: %s", chip))
    
    matrices_i <- matrix_lookup[[chip]]
    comparison_map_i <- comparison_lookup[[chip]]
    
    if (is.null(matrices_i) || is.null(comparison_map_i)) {
      warning(sprintf("Skipping chip %s: missing matrix or comparison map.", chip))
      next
    }
    
    for (filter_regime in filter_regimes) {
      
      message(sprintf("  Probe-selection regime: %s", filter_regime))
      
      filtered_path <- here::here(
        filtered_probes_dir,
        sprintf("filtered_probes_%s_%s.rds", chip, filter_regime)
      )
      
      filtered_i <- load_required_rds(filtered_path)
      
      group_names <- setdiff(
        names(filtered_i),
        c("__summary__", "__metadata__")
      )
      
      for (group_name in group_names) {
        
        group_obj <- filtered_i[[group_name]]
        comparison_map_group <- comparison_map_i[[group_name]]
        
        if (is.null(comparison_map_group)) {
          warning(sprintf("No comparison map found for group: %s", group_name))
          next
        }
        
        comparisons <- intersect(
          names(group_obj),
          names(comparison_map_group)
        )
        
        if (length(comparisons) == 0) {
          warning(sprintf(
            "No matching comparisons found for chip=%s, regime=%s, group=%s",
            chip, filter_regime, group_name
          ))
          next
        }
        
        for (comparison in comparisons) {
          
          comparison_obj <- group_obj[[comparison]]
          selected_probes <- extract_selected_probes(comparison_obj)
          
          if (length(selected_probes) < 5) {
            warning(sprintf(
              "Skipping %s / %s / %s / %s: fewer than five selected probes.",
              chip, filter_regime, group_name, comparison
            ))
            next
          }
          
          comparison_labels <- comparison_map_group[[comparison]]
          
          pair_summary <- tryCatch(
            compare_pair_entropy(
              matrices_i = matrices_i,
              comparison_labels = comparison_labels,
              selected_probes = selected_probes,
              comparison = comparison,
              group = group_name,
              chip = chip,
              filter_regime = filter_regime,
              entropy_fn = entropy_fn,
              
              run_permutation = run_permutation,
              n_perm = n_perm,
              
              run_bootstrap = run_bootstrap,
              n_boot = n_boot,
              
              permutation_metric = permutation_metric,
              bootstrap_metric = bootstrap_metric,
              covariance_space = covariance_space,
              
              permutation_unit = permutation_unit,
              bootstrap_unit = bootstrap_unit,
              
              seed = seed
            ),
            error = function(e) {
              warning(sprintf(
                "Skipping %s / %s / %s / %s: %s",
                chip, filter_regime, group_name, comparison, conditionMessage(e)
              ))
              NULL
            }
          )
          
          if (is.null(pair_summary)) {
            next
          }
          
          result_key <- paste(
            chip,
            filter_regime,
            group_name,
            comparison,
            sep = "__"
          )
          
          all_results[[result_key]] <- pair_summary
        }
      }
    }
  }
  
  if (length(all_results) == 0) {
    stop(
      paste(
        "No entropy summaries were generated.",
        "Check filtered-probe files, comparison-map names, matrix-list names,",
        "and compare_pair_entropy() argument compatibility."
      ),
      call. = FALSE
    )
  }
  
  summary <- dplyr::bind_rows(all_results)
  
  if ("p_perm_shannon" %in% names(summary)) {
    
    summary <- summary |>
      dplyr::group_by(chip, filter_regime, gene_set_mode) |>
      dplyr::mutate(
        p_perm_shannon_fdr = if (all(is.na(p_perm_shannon))) {
          NA_real_
        } else {
          p.adjust(p_perm_shannon, method = "BH")
        }
      ) |>
      dplyr::ungroup()
    
  } else {
    
    summary <- summary |>
      dplyr::mutate(
        p_perm_shannon_fdr = NA_real_
      )
  }
  
  if ("p_perm_spectral" %in% names(summary)) {
    
    summary <- summary |>
      dplyr::group_by(chip, filter_regime, gene_set_mode) |>
      dplyr::mutate(
        p_perm_spectral_fdr = if (all(is.na(p_perm_spectral))) {
          NA_real_
        } else {
          p.adjust(p_perm_spectral, method = "BH")
        }
      ) |>
      dplyr::ungroup()
    
  } else {
    
    summary <- summary |>
      dplyr::mutate(
        p_perm_spectral_fdr = NA_real_
      )
  }
  
  list(
    summary = summary
  )
}