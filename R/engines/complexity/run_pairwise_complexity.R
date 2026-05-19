# ==============================================================================
# File: run_pairwise_complexity.R
# Purpose: Run pairwise complexity analysis across chips and probe-selection regimes
# Role: Engine-level loop for the Global Cancer structural inference pipeline
# ==============================================================================

#' Run pairwise complexity analysis
#'
#' This function iterates over chips, resolved probe-selection regimes, cancer
#' groups, and normal/tumor comparisons. It computes core complexity descriptors
#' for each eligible comparison and optionally invokes permutation/bootstrap
#' inferential layers when enabled by the calling wrapper.
#'
#' Loading of matrix maps and comparison maps is handled upstream. Saving of
#' results is also handled upstream. This function performs computation only.
#'
#' Required functions in environment:
#'   - load_required_rds()
#'   - extract_selected_probes()
#'   - compare_pair_complexity()
#'   - get_svd_kappa()
#'
#' @param matrix_lookup Named list of expression-matrix lists by chip.
#' @param comparison_lookup Named list of comparison-map objects by chip.
#' @param filtered_probes_dir Directory containing filtered-probe RDS files.
#' @param chips Character vector of chip/platform IDs.
#' @param filter_regimes Character vector of resolved probe-selection regimes.
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
#'   Usually "sample", corresponding to cov(mat).
#' @param seed Optional integer seed for resampling routines.
#'
#' @return A list containing a summary tibble.
run_pairwise_complexity <- function(matrix_lookup,
                                    comparison_lookup,
                                    filtered_probes_dir,
                                    chips,
                                    filter_regimes,
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
  
  all_results <- list()
  
  for (chip in chips) {
    
    message(sprintf("Processing chip: %s", chip))
    
    matrices_i <- matrix_lookup[[chip]]
    comparison_map_i <- comparison_lookup[[chip]]
    
    if (is.null(matrices_i) || is.null(comparison_map_i)) {
      warning(sprintf(
        "Skipping chip %s: missing matrix list or comparison map.",
        chip
      ))
      next
    }
    
    for (filter_regime in filter_regimes) {
      
      message(sprintf("  Probe-selection regime: %s", filter_regime))
      
      filtered_path <- file.path(
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
          warning(sprintf(
            "No comparison map found for chip=%s, regime=%s, group=%s",
            chip,
            filter_regime,
            group_name
          ))
          next
        }
        
        comparisons <- intersect(
          names(group_obj),
          names(comparison_map_group)
        )
        
        if (length(comparisons) == 0) {
          warning(sprintf(
            "No matching comparisons found for chip=%s, regime=%s, group=%s",
            chip,
            filter_regime,
            group_name
          ))
          next
        }
        
        for (comparison in comparisons) {
          
          comparison_obj <- group_obj[[comparison]]
          selected_probes <- extract_selected_probes(comparison_obj)
          
          if (length(selected_probes) < 5) {
            warning(sprintf(
              "Skipping %s / %s / %s / %s: fewer than five selected probes.",
              chip,
              filter_regime,
              group_name,
              comparison
            ))
            next
          }
          
          comparison_labels <- comparison_map_group[[comparison]]
          
          pair_summary <- tryCatch(
            compare_pair_complexity(
              matrices_i = matrices_i,
              comparison_labels = comparison_labels,
              selected_probes = selected_probes,
              comparison = comparison,
              group = group_name,
              chip = chip,
              filter_regime = filter_regime,
              complexity_fn = complexity_fn,
              
              run_permutation = run_permutation,
              n_perm = n_perm,
              permutation_metric = permutation_metric,
              permutation_unit = permutation_unit,
              
              run_bootstrap = run_bootstrap,
              n_boot = n_boot,
              bootstrap_metric = bootstrap_metric,
              bootstrap_unit = bootstrap_unit,
              
              covariance_space = covariance_space,
              seed = seed
            ),
            error = function(e) {
              warning(sprintf(
                "Skipping %s / %s / %s / %s: %s",
                chip,
                filter_regime,
                group_name,
                comparison,
                conditionMessage(e)
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
        "No complexity summaries were generated.",
        "Check filtered-probe files, comparison-map names, matrix-list names,",
        "and compare_pair_complexity() argument compatibility."
      ),
      call. = FALSE
    )
  }
  
  summary <- dplyr::bind_rows(all_results)
  
  if ("p_perm" %in% names(summary)) {
    summary <- summary |>
      dplyr::group_by(chip, filter_regime, gene_set_mode) |>
      dplyr::mutate(
        p_perm_fdr = p.adjust(p_perm, method = "BH")
      ) |>
      dplyr::ungroup()
  }
  
  list(
    summary = summary
  )
}