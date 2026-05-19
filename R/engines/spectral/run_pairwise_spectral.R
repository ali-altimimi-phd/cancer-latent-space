# ==============================================================================
# Run pairwise spectral analysis
# ==============================================================================
#
# Purpose:
#   Engine-level loop over chips, probe-selection regimes, cancer groups,
#   and comparisons.
#
#   This function performs computation only; loading and saving are handled by
#   the runner.
#
# Dependencies:
#   R/engines/spectral/resolve_comparison_matrices.R
#   R/engines/spectral/compare_pair_spectral.R
#   R/engines/spectral/compute_spectral_deltas.R
#
# ==============================================================================

#' Run pairwise spectral analysis across chips and probe-selection regimes
#'
#' @param matrix_lookup Named list of matrix lists by chip.
#' @param comparison_lookup Named list of comparison-map objects by chip.
#' @param filtered_probes_dir Directory containing filtered probe RDS files.
#' @param chips Character vector of chips/platforms to analyze.
#' @param filter_regimes Character vector of fully resolved probe-selection regimes.
#' @param method Spectral method. Currently "marchenko_pastur".
#' @param min_samples_per_condition Minimum samples required in each condition
#'   group before structural computation.
#' @param min_selected_probes Minimum number of selected probes required before
#'   spectral computation.
#' @param use_correlation Logical; if TRUE use correlation, otherwise covariance.
#' @param standardize Logical; if TRUE standardize each probe across samples.
#'
#' @return List with summary and deltas tibbles.
run_pairwise_spectral <- function(matrix_lookup,
                                  comparison_lookup,
                                  filtered_probes_dir,
                                  chips,
                                  filter_regimes,
                                  method = "marchenko_pastur",
                                  min_samples_per_condition = 5,
                                  min_selected_probes = 5,
                                  use_correlation = TRUE,
                                  standardize = TRUE) {
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
          
          if (length(selected_probes) < min_selected_probes) {
            warning(sprintf(
              paste(
                "Skipping %s / %s / %s / %s:",
                "selected probes = %d, minimum required = %d."
              ),
              chip, filter_regime, group_name, comparison,
              length(selected_probes), min_selected_probes
            ))
            next
          }
          
          comparison_labels <- comparison_map_group[[comparison]]
          
          pair_summary <- tryCatch(
            compare_pair_spectral(
              matrices_i = matrices_i,
              comparison_labels = comparison_labels,
              selected_probes = selected_probes,
              comparison = comparison,
              group = group_name,
              chip = chip,
              filter_regime = filter_regime,
              method = method,
              min_samples_per_condition = min_samples_per_condition,
              use_correlation = use_correlation,
              standardize = standardize
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
        "No spectral summaries were generated.",
        "Check filtered-probe files, comparison-map names, matrix-list names,",
        "and compare_pair_spectral() argument compatibility."
      ),
      call. = FALSE
    )
  }
  
  summary <- dplyr::bind_rows(all_results)
  deltas <- compute_spectral_deltas(summary)
  
  list(
    summary = summary,
    deltas = deltas
  )
}