# ------------------------------------------------------------------------------
# File: run_grouped_probe_filtering.R
# Purpose: Apply grouped probe selection with comparison/global variance support
# Role: Structural inference filtering wrapper
# Pipeline: Structural Inference
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Grouped structural probe-selection wrapper
#'
#' Supports:
#'   - limma comparison-specific filtering
#'   - variance comparison-specific filtering
#'   - variance global chip-level filtering
#'
#' @param matrix_list Named list of expression matrices.
#' @param comparison_map Named list of group -> label -> tissue comparisons.
#' @param chip_id Chip ID, e.g. `"hu35ksuba"`.
#' @param method Filtering method: `"limma"` or `"variance"`.
#' @param filter_scope Filtering scope: `"comparison"` or `"global"`.
#' @param logfc_cutoff Log2 fold-change threshold for limma.
#' @param pval_cutoff Adjusted p-value threshold for limma.
#' @param var_threshold Quantile threshold for variance filtering.
#' @param top_n Number of high-variance probes to retain.
#' @param variance_selection_mode `"top_n"` or `"threshold"`.
#' @param save_path Optional RDS save path.
#'
#' @return Nested filtered-probe result object.
run_grouped_probe_filtering <- function(matrix_list,
                                        comparison_map,
                                        chip_id,
                                        method,
                                        filter_scope = "comparison",
                                        logfc_cutoff = NULL,
                                        pval_cutoff = NULL,
                                        var_threshold = NULL,
                                        top_n = NULL,
                                        variance_selection_mode = NULL,
                                        save_path = NULL) {
  
  if (!method %in% c("limma", "variance")) {
    stop("Unknown filtering method: use 'limma' or 'variance'.", call. = FALSE)
  }
  
  if (!filter_scope %in% c("comparison", "global")) {
    stop("filter_scope must be either 'comparison' or 'global'.", call. = FALSE)
  }
  
  if (identical(method, "limma") && !identical(filter_scope, "comparison")) {
    stop(
      "limma filtering is comparison-specific and cannot be run with filter_scope = 'global'.",
      call. = FALSE
    )
  }
  
  if (identical(method, "variance")) {
    if (is.null(variance_selection_mode)) {
      variance_selection_mode <- if (!is.null(top_n)) "top_n" else "threshold"
    }
    
    if (!variance_selection_mode %in% c("top_n", "threshold")) {
      stop(
        "variance_selection_mode must be either 'top_n' or 'threshold'.",
        call. = FALSE
      )
    }
  }
  
  results <- list()
  probe_tracker <- list()
  
  # ---------------------------------------------------------------------------
  # Global variance probe set
  # ---------------------------------------------------------------------------
  
  global_variance_probes <- NULL
  
  if (identical(method, "variance") && identical(filter_scope, "global")) {
    
    common_probes <- Reduce(
      intersect,
      lapply(matrix_list, rownames)
    )
    
    if (length(common_probes) == 0) {
      stop(
        sprintf("No common probes found across matrix_list for chip: %s", chip_id),
        call. = FALSE
      )
    }
    
    chip_matrix <- do.call(
      cbind,
      lapply(matrix_list, function(x) {
        x[common_probes, , drop = FALSE]
      })
    )
    
    global_variance_probes <- select_high_variance_probes(
      chip_matrix,
      method = "mad",
      threshold = var_threshold,
      top_n = top_n
    )
  }
  
  # ---------------------------------------------------------------------------
  # Per-comparison filtering / matrix construction
  # ---------------------------------------------------------------------------
  
  for (group in names(comparison_map)) {
    results[[group]] <- list()
    
    for (label in names(comparison_map[[group]])) {
      
      pair <- comparison_map[[group]][[label]]
      ctrl <- paste0("m_", make.names(pair[1]))
      case <- paste0("m_", make.names(pair[2]))
      
      if (!(ctrl %in% names(matrix_list)) || !(case %in% names(matrix_list))) {
        warning(glue::glue("Skipping comparison '{label}' — matrix not found."))
        next
      }
      
      ctrl_mat <- matrix_list[[ctrl]]
      case_mat <- matrix_list[[case]]
      
      common_pair_probes <- intersect(rownames(ctrl_mat), rownames(case_mat))
      
      if (length(common_pair_probes) == 0) {
        warning(glue::glue("Skipping comparison '{label}' — no common probes."))
        next
      }
      
      ctrl_mat <- ctrl_mat[common_pair_probes, , drop = FALSE]
      case_mat <- case_mat[common_pair_probes, , drop = FALSE]
      combined <- cbind(ctrl_mat, case_mat)
      
      group_labels <- factor(
        c(
          rep("control", ncol(ctrl_mat)),
          rep("case", ncol(case_mat))
        )
      )
      
      if (identical(method, "limma")) {
        
        design <- model.matrix(~ group_labels)
        fit <- limma::lmFit(combined, design)
        fit <- limma::eBayes(fit)
        
        tab <- limma::topTable(
          fit,
          coef = 2,
          number = Inf,
          sort.by = "none"
        ) |>
          tibble::rownames_to_column(var = "probe_id")
        
        filtered <- tab |>
          dplyr::filter(abs(logFC) >= logfc_cutoff, adj.P.Val < pval_cutoff)
        
        probe_ids <- filtered$probe_id
        
        probe_tracker[[paste(group, label, sep = "::")]] <- probe_ids
        
        results[[group]][[label]] <- list(
          filtered_probes = probe_ids,
          filtered_matrix = combined[probe_ids, , drop = FALSE],
          stats_table = filtered,
          limma_full = tab,
          metadata = list(
            chip = chip_id,
            group = group,
            label = label,
            comparison = pair,
            method = method,
            filter_scope = filter_scope
          )
        )
        
      } else if (identical(method, "variance")) {
        
        if (identical(filter_scope, "comparison")) {
          probe_ids <- select_high_variance_probes(
            combined,
            method = "mad",
            threshold = var_threshold,
            top_n = top_n
          )
        } else {
          probe_ids <- intersect(global_variance_probes, rownames(combined))
        }
        
        probe_tracker[[paste(group, label, sep = "::")]] <- probe_ids
        
        results[[group]][[label]] <- list(
          filtered_probes = probe_ids,
          filtered_matrix = combined[probe_ids, , drop = FALSE],
          metadata = list(
            chip = chip_id,
            group = group,
            label = label,
            comparison = pair,
            method = method,
            filter_scope = filter_scope,
            variance_selection_mode = variance_selection_mode
          )
        )
      }
    }
  }
  
  # ---------------------------------------------------------------------------
  # SUMMARY: Cross-comparison probe usage
  # ---------------------------------------------------------------------------
  
  all_probes <- unlist(probe_tracker)
  
  results$`__summary__` <- list(
    probe_hits_by_group = probe_tracker,
    probe_multi_hit_counts = sort(table(all_probes), decreasing = TRUE),
    global_variance_probes = global_variance_probes
  )
  
  # ---------------------------------------------------------------------------
  # METADATA: Pipeline provenance
  # ---------------------------------------------------------------------------
  
  results$`__metadata__` <- list(
    pipeline_stage = "structural_probe_filtering",
    wrapper = "R/wrappers/run_grouped_probe_filtering.R",
    chip_id = chip_id,
    created_at = as.character(Sys.time()),
    method = method,
    filter_scope = filter_scope,
    variance_selection_mode = variance_selection_mode,
    parameters = list(
      logfc_cutoff = if (identical(method, "limma")) logfc_cutoff else NULL,
      pval_cutoff = if (identical(method, "limma")) pval_cutoff else NULL,
      var_threshold = if (
        identical(method, "variance") &&
        identical(variance_selection_mode, "threshold")
      ) {
        var_threshold
      } else {
        NULL
      },
      top_n = if (
        identical(method, "variance") &&
        identical(variance_selection_mode, "top_n")
      ) {
        top_n
      } else {
        NULL
      }
    )
  )
  
  # ---------------------------------------------------------------------------
  # SAVE
  # ---------------------------------------------------------------------------
  
  if (!is.null(save_path)) {
    saveRDS(results, file = save_path)
  }
  
  return(results)
}