#' Build Pairwise Comparison Input List from Filtered Results
#'
#' Constructs a flat list of pairwise comparison input objects for complexity/entropy engines,
#' using group-aware filtered matrices. Assumes tissue-specific filtering was applied,
#' and results are structured by group and comparison name.
#'
#' @param comparison_map A named list of comparison pairs per group
#'   (e.g., from `define_predefined_comparisons()`).
#' @param filtered_results A named list of results per group and comparison, as returned by
#'   `run_grouped_probe_filtering()`.
#' @param chip_id Character. Chip name used for output tracking.
#'
#' @return A named list where each entry contains:
#'   - `name`: Comparison name (e.g., `"LU/LUAD"`)
#'   - `group`: Group name (e.g., `"carcinomas"`)
#'   - `chip`: Chip ID
#'   - `m1`: Control matrix (filtered)
#'   - `m2`: Case matrix (filtered)

build_comparison_input_list <- function(comparison_map,
                                        filtered_results,
                                        chip_id) {
  stopifnot(is.list(comparison_map), is.list(filtered_results))
  
  out <- list()
  
  for (group_name in names(comparison_map)) {
    group_comparisons <- comparison_map[[group_name]]
    
    for (cmp_name in names(group_comparisons)) {
      if (!group_name %in% names(filtered_results)) next
      if (!cmp_name %in% names(filtered_results[[group_name]])) next
      
      comparison_entry <- filtered_results[[group_name]][[cmp_name]]
      mat <- comparison_entry$filtered_matrix
      
      # Skip comparisons with missing or empty matrices
      if (is.null(mat) || nrow(mat) == 0 || ncol(mat) == 0) next
      
      # Split back into m1 and m2 based on known sample order
      cmp_pair <- group_comparisons[[cmp_name]]
      ctrl_n <- sum(grepl("normal", cmp_pair[1], ignore.case = TRUE))
      case_n <- sum(grepl("cancer|tumor|carcinoma|adenocarcinoma", cmp_pair[2], ignore.case = TRUE))
      
      total_cols <- ncol(mat)
      split_point <- floor(total_cols / 2)
      m1 <- mat[, seq_len(split_point), drop = FALSE]
      m2 <- mat[, (split_point + 1):total_cols, drop = FALSE]
      
      out[[cmp_name]] <- list(
        name = cmp_name,
        group = group_name,
        chip = chip_id,
        m1 = m1,
        m2 = m2
      )
    }
  }
  
  return(out)
}
