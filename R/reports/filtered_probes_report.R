#' Report number of filtered probes per comparison
#'
#' Generates a plain-text report showing how many probes passed filtering
#' for each comparison within each chip.
#'
#' @param chips Character vector: chip IDs (e.g., c("hu35ksuba", "hu6800"))
#' @param output_file Optional file path to write report
#'
#' @return Character vector of report lines (also writes to file if specified)
#' @export
report_filtered_probe_counts <- function(chips = c("hu35ksuba", "hu6800"),
                                         output_file = NULL) {
  lines <- c("🧬 Filtered Probe Summary", strrep("=", 60))
  
  # ---- Optional: include threshold log if exists ----
  thresh_path <- here::here("output/global_cancer/logs/pairwise/dynamic_thresholds.txt")
  if (file.exists(thresh_path)) {
    thresh_lines <- readLines(thresh_path)
    lines <- c(lines, "📊 Dynamic Thresholds:", thresh_lines, strrep("-", 60))
  }
  
  # ---- Loop through chips ----
  for (chip in chips) {
    res_obj_name <- glue::glue("res_{chip}")
    if (!exists(res_obj_name, envir = .GlobalEnv)) {
      warning("Results not found for chip: ", chip)
      next
    }
    
    res <- get(res_obj_name, envir = .GlobalEnv)
    lines <- c(lines, glue::glue("🔹 Chip: {chip}"))
    
    for (group in names(res)) {
      if (group == "__summary__") next
      group_data <- res[[group]]
      lines <- c(lines, glue::glue("  Group: {group}"))
      
      for (cmp_name in names(group_data)) {
        cmp <- group_data[[cmp_name]]
        n <- length(cmp$filtered_probes)
        lines <- c(lines, glue::glue("    - {cmp_name}: {n} probes"))
      }
      
      # Optional: show shared probes across group comparisons
      probe_sets <- lapply(group_data, function(x) x$filtered_probes)
      shared <- Reduce(intersect, probe_sets)
      if (length(shared) > 0) {
        lines <- c(lines, glue::glue("    • Shared probes across group: {length(shared)}"))
      } else {
        lines <- c(lines, "    • No shared probes across group")
      }
    }
    
    lines <- c(lines, strrep("-", 60))
  }
  
  # ---- Output ----
  if (!is.null(output_file)) {
    readr::write_lines(lines, output_file)
    message(glue::glue("📄 Probe count summary written to: {output_file}"))
  }
  
  return(lines)
}
