# ------------------------------------------------------------------------------
# File: summarize_pairwise_results.R
# Purpose: Generate human-readable summary reports from aggregated
#   complexity or entropy results for comparison-level interpretation.
# Role: Helper (text summary generator)
# Pipeline: Reporting
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Summarize Pairwise Results for Reporting
#'
#' Generates a human-readable textual summary of comparison-level results for
#' either complexity or entropy analyses. The output includes overall probe-level
#' summaries and significant gene-set findings for GO, KEGG, and MSigDB Hallmark
#' collections.
#'
#' @param df Data frame containing either complexity or entropy results.
#' @param engine Character string specifying the analysis engine:
#'   \code{"complexity"} or \code{"entropy"}.
#' @param p_thresh Numeric significance threshold for reporting gene sets
#'   (default = 0.05).
#' @param output_path Optional file path to write the summary report. If
#'   \code{NULL}, the summary is returned as a character vector.
#'
#' @details
#' For complexity results, directionality is derived from the \code{direction}
#' column. For entropy results, directionality is derived from
#' \code{spectral_direction} or \code{shannon_direction}, depending on
#' availability.
#'
#' @return A character vector of formatted summary lines, or invisibly writes to
#'   file if \code{output_path} is provided.

summarize_pairwise_results <- function(df,
                                       engine = c("complexity", "entropy"),
                                       p_thresh = 0.05,
                                       output_path = NULL) {
  get_direction <- function(row, engine) {
    if (engine == "complexity") {
      return(row$direction)
    } else if (engine == "entropy") {
      return(dplyr::coalesce(
        row$spectral_direction,
        row$shannon_direction
      ))
    } else {
      return(NA_character_)
    }
  }
  
  engine <- match.arg(engine)
  lines <- c()
  
  for (chip_id in unique(df$chip)) {
    chip_df <- dplyr::filter(df, chip == chip_id)
    comparisons <- unique(chip_df$comparison)
    
    lines <- c(lines,
               glue::glue("📌 CHIP: {chip_id} — {engine}"),
               strrep("-", 60))
    
    for (cmp in comparisons) {
      cmp_df <- dplyr::filter(chip_df, comparison == cmp)
      lines <- c(lines, glue::glue("comparison: {cmp}"))
      
      # --- Level 1: FULL Probes ---
      all_row <- dplyr::filter(cmp_df, gene_set_normalized == "FULL")
      
      if (nrow(all_row) == 1) {
        p_val <- signif(all_row$p_perm, 4)
        direction <- get_direction(all_row, engine)
        lines <- c(
          lines,
          glue::glue(
            "  FULL Probes:     p = {p_val} | {stringr::str_to_title(direction)}"
          )
        )
      } else {
        lines <- c(lines, "  FULL Probes:     (missing or malformed)")
      }
      
      # --- Level 2a: Significant GO Terms ---
      go_sig <- cmp_df |>
        dplyr::filter(
          stringr::str_starts(gene_set_normalized, "GO:"),!is.na(p_perm),
          p_perm <= p_thresh
        )
      
      if (nrow(go_sig) > 0) {
        lines <- c(lines, "  ▸ Significant GO Terms:")
        for (i in seq_len(nrow(go_sig))) {
          row <- go_sig[i, ]
          direction <- get_direction(row, engine)
          name <- dplyr::coalesce(row$gene_set_name, row$gene_set_normalized)
          lines <- c(
            lines,
            glue::glue(
              "    - {name}: p = {signif(row$p_perm, 4)} | {stringr::str_to_title(direction)}"
            )
          )
        }
      }
      
      # --- Level 2b: Significant KEGG Pathways + Always Include Cancer Pathways ---
      kegg_sig <- cmp_df |>
        dplyr::filter(
          stringr::str_detect(gene_set_normalized, "^[0-9]{4,5}$"),!is.na(p_perm),
          p_perm <= p_thresh
        )
      
      # Add KEGG cancer pathways regardless of p-value
      kegg_cancer <- cmp_df |>
        dplyr::filter(
          stringr::str_detect(gene_set_normalized, "^[0-9]{4,5}$"),
          kegg_cancer == TRUE
        )
      
      # Combine and remove duplicates
      kegg_combined <- dplyr::bind_rows(kegg_sig, kegg_cancer) |>
        dplyr::distinct(gene_set_normalized, .keep_all = TRUE)
      
      if (nrow(kegg_combined) > 0) {
        lines <- c(lines, "  ▸ Significant or Cancer-related KEGG Pathways:")
        for (i in seq_len(nrow(kegg_combined))) {
          row <- kegg_combined[i, ]
          direction <- get_direction(row, engine)
          name <- dplyr::coalesce(row$gene_set_name, row$gene_set_normalized)
          flag <- if (row$kegg_cancer)
            " (cancer)"
          else
            ""
          lines <- c(
            lines,
            glue::glue(
              "    - {name}{flag}: p = {signif(row$p_perm, 4)} | {stringr::str_to_title(direction)}"
            )
          )
        }
      }
      
      # --- Fallback: No significant gene sets ---
      if (nrow(go_sig) == 0 && nrow(kegg_sig) == 0) {
        lines <- c(lines, "  ▸ No significant GO/KEGG terms (p <= threshold)")
      }
      
    }
    
    # --- Level 2c: MSig Hallmark Genes ---
    msig_sig <- cmp_df |>
      dplyr::filter(
        stringr::str_detect(gene_set_normalized, "HALLMARK"),!is.na(p_perm),
        p_perm <= p_thresh
      )
    
    if (nrow(msig_sig) > 0) {
      lines <- c(lines, "  ▸ Significant HALLMARK Genes:")
      for (i in seq_len(nrow(msig_sig))) {
        row <- msig_sig[i, ]
        direction <- get_direction(row, engine)
        name <- dplyr::coalesce(row$gene_set_name, row$gene_set_normalized)
        lines <- c(
          lines,
          glue::glue(
            "    - {name}: p = {signif(row$p_perm, 4)} | {stringr::str_to_title(direction)}"
          )
        )
      }
    }
    
    lines <- c(lines, strrep("-", 60))  # separator line
    
  }
  
  # --- Output to file or return as character vector ---
  if (!is.null(output_path)) {
    readr::write_lines(lines, output_path)
    message(glue::glue("📄 Summary saved to: {output_path}"))
  } else {
    return(lines)
  }
}
