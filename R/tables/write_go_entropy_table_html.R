# ------------------------------------------------------------------------------
# File: write_go_entropy_table_html.R
# Purpose: Generate comparison-level GO term reporting tables for entropy
#   results and write them as HTML fragments for Quarto integration.
# Role: Helper (GO entropy table writer)
# Pipeline: Reporting
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Write GO Entropy Table (HTML)
#'
#' Generates a comparison-specific reporting table of significant GO terms based
#' on entropy analysis results and writes it to an HTML file.
#'
#' The table groups terms by spectral entropy direction, using the project
#' interpretation labels for anti-chaotic, neutral, and chaotic structure.
#'
#' Output is formatted as an HTML fragment for inclusion in Quarto-generated
#' reports.
#'
#' @param comparison Character string identifying the comparison
#'   (e.g., "BR/BRAD").
#' @param entropy_df Data frame containing entropy results.
#' @param output_dir Directory where the HTML table will be written.
#'
#' @details
#' Filters GO-mode entropy results with permutation p-value ≤ 0.05, classifies
#' terms according to \code{spectral_direction}, orders results by group and
#' gene set name, and writes grouped HTML table blocks.
#'
#' @return Invisibly returns \code{NULL}; writes an HTML file to disk.

write_go_entropy_table_html <- function(comparison, entropy_df, output_dir = "quarto/resources/tables/go") {
  clean_name <- gsub("[^a-zA-Z0-9]", "_", tolower(comparison))
  out_file <- file.path(output_dir, paste0(clean_name, "_go_entropy.html"))
  
  go_entropy <- entropy_df %>%
    dplyr::filter(comparison == !!comparison, mode == "GO", p_perm <= 0.05) %>%
    dplyr::mutate(group = dplyr::case_when(
      grepl("strongly anti-chaotic", spectral_direction, ignore.case = TRUE) ~ "Strongly Anti-Chaotic",
      grepl("mildly anti-chaotic", spectral_direction, ignore.case = TRUE)   ~ "Mildly Anti-Chaotic",
      grepl("neutral", spectral_direction, ignore.case = TRUE)               ~ "Neutral",
      grepl("mildly chaotic", spectral_direction, ignore.case = TRUE)        ~ "Mildly Chaotic",
      grepl("strongly chaotic", spectral_direction, ignore.case = TRUE)      ~ "Strongly Chaotic",
      TRUE ~ "Uncategorized"
    )) %>%
    dplyr::mutate(group = factor(group, levels = c(
      "Strongly Anti-Chaotic", "Mildly Anti-Chaotic", "Neutral",
      "Mildly Chaotic", "Strongly Chaotic", "Uncategorized"
    ))) %>%
    dplyr::arrange(group, gene_set_name, p_perm)
  
  if (nrow(go_entropy) == 0) {
    writeLines("<p><em>No significant GO terms found.</em></p>", out_file)
    return(invisible(out_file))
  }
  
  go_entropy_split <- dplyr::group_split(go_entropy, group)
  
  html_blocks <- lapply(go_entropy_split, function(subgroup) {
    subgroup_title <- unique(subgroup$group)
    
    tbl <- subgroup %>%
      dplyr::select(GeneSet = gene_set_name, `p-value` = p_perm) %>%
      dplyr::arrange(GeneSet) %>%
      dplyr::mutate(`p-value` = sprintf("%.3f", `p-value`))
    
    table_html <- tbl %>%
      kableExtra::kable("html", align = c("l", "r")) %>%
      kableExtra::kable_styling(
        bootstrap_options = c("striped", "hover", "condensed"),
        full_width = TRUE
      )
    
    paste0(
      "<h4>", subgroup_title, "</h4>\n",
      as.character(table_html)
    )
  })
  
  full_html <- paste(html_blocks, collapse = "\n\n")
  writeLines(full_html, out_file)
  invisible(out_file)
}
