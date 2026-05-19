# ------------------------------------------------------------------------------
# File: write_kegg_complexity_table_html.R
# Purpose: Generate comparison-level KEGG pathway reporting tables for complexity
#   results and write them as HTML fragments for Quarto integration.
# Role: Helper (KEGG complexity table writer)
# Pipeline: Reporting
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Write KEGG Complexity Table (HTML)
#'
#' Generates a comparison-specific reporting table of significant KEGG pathways
#' based on complexity analysis results and writes it to an HTML file.
#'
#' The table is grouped by direction of structural change (e.g., "gained",
#' "lost") and includes pathway names and associated p-values.
#'
#' Output is formatted as an HTML fragment for inclusion in Quarto-generated
#' reports.
#'
#' @param comparison Character string identifying the comparison
#'   (e.g., "BR/BRAD").
#' @param complexity_df Data frame containing complexity results.
#' @param output_dir Directory where the HTML table will be written.
#'
#' @details
#' Filters KEGG-mode complexity results with permutation p-value ≤ 0.05, orders
#' results by direction and pathway name, and splits output into directional
#' subgroups.
#'
#' @return Invisibly returns the output file path.

write_kegg_complexity_table_html <- function(comparison, complexity_df, output_dir = "quarto/resources/tables") {
  clean_name <- gsub("[^a-zA-Z0-9]", "_", tolower(comparison))
  out_file <- file.path(output_dir, paste0(clean_name, "_kegg_complexity.html"))
  
  kegg_complex <- complexity_df %>%
    dplyr::filter(
      comparison == !!comparison,
      mode == "KEGG",
      kegg_cancer == TRUE | p_perm <= 0.05
    ) %>%
    dplyr::mutate(direction = factor(direction, levels = c("gained", "lost"))) %>%
    dplyr::arrange(direction, gene_set_name, p_perm)
  
  if (nrow(kegg_complex) == 0) {
    writeLines("<p><em>No significant KEGG pathways found.</em></p>", out_file)
    return(invisible(out_file))
  }
  
  kegg_complex_split <- dplyr::group_split(kegg_complex, direction)
  
  html_blocks <- lapply(kegg_complex_split, function(subgroup) {
    subgroup_title <- stringr::str_to_title(unique(subgroup$direction))
    
    tbl <- subgroup %>%
      dplyr::mutate(`Cancer Pathway` = ifelse(kegg_cancer, "✅", "")) %>%
      dplyr::select(GeneSet = gene_set_name, `p-value` = p_perm, `Cancer Pathway`) %>%
      dplyr::arrange(GeneSet) %>%
      dplyr::mutate(`p-value` = sprintf("%.3f", `p-value`))
    
    table_html <- tbl %>%
      kableExtra::kable("html", align = c("l", "r", "c")) %>%
      kableExtra::kable_styling(
        bootstrap_options = c("striped", "hover", "condensed"),
        full_width = TRUE
      )
    
    paste0(
      "<h4>", subgroup_title, " Complexity</h4>\n",
      as.character(table_html)
    )
  })
  
  full_html <- paste(html_blocks, collapse = "\n\n")
  writeLines(full_html, out_file)
  invisible(out_file)
}
