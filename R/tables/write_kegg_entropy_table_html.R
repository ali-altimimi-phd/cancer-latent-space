# ------------------------------------------------------------------------------
# File: write_kegg_entropy_table_html.R
# Purpose: Generate comparison-level KEGG pathway reporting tables for entropy
#   results and write them as HTML fragments for Quarto integration.
# Role: Helper (KEGG entropy table writer)
# Pipeline: Reporting
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Write KEGG Entropy Table (HTML)
#'
#' Generates a comparison-specific reporting table of significant KEGG pathways
#' based on entropy analysis results and writes it to an HTML file.
#'
#' The table groups pathways by spectral entropy direction, using the project
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
#' Filters KEGG-mode entropy results with permutation p-value ≤ 0.05, classifies
#' pathways according to \code{spectral_direction}, orders results by group and
#' pathway name, and writes grouped HTML table blocks.
#'
#' @return Invisibly returns \code{NULL}; writes an HTML file to disk.

write_kegg_entropy_table_html <- function(comparison, entropy_df, output_dir = "quarto/resources/tables") {
  clean_name <- gsub("[^a-zA-Z0-9]", "_", tolower(comparison))
  out_file <- file.path(output_dir, paste0(clean_name, "_kegg_entropy.html"))
  
  kegg_entropy <- entropy_df %>%
    dplyr::filter(
      comparison == !!comparison,
      mode == "KEGG",
      kegg_cancer == TRUE | p_perm <= 0.05
    ) %>%
    dplyr::mutate(
      group = dplyr::case_when(
        grepl("strongly anti-chaotic", spectral_direction, ignore.case = TRUE) ~ "Strongly Anti-Chaotic",
        grepl("mildly anti-chaotic",   spectral_direction, ignore.case = TRUE) ~ "Mildly Anti-Chaotic",
        grepl("neutral",               spectral_direction, ignore.case = TRUE) ~ "Neutral",
        grepl("mildly chaotic",        spectral_direction, ignore.case = TRUE) ~ "Mildly Chaotic",
        grepl("strongly chaotic",      spectral_direction, ignore.case = TRUE) ~ "Strongly Chaotic",
        TRUE ~ "Uncategorized"
      ),
      group = factor(
        group,
        levels = c(
          "Strongly Anti-Chaotic", "Mildly Anti-Chaotic", "Neutral",
          "Mildly Chaotic", "Strongly Chaotic", "Uncategorized"
        )
      )
    ) %>%
    dplyr::arrange(group, gene_set_name, p_perm)
  
  if (nrow(kegg_entropy) == 0) {
    writeLines("<p><em>No significant KEGG pathways found.</em></p>", out_file)
    return(invisible(out_file))
  }
  
  kegg_entropy_split <- dplyr::group_split(kegg_entropy, group)
  
  html_blocks <- lapply(kegg_entropy_split, function(subgroup) {
    subgroup_title <- unique(subgroup$group)
    
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
      "<h4>", subgroup_title, "</h4>\n",
      as.character(table_html)
    )
  })
  
  full_html <- paste(html_blocks, collapse = "\n\n")
  writeLines(full_html, out_file)
  invisible(out_file)
}
