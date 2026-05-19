# ------------------------------------------------------------------------------
# File: write_msig_entropy_table_html.R
# Purpose: Generate comparison-level MSigDB Hallmark reporting tables for
#   entropy results and write them as HTML fragments for Quarto integration.
# Role: Helper (MSigDB entropy table writer)
# Pipeline: Reporting
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Write MSigDB Hallmark Entropy Table (HTML)
#'
#' Generates a comparison-specific reporting table of significant MSigDB
#' Hallmark gene sets based on entropy analysis results and writes it to an
#' HTML file.
#'
#' The table groups gene sets by spectral entropy direction, using the project
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
#' Filters MSigDB Hallmark entropy results with permutation p-value ≤ 0.05,
#' classifies gene sets according to \code{spectral_direction}, orders results
#' by group and gene set name, and writes grouped HTML table blocks.
#'
#' @return Invisibly returns the output file path.

write_msig_entropy_table_html <- function(comparison, entropy_df, output_dir = "quarto/resources/tables/msig") {
  fs::dir_create(output_dir)
  clean_name <- gsub("[^a-zA-Z0-9]", "_", tolower(comparison))
  out_file <- file.path(output_dir, paste0(clean_name, "_msig_entropy.html"))
  
  msig_entropy <- entropy_df %>%
    dplyr::filter(
      comparison == !!comparison,
      mode == "MSIG",
      p_perm <= 0.05
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
  
  if (nrow(msig_entropy) == 0) {
    writeLines("<p><em>No significant MSigDB gene sets found.</em></p>", out_file)
    return(invisible(out_file))
  }
  
  msig_entropy_split <- dplyr::group_split(msig_entropy, group)
  
  html_blocks <- lapply(msig_entropy_split, function(subgroup) {
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
