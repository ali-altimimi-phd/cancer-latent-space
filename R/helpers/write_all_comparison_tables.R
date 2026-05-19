# ------------------------------------------------------------------------------
# File: write_all_comparison_tables.R
# Purpose: Write comparison-level reporting tables for GO, KEGG, and MSigDB
#   results across complexity and entropy engines.
# Role: Helper (table generation dispatcher)
# Pipeline: Reporting
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Write All Reporting Tables for a Comparison
#'
#' Generates comparison-specific reporting tables for GO, KEGG, and MSigDB
#' gene-set collections using both complexity and entropy results.
#'
#' This function serves as a dispatcher that calls specialized table-writing
#' helpers for each gene-set collection and analysis engine. Output tables are
#' written to disk (typically as HTML) for downstream inclusion in Quarto-based
#' reports.
#'
#' Used within `run_all_reports()` to modularize and centralize table generation.
#'
#' @param comparison Character string identifying the comparison
#'   (e.g., "BR/BRAD").
#' @param complexity_df Data frame containing complexity results.
#' @param entropy_df Data frame containing entropy results.
#' @param output_dir Directory where reporting tables will be written.
#'
#' @details
#' Delegates table generation to the following helpers:
#' \itemize{
#'   \item GO: \code{write_go_complexity_table_html()},
#'         \code{write_go_entropy_table_html()}
#'   \item KEGG: \code{write_kegg_complexity_table_html()},
#'         \code{write_kegg_entropy_table_html()}
#'   \item MSigDB: \code{write_msig_complexity_table_html()},
#'         \code{write_msig_entropy_table_html()}
#' }
#'
#' @return Invisibly returns \code{NULL}; writes reporting tables to disk.
write_all_comparison_tables <- function(comparison,
                                        complexity_df,
                                        entropy_df,
                                        output_dir) {
  
  source(here::here("R/tables/write_go_complexity_table_html.R"))
  source(here::here("R/tables/write_go_entropy_table_html.R"))
  source(here::here("R/tables/write_kegg_complexity_table_html.R"))
  source(here::here("R/tables/write_kegg_entropy_table_html.R"))
  source(here::here("R/tables/write_msig_complexity_table_html.R"))
  source(here::here("R/tables/write_msig_entropy_table_html.R"))
  
  write_go_complexity_table_html(
    comparison    = comparison,
    complexity_df = complexity_df,
    output_dir    = output_dir
  )
  
  write_go_entropy_table_html(
    comparison = comparison,
    entropy_df = entropy_df,
    output_dir = output_dir
  )
  
  write_kegg_complexity_table_html(
    comparison    = comparison,
    complexity_df = complexity_df,
    output_dir    = output_dir
  )
  
  write_kegg_entropy_table_html(
    comparison = comparison,
    entropy_df = entropy_df,
    output_dir = output_dir
  )
  
  write_msig_complexity_table_html(
    comparison    = comparison,
    complexity_df = complexity_df,
    output_dir    = output_dir
  )
  
  write_msig_entropy_table_html(
    comparison = comparison,
    entropy_df = entropy_df,
    output_dir = output_dir
  )
  
  invisible(NULL)
}