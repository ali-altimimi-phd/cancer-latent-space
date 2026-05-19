# ------------------------------------------------------------------------------
# File: sanitize_comparison.R
# Purpose: Normalize comparison identifiers into filesystem-safe strings
# Role: Shared helper utility for reporting and output generation
# Pipeline: Shared
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Sanitize Comparison Labels for File and Directory Names
#'
#' Converts comparison identifiers into filesystem-safe strings for use in
#' output filenames, directories, Quarto resources, tables, plots, and
#' intermediate reporting artifacts.
#'
#' The transformation:
#' - converts text to lowercase
#' - replaces all non-alphanumeric characters with underscores
#'
#' This helper is intentionally minimal and stable because it is used across
#' multiple stages of the analysis and reporting pipelines. Behavioral changes
#' should be made cautiously to preserve backward compatibility of generated
#' paths and filenames.
#'
#' Examples:
#' \describe{
#'   \item{"BR/BRAD"}{"br_brad"}
#'   \item{"Kidney RCC"}{"kidney_rcc"}
#'   \item{"COL-COADREAD"}{"col_coadread"}
#' }
#'
#' @param x Character vector of comparison labels.
#'
#' @return Character vector containing sanitized comparison identifiers.
sanitize_comparison <- function(x) {
  gsub("[^A-Za-z0-9]", "_", tolower(x))
}