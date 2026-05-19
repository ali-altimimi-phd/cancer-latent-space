# term_helpers.R
# -------------------------------------------------------------
# Utilities for extracting significant gene sets by comparison
# -------------------------------------------------------------

#' Extract significant gene sets from a results dataframe
#'
#' Filters for gene sets in a specific comparison and mode that pass a p-value threshold.
#'
#' @param df A results dataframe (e.g., `complexity_df` or `entropy_df`).
#' @param comparison A string identifying the comparison (e.g., "PB/T-ALL").
#' @param mode The gene set mode to filter by (e.g., "GO", "KEGG", "MSIG").
#' @param p_threshold Numeric p-value cutoff for significance (default = 0.05).
#'
#' @return A tibble of significant gene sets for the given comparison and mode.
#' @export
extract_significant_terms <- function(df, comparison, mode, p_threshold = 0.05) {
  df |>
    dplyr::filter(
      .data$comparison == !!comparison,
      .data$mode == !!mode,
      .data$p_perm <= p_threshold
    )
}
