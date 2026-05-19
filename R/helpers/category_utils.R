#' category_utils.R
#' -------------------------------------------------------------
#' Helpers for categorizing comparisons, gene sets, and biological signals
#' Used in reporting and summarization of entropy/complexity results
#' -------------------------------------------------------------

#' Add cancer category to a data frame based on comparison codes
#'
#' @param df Data frame with a "comparison" column.
#' @param comparison_col Name of the column with comparison codes (default: "comparison").
#' @return The input data frame with an added "cancer_category" column.
#' @export
add_cancer_category <- function(df, comparison_col = "comparison") {
  comparison_to_category <- c(
    # Carcinomas
    "BLAD/TCC" = "carcinomas", "BR/BRAD" = "carcinomas", "COL/COADREAD" = "carcinomas",
    "KID/RCC" = "carcinomas", "LU/LUAD" = "carcinomas", "OV/OVAD" = "carcinomas",
    "PA/PAAD" = "carcinomas", "PR/PRAD" = "carcinomas", "UT/EAC" = "carcinomas",
    # Blastomas
    "Brain/GBM" = "blastomas", "Brain/MB" = "blastomas",
    # Lymphomas
    "GC/FL" = "lymphomas", "GC/LBCL" = "lymphomas",
    # Leukemias
    "PB/AML" = "leukemias", "PB/B-ALL" = "leukemias", "PB/T-ALL" = "leukemias"
  )
  
  df$cancer_category <- comparison_to_category[df[[comparison_col]]]
  return(df)
}

#' Group and sort gene sets by direction
#'
#' Organizes gene sets by directionality and sorts them by p-value. Used in
#' reporting of complexity and entropy-based gene set results.
#'
#' @param df Data frame with gene set results.
#' @param direction_col Column name indicating direction (e.g., "direction" or "spectral_direction").
#' @param term_col Column name with gene set names. Default is "gene_set_name".
#' @param p_col Column name with p-values. Default is "p_perm".
#'
#' @return A tibble grouped by direction and sorted by p-value.
#' @export
group_and_sort_by_direction <- function(df,
                                        direction_col = "direction",
                                        term_col = "gene_set_name",
                                        p_col = "p_perm") {
  df |>
    dplyr::transmute(
      term = .data[[term_col]],
      p = .data[[p_col]],
      direction = .data[[direction_col]]
    ) |>
    dplyr::distinct(term, .keep_all = TRUE) |>
    dplyr::arrange(factor(direction, levels = c(
      "strongly gained", "mildly gained", "gained",
      "no clear change",
      "lost", "mildly lost", "strongly lost",
      "strongly chaotic", "mildly chaotic",
      "neutral",
      "mildly anti-chaotic", "strongly anti-chaotic"
    )), p)
}

#' Format grouped gene sets into markdown-style bullet lines
#'
#' Outputs a character vector of bullets: "- TERM: p = 0.012 | direction"
#'
#' @param df Data frame with `term`, `p`, and `direction` columns.
#' @return Character vector of formatted lines.
#' @export
format_gene_set_lines <- function(df) {
  df |>
    dplyr::mutate(p_fmt = sprintf("p = %.3f", p)) |>
    dplyr::rowwise() |>
    dplyr::mutate(line = paste0("- ", term, ": ", p_fmt, " | ", direction)) |>
    dplyr::pull(line)
}
