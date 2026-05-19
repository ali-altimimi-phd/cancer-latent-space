# ==============================================================================
# Compare paired normal/tumor spectral summaries
# ==============================================================================
#
# Purpose:
#   Compute paired spectral summaries for one normal/tumor comparison.
#
# Dependencies:
#   R/engines/spectral/resolve_comparison_matrices.R
#   R/engines/spectral/compute_single_spectral.R
#
# ==============================================================================

#' Compute spectral summaries for one paired comparison
#'
#' @param matrices_i Named list of condition-specific expression matrices.
#' @param comparison_labels Character vector of length two: normal label, tumor label.
#' @param selected_probes Character vector of selected probe IDs.
#' @param comparison Comparison code.
#' @param group Cancer group.
#' @param chip Chip/platform identifier.
#' @param filter_regime Filtering regime label.
#' @param method Spectral method. Currently "marchenko_pastur".
#' @param min_samples_per_condition Minimum samples required in each
#'   condition group before spectral computation.
#' @param use_correlation Logical; if TRUE use correlation, otherwise covariance.
#' @param standardize Logical; if TRUE standardize each probe across samples.
#' @return Tibble with normal and tumor spectral summaries.
compare_pair_spectral <- function(matrices_i,
                                  comparison_labels,
                                  selected_probes,
                                  comparison,
                                  group,
                                  chip,
                                  filter_regime,
                                  method = "marchenko_pastur",
                                  min_samples_per_condition = 5,
                                  use_correlation = TRUE,
                                  standardize = TRUE) {

  group_mats <- get_group_matrices(
    matrices_i = matrices_i,
    comparison_labels = comparison_labels,
    selected_probes = selected_probes
  )

  normal_summary <- compute_single_spectral(
    expr_matrix = group_mats$normal,
    comparison = comparison,
    group = group,
    chip = chip,
    filter_regime = filter_regime,
    condition_label = "normal",
    matrix_key = group_mats$normal_key,
    method = method,
    min_samples_per_condition = min_samples_per_condition,
    use_correlation = use_correlation,
    standardize = standardize
  )

  tumor_summary <- compute_single_spectral(
    expr_matrix = group_mats$tumor,
    comparison = comparison,
    group = group,
    chip = chip,
    filter_regime = filter_regime,
    condition_label = "tumor",
    matrix_key = group_mats$tumor_key,
    method = method,
    min_samples_per_condition = min_samples_per_condition,
    use_correlation = use_correlation,
    standardize = standardize
  )

  dplyr::bind_rows(normal_summary, tumor_summary)
}
