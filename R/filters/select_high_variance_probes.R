# ------------------------------------------------------------------------------
# File: select_high_variance_probes.R
# Purpose: Select probes based on intrinsic variability across samples using
#   variance-based filtering for downstream complexity and entropy analysis
# Role: Probe-level variability filtering utility
# Pipeline: Shared utility (Analysis; Preprocessing refactor planned)
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Select High-Variance Probes
#'
#' @description
#' Identifies probes exhibiting high intrinsic variability across samples,
#' thereby isolating transcriptomic components that contribute most strongly to
#' dispersion and heterogeneity.
#' 
#' In contrast to differential-expression filtering, this approach captures
#' variability independent of class labels (e.g., normal vs tumor), providing
#' a complementary perspective on transcriptomic complexity.
#'
#' The function supports both threshold-based selection and fixed-count
#' selection. Threshold-based selection returns probes whose variability meets
#' or exceeds a chosen quantile cutoff. Fixed-count selection returns the top
#' `top_n` probes ranked by the requested variability statistic.
#' 
#' If `top_n` is supplied, fixed-count selection is used and `threshold`
#' is ignored. If `top_n = NULL`, threshold-based quantile selection is used.
#' The calling pipeline is responsible for deciding which mode to use.
#'
#' @param expr_matrix A numeric matrix in probes x samples orientation.
#' @param method Character string specifying the variability statistic.
#'   Supported values are `"mad"`, `"sd"`, `"var"`, and `"variance"`.
#'   The `"variance"` and `"var"` options are equivalent.
#' @param threshold Numeric quantile cutoff used when `top_n` is `NULL`.
#'   Defaults to `0.75`.
#' @param top_n Optional integer. If supplied, returns the `top_n` probes with
#'   the highest variability and ignores `threshold`. If `NULL`, selection is
#'   performed using `threshold`.
#' @param return_stats Logical. If `TRUE`, returns a list containing retained
#'   probes, full variability statistics, retained statistics, cutoff, and
#'   method metadata. If `FALSE` (default), returns only a character vector of
#'   retained probe IDs.
#'
#' @return
#' If `return_stats = FALSE`, a character vector of retained probe IDs.
#'
#' If `return_stats = TRUE`, a list with elements:
#' \itemize{
#'   \item `probes`: retained probe IDs
#'   \item `stats`: named numeric vector of variability statistics for all probes
#'   \item `retained_stats`: named numeric vector of retained probe statistics
#'   \item `cutoff`: numeric cutoff used for threshold mode, otherwise `NA_real_`
#'   \item `method`: normalized method name actually used
#'   \item `selection_mode`: either `"threshold"` or `"top_n"`
#' }
#'
#' @examples
#' \dontrun{
#' keep <- select_high_variance_probes(expr_matrix, method = "variance", top_n = 3000)
#'
#' out <- select_high_variance_probes(
#'   expr_matrix,
#'   method = "mad",
#'   threshold = 0.75,
#'   return_stats = TRUE
#' )
#' }
select_high_variance_probes <- function(expr_matrix,
                                        method = "mad",
                                        threshold = 0.75,
                                        top_n = NULL,
                                        return_stats = FALSE) {
  if (!is.matrix(expr_matrix) || !is.numeric(expr_matrix)) {
    stop("Input must be a numeric matrix.", call. = FALSE)
  }

  if (is.null(rownames(expr_matrix))) {
    stop("Expression matrix must have row names corresponding to probe IDs.", call. = FALSE)
  }

  normalized_method <- tolower(method)
  if (identical(normalized_method, "variance")) {
    normalized_method <- "var"
  }

  stats_vec <- switch(
    normalized_method,
    mad = matrixStats::rowMads(expr_matrix, na.rm = TRUE),
    sd  = matrixStats::rowSds(expr_matrix, na.rm = TRUE),
    var = matrixStats::rowVars(expr_matrix, na.rm = TRUE),
    stop("Method must be one of 'mad', 'sd', 'var', or 'variance'.", call. = FALSE)
  )

  names(stats_vec) <- rownames(expr_matrix)

  if (!is.null(top_n)) {
    if (!is.numeric(top_n) || length(top_n) != 1L || is.na(top_n) || top_n < 1) {
      stop("top_n must be a single positive integer.", call. = FALSE)
    }

    top_n <- min(as.integer(top_n), length(stats_vec))
    keep_idx <- order(stats_vec, decreasing = TRUE)[seq_len(top_n)]
    probes <- names(stats_vec)[keep_idx]
    retained_stats <- stats_vec[keep_idx]
    cutoff <- NA_real_
    selection_mode <- "top_n"
  } else {
    if (!is.numeric(threshold) || length(threshold) != 1L || is.na(threshold) ||
        threshold < 0 || threshold > 1) {
      stop("threshold must be a single numeric value between 0 and 1.", call. = FALSE)
    }

    cutoff <- stats::quantile(stats_vec, threshold, na.rm = TRUE)
    keep_idx <- which(stats_vec >= cutoff)
    probes <- names(stats_vec)[keep_idx]
    retained_stats <- stats_vec[keep_idx]
    selection_mode <- "threshold"
  }

  if (!return_stats) {
    return(probes)
  }

  list(
    probes = probes,
    stats = stats_vec,
    retained_stats = retained_stats,
    cutoff = cutoff,
    method = normalized_method,
    selection_mode = selection_mode
  )
}
