#' Build Cross-Engine Calibration Table
#'
#' Computes pairwise Spearman correlations among canonical structural metrics.
#' This stage asks whether engines are redundant, strongly coupled, moderately
#' coupled, or approximately independent.

suppressPackageStartupMessages({
  library(DBI)
  library(dplyr)
  library(purrr)
  library(tibble)
  library(readr)
})

# ------------------------------------------------------------------------------
# Helper: safely select existing metric columns
# ------------------------------------------------------------------------------

get_existing_metric_cols <- function(df, metric_cols) {
  existing <- intersect(metric_cols, names(df))
  missing  <- setdiff(metric_cols, names(df))

  if (length(missing) > 0) {
    warning(
      "The following requested metric columns were not found and will be skipped: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  existing
}

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------

build_cross_engine_calibration <- function(
    con,
    output_dir,
    source_view = "vw_structural_phenotype_wide_with_latent_overlay"
) {

  message("Building cross-engine metric calibration table...")

  df <- DBI::dbGetQuery(con, paste0("SELECT * FROM ", source_view))

  # Canonical structural metrics.
  #
  # These should remain intentionally limited. The goal is calibration among
  # representative metrics, not an exhaustive all-metric correlation screen.

  requested_metric_cols <- c(
    "complexity__effrank_delta",
    "complexity__kappa_delta",
    "complexity__composite_kappa_delta",
    "entropy__spectral_delta",
    "entropy__shannon_delta",
    "mp__spectral_entropy_delta",
    "mp__participation_ratio_delta",
    "mp__largest_eigenvalue_fraction_delta",
    "latent__centroid_distance",
    "latent__anisotropy_delta",
    "latent__eig_entropy_delta",
    "latent__radius_delta"
  )

  metric_cols <- get_existing_metric_cols(df, requested_metric_cols)

  if (length(metric_cols) < 2) {
    stop("Fewer than two metric columns are available for calibration.")
  }

  cor_results <- combn(metric_cols, 2, simplify = FALSE) |>
    purrr::map_dfr(function(pair) {
      x <- df[[pair[1]]]
      y <- df[[pair[2]]]
      ok <- stats::complete.cases(x, y)
      n_ok <- sum(ok)

      if (n_ok < 5) {
        return(tibble::tibble(
          metric_x = pair[1],
          metric_y = pair[2],
          n = n_ok,
          spearman_r = NA_real_,
          p_value = NA_real_
        ))
      }

      test <- suppressWarnings(stats::cor.test(x[ok], y[ok], method = "spearman"))

      tibble::tibble(
        metric_x = pair[1],
        metric_y = pair[2],
        n = n_ok,
        spearman_r = unname(test$estimate),
        p_value = test$p.value
      )
    }) |>
    dplyr::mutate(
      p_fdr = stats::p.adjust(p_value, method = "BH"),
      relationship = dplyr::case_when(
        is.na(spearman_r) ~ "insufficient_data",
        abs(spearman_r) >= 0.90 ~ "near_redundant_or_dual",
        abs(spearman_r) >= 0.70 ~ "strongly_coupled",
        abs(spearman_r) >= 0.40 ~ "moderately_coupled",
        TRUE ~ "weak_or_independent"
      )
    ) |>
    dplyr::arrange(dplyr::desc(abs(spearman_r)), p_fdr)

  DBI::dbWriteTable(
    con,
    "cross_engine_metric_correlations",
    cor_results,
    overwrite = TRUE
  )

  readr::write_csv(
    cor_results,
    file.path(output_dir, "cross_engine_metric_correlations.csv")
  )

  message("Wrote table: cross_engine_metric_correlations")

  invisible(cor_results)
}
