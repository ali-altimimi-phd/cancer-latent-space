# ==============================================================================
# File: R/structural_concordance/compute_engine_correlations.R
# Purpose: Compute cross-engine structural metric concordance
# Project: Global Cancer Complexity / Structural Inference
# ------------------------------------------------------------------------------
# Notes:
#   - Operates on the canonical concordance table.
#   - Produces pairwise metric correlations across available chip/filter/comparison
#     rows. Spearman correlation is the default because structural descriptors may
#     have different scales and non-Gaussian distributions.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
})

infer_metric_engine <- function(metric_name) {
  dplyr::case_when(
    grepl("^complexity__", metric_name) ~ "complexity",
    grepl("^entropy__", metric_name) ~ "entropy",
    grepl("^mp__", metric_name) ~ "mp",
    grepl("^latent__", metric_name) ~ "latent",
    TRUE ~ "unknown"
  )
}

compute_engine_correlations <- function(canonical_tbl,
                                        metric_cols = NULL,
                                        method = "spearman",
                                        min_complete = 5,
                                        logger = NULL) {
  id_cols <- c("chip", "filter_regime", "group", "comparison")

  if (is.null(metric_cols)) {
    metric_cols <- names(canonical_tbl)[
      grepl("^(complexity|entropy|mp|latent)__", names(canonical_tbl))
    ]
  }

  metric_cols <- optional_numeric_cols(canonical_tbl, metric_cols)

  if (length(metric_cols) < 2) {
    sc_log("⚠️ Fewer than two numeric structural metrics available.", logger = logger)
    return(tibble())
  }

  sc_log("🔗 Computing pairwise structural metric correlations...", logger = logger)

  pairs <- utils::combn(metric_cols, 2, simplify = FALSE)

  purrr::map_dfr(
    pairs,
    function(pair_i) {
      x_name <- pair_i[[1]]
      y_name <- pair_i[[2]]
      x <- canonical_tbl[[x_name]]
      y <- canonical_tbl[[y_name]]
      ok <- stats::complete.cases(x, y)
      n_complete <- sum(ok)

      if (n_complete < min_complete) {
        return(tibble(
          metric_x = x_name,
          metric_y = y_name,
          engine_x = infer_metric_engine(x_name),
          engine_y = infer_metric_engine(y_name),
          method = method,
          n_complete = n_complete,
          estimate = NA_real_,
          p_value = NA_real_
        ))
      }

      test <- suppressWarnings(stats::cor.test(x[ok], y[ok], method = method))

      tibble(
        metric_x = x_name,
        metric_y = y_name,
        engine_x = infer_metric_engine(x_name),
        engine_y = infer_metric_engine(y_name),
        method = method,
        n_complete = n_complete,
        estimate = unname(test$estimate),
        p_value = test$p.value
      )
    }
  ) |>
    mutate(
      p_fdr = stats::p.adjust(p_value, method = "BH"),
      abs_estimate = abs(estimate),
      engine_pair = paste(engine_x, engine_y, sep = "__")
    ) |>
    arrange(desc(abs_estimate), p_fdr)
}
