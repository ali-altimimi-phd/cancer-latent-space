# ==============================================================================
# File: R/structural_concordance/compute_quadrant_stability.R
# Purpose: Summarize quadrant reproducibility across chips and filter regimes
# Project: Global Cancer Complexity / Structural Inference
# ------------------------------------------------------------------------------
# Notes:
#   - This table evaluates whether a comparison is assigned to the same quadrant
#     across chip/filter-regime realizations.
#   - It is comparison-level and should be interpreted as structural
#     reproducibility, not biological enrichment.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
})

compute_quadrant_stability <- function(quadrant_tbl,
                                       logger = NULL) {
  require_columns(
    quadrant_tbl,
    c("chip", "filter_regime", "group", "comparison", "quadrant"),
    table_label = "quadrant assignment table"
  )

  sc_log("🧪 Computing quadrant stability across chips and filter regimes...", logger = logger)

  quadrant_tbl |>
    filter(!is.na(quadrant), quadrant != "boundary") |>
    group_by(group, comparison) |>
    summarise(
      n_assignments = n(),
      n_chips = n_distinct(chip),
      n_filter_regimes = n_distinct(filter_regime),
      chips_seen = paste(sort(unique(chip)), collapse = ";"),
      filter_regimes_seen = paste(sort(unique(filter_regime)), collapse = ";"),
      n_unique_quadrants = n_distinct(quadrant),
      modal_quadrant = majority_label(quadrant),
      modal_quadrant_fraction = majority_fraction(quadrant),
      stable_all_assignments = n_unique_quadrants == 1,
      high_confidence_fraction = mean(quadrant_confidence == "high", na.rm = TRUE),
      median_distance_from_boundary_z = stats::median(
        distance_from_boundary_z,
        na.rm = TRUE
      ),
      .groups = "drop"
    ) |>
    mutate(
      stability_class = case_when(
        stable_all_assignments & n_chips >= 2 & n_filter_regimes >= 2 ~
          "cross-chip/cross-filter stable",
        stable_all_assignments ~
          "stable among available assignments",
        modal_quadrant_fraction >= 0.67 ~
          "majority-stable",
        TRUE ~
          "unstable"
      )
    ) |>
    arrange(desc(modal_quadrant_fraction), comparison)
}
