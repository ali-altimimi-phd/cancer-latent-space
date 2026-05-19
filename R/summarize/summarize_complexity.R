#' Recalculate Complexity Summary Scores by Comparison and Mode
#'
#' Computes weighted complexity summary scores based on observed effect sizes and permutation p-values.
#' Expands GO mode into GO_BP and GO_MF using `go_ontology` column.
#'
#' @param complexity_df A data frame with complexity results, including:
#'   - `comparison`, `mode`, `δ svd κ`, `p_perm`, `direction`, and `go_ontology`
#' @return A tibble with one row per `comparison` × `mode` containing:
#'   - weighted complexity score,
#'   - count and proportion of significant gene sets,
#'   - directionality confidence labels
#' @export
recalculate_complexity_summary <- function(complexity_df) {
  library(dplyr)
  
  # Defensive checks
  required_cols <- c("comparison", "mode", "go_ontology", "p_perm", "δ svd κ", "direction")
  stopifnot(all(required_cols %in% colnames(complexity_df)))
  
  complexity_df %>%
    mutate(
      # Expand GO mode into GO_BP or GO_MF
      mode = case_when(
        mode == "GO" & go_ontology == "BP" ~ "GO_BP",
        mode == "GO" & go_ontology == "MF" ~ "GO_MF",
        mode == "GO" & is.na(go_ontology) ~ "GO_UNSPECIFIED",
        TRUE ~ mode
      ),
      # Direction sign: +1 for gain, -1 for loss, 0 otherwise
      direction_sign = case_when(
        direction == "gained" ~ 1,
        direction == "lost" ~ -1,
        TRUE ~ 0
      ),
      # Weight based on permutation p-value; stronger weight for p < 0.1
      base_weight = case_when(
        is.na(p_perm) ~ 0,
        p_perm < 0.1 ~ -log10(p_perm),
        TRUE ~ 1
      ),
      # Weighted score combines direction and strength
      weighted_score = direction_sign * base_weight * abs(`δ svd κ`)
    ) %>%
    group_by(comparison, mode) %>%
    summarise(
      complexity_weighted_score = sum(weighted_score, na.rm = TRUE),
      total_gene_sets = n(),
      significant_sets = sum(p_perm < 0.05, na.rm = TRUE),
      prop_significant = significant_sets / total_gene_sets,
      prop_direction_consistent = mean(direction_sign == sign(complexity_weighted_score), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      # Assign confidence labels based on proportion of significant and consistent sets
      confidence_label = case_when(
        prop_significant >= 0.1 & prop_direction_consistent >= 0.7 ~ "well supported",
        prop_significant >= 0.05 & prop_direction_consistent >= 0.5 ~ "moderately supported",
        TRUE ~ "uncertain"
      ),
      # Categorize strength of weighted score
      strength_label = case_when(
        complexity_weighted_score >= 5 ~ "strong gain",
        complexity_weighted_score >= 2 ~ "mild gain",
        complexity_weighted_score <= -5 ~ "strong loss",
        complexity_weighted_score <= -2 ~ "mild loss",
        TRUE ~ "no clear change"
      )
    )
}

