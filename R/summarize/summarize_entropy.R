#' Recalculate Entropy Summary Scores by Comparison and Mode
#'
#' Computes weighted entropy summary scores based on directionality and permutation p-values
#' for both Shannon and spectral entropy metrics. Summarizes net signal strength, consistency,
#' and support across gene sets.
#'
#' @param entropy_df Data frame containing entropy results with columns:
#'   - `comparison`
#'   - `mode`
#'   - `gene_set_name`
#'   - `go_ontology` (optional; used to split GO into GO_BP, GO_MF, etc.)
#'   - `p_perm`
#'   - `shannon_direction`
#'   - `spectral_direction`
#'   - `shannon_delta`
#'   - `spectral_delta`
#'
#' @return A tibble with one row per `comparison` × `mode` × `entropy type`, with:
#'   - weighted entropy score,
#'   - count and proportion of significant gene sets,
#'   - directionality consistency,
#'   - and confidence + strength labels.
#'
#' @export
recalculate_entropy_summary <- function(entropy_df) {
  library(dplyr)
  library(tidyr)
  
  # Defensive column check
  required_cols <- c(
    "comparison", "mode", "gene_set_name", "go_ontology",
    "p_perm", "shannon_direction", "spectral_direction",
    "shannon_delta", "spectral_delta"
  )
  stopifnot(all(required_cols %in% colnames(entropy_df)))
  
  # Normalize GO modes before pivoting
  entropy_df <- entropy_df %>%
    mutate(
      mode = case_when(
        mode == "GO" & go_ontology == "BP" ~ "GO_BP",
        mode == "GO" & go_ontology == "MF" ~ "GO_MF",
        mode == "GO" & is.na(go_ontology) ~ "GO_UNSPECIFIED",
        TRUE ~ mode
      )
    )
  
  # Pivot to long format by entropy type
  entropy_long <- entropy_df %>%
    pivot_longer(
      cols = c(shannon_direction, spectral_direction),
      names_to = "type",
      values_to = "direction"
    ) %>%
    mutate(
      delta = case_when(
        type == "shannon_direction" ~ shannon_delta,
        type == "spectral_direction" ~ spectral_delta,
        TRUE ~ NA_real_
      ),
      direction_sign = case_when(
        grepl("anti-chaotic", direction) ~ -1,
        grepl("chaotic", direction) ~ 1,
        TRUE ~ 0
      ),
      base_weight = case_when(
        is.na(p_perm) ~ 0,
        p_perm < 0.1 ~ -log10(p_perm),
        TRUE ~ 1
      ),
      weighted_score = direction_sign * base_weight * abs(delta)
    )
  
  # Summarize by comparison × mode × entropy type
  entropy_summary <- entropy_long %>%
    group_by(comparison, mode, type) %>%
    summarise(
      entropy_weighted_score = sum(weighted_score, na.rm = TRUE),
      total_gene_sets = n(),
      significant_sets = sum(p_perm < 0.05, na.rm = TRUE),
      prop_significant = significant_sets / total_gene_sets,
      prop_direction_consistent = mean(direction_sign == sign(entropy_weighted_score), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      confidence_label = case_when(
        prop_significant >= 0.1 & prop_direction_consistent >= 0.7 ~ "well supported",
        prop_significant >= 0.05 & prop_direction_consistent >= 0.5 ~ "moderately supported",
        TRUE ~ "uncertain"
      ),
      strength_label = case_when(
        entropy_weighted_score >= 5 ~ "strongly chaotic",
        entropy_weighted_score >= 2 ~ "mildly chaotic",
        entropy_weighted_score <= -5 ~ "strongly anti-chaotic",
        entropy_weighted_score <= -2 ~ "mildly anti-chaotic",
        TRUE ~ "no clear change"
      )
    )
  
  return(entropy_summary)
}
