#' Combine Complexity and Entropy Summary Data Frames with Interpretation
#'
#' Merges complexity and entropy summaries into a unified table with a natural-language summary.
#'
#' @param summary_complexity_df Data frame from `recalculate_complexity_summary()`
#' @param summary_entropy_df Data frame from `recalculate_entropy_summary()`
#' @return Tibble with combined metrics, interpretation, and cancer category
#' @export
combine_entropy_complexity_summaries <- function(summary_complexity_df,
                                                 summary_entropy_df) {
  library(dplyr)
  library(tidyr)
  library(glue)
  library(stringr)
  library(readr)
  
  source("R/helpers/category_utils.R")  # Make sure this helper is sourced
  
  # Clean entropy types
  summary_entropy_df_clean <- summary_entropy_df %>%
    mutate(type = str_remove(type, "_direction"))
  
  entropy_selected <- summary_entropy_df_clean %>%
    select(comparison, mode, type,
           entropy_weighted_score,
           strength_label,
           confidence_label)
  
  entropy_wide <- entropy_selected %>%
    pivot_wider(
      names_from = type,
      values_from = c(entropy_weighted_score, strength_label, confidence_label),
      names_sep = "_"
    )
  
  combined <- summary_complexity_df %>%
    left_join(entropy_wide, by = c("comparison", "mode")) %>%
    mutate(across(
      c(strength_label_shannon, confidence_label_shannon,
        strength_label_spectral, confidence_label_spectral),
      ~ ifelse(is.na(.x), "unknown", .x)
    )) %>%
    rename(
      complexity_strength_label = strength_label,
      complexity_confidence_label = confidence_label
    ) %>%
    mutate(
      mode_label = case_when(
        mode == "FULL" ~ "all probes",
        mode == "GO" ~ "GO MF terms",
        mode == "KEGG" ~ "KEGG pathways",
        mode == "MSIG" ~ "Hallmark genes",
        TRUE ~ mode
      )
    ) %>%
    add_cancer_category() %>%
    mutate(
      summary_statement = glue(
        "In comparison {comparison} ({mode_label}), complexity is {complexity_strength_label} ",
        "with {complexity_confidence_label} support. ",
        "Shannon entropy is {strength_label_shannon} ({confidence_label_shannon}), ",
        "and spectral entropy is {strength_label_spectral} ({confidence_label_spectral})."
      )
    )
  
  # Optional: export structured summaries for downstream 
  # AI-assisted interpretation. This is not part of the core analytical pipeline.
  # Save ChatGPT input for review
  chatgpt_dir <- here::here("output", "global_cancer", "chatgpt")
  dir.create(chatgpt_dir, recursive = TRUE, showWarnings = FALSE)
  
  combined %>%
    select(cancer_category, summary_statement) %>%
    write_csv(here::here("output/global_cancer/chatgpt/chatgpt_input.txt"))
  
  return(combined)
}
