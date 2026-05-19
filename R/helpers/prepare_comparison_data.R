#' Prepare Comparison Data for Reporting
#'
#' Filters and packages all relevant data for a given comparison into a single `.rds` file,
#' including summaries, complexity/entropy results, and per-chip probe counts.
#'
#' This `.rds` file can be passed as a parameter to Quarto report templates.
#' A corresponding `.csv` summary is also written for optional debugging or review.
#'
#' @param comparison Comparison code (e.g., `"BR/BRAD"`)
#' @param summaries_combined_df Data frame of combined summary statements
#' @param complexity_df Data frame of complexity results
#' @param entropy_df Data frame of entropy results
#' @param res_hu35ksuba Result object for the hu35ksuba chip
#' @param res_hu6800 Result object for the hu6800 chip
#' @param output_dir Directory to write the resulting `.rds` and `.csv` files
#'
#' @return The full path to the generated `.rds` file (invisible)
#' @export
prepare_comparison_data <- function(comparison,
                                    summaries_combined_df,
                                    complexity_df,
                                    entropy_df,
                                    res_hu35ksuba,
                                    res_hu6800,
                                    output_dir = "output/global_cancer/RData/comparison_data") {
  # Load libraries (assumed to already be loaded in pipeline, but safe for standalone use)
  library(dplyr)
  library(readr)
  
  # Ensure output directory exists
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Clean comparison name for filenames
  clean_comp <- gsub("[^a-zA-Z0-9]", "_", tolower(comparison))
  
  # Filter summary data
  summaries_df_comp <- summaries_combined_df %>%
    filter(comparison == !!comparison)
  
  # Filter significant complexity records
  complexity_df_comp <- complexity_df %>%
    filter(comparison == !!comparison) %>%
    filter(p_perm <= 0.05 | (mode == "KEGG" & kegg_cancer == TRUE))
  
  # Filter significant entropy records
  entropy_df_comp <- entropy_df %>%
    filter(comparison == !!comparison) %>%
    filter(p_perm <= 0.05 | (mode == "KEGG" & kegg_cancer == TRUE))
  
  # Get probe counts per chip
  count_35k <- get_filtered_probe_count(res_hu35ksuba, comparison)
  count_6800 <- get_filtered_probe_count(res_hu6800, comparison)
  
  # Package all relevant data
  comp_data <- list(
    comparison = comparison,
    summaries_df = summaries_df_comp,
    complexity_df = complexity_df_comp,
    entropy_df = entropy_df_comp,
    count_35k = count_35k,
    count_6800 = count_6800
  )
  
  # Write to RDS and CSV
  rds_file <- file.path(output_dir, paste0("comparison_data_", clean_comp, ".rds"))
  csv_file <- file.path(output_dir, paste0("comparison_summary_", clean_comp, ".csv"))
  
  saveRDS(comp_data, rds_file)
  readr::write_csv(summaries_df_comp, csv_file)
  
  message("✓ Saved RDS: ", rds_file)
  message("✓ Saved CSV: ", csv_file)
  
  invisible(rds_file)
}
