# ------------------------------------------------------------------------------
# File: run_all_reports.R
# Purpose: Generate comparison-level Quarto source reports and supporting
#   reporting artifacts by iterating over biological contrasts and invoking
#   data preparation, plot generation, and table export helpers.
# Role: Pipeline coordinator (batch execution wrapper)
# Pipeline: Reporting
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Generate All Comparison-Level Quarto Report Sources
#'
#' This function prepares RDS data files, regenerates plots, writes
#' parameterized `.qmd` Quarto files, and delegates table writing to a helper
#' for each unique comparison in the data.
#'
#' Each generated `.qmd` file is placed in:
#' `quarto/reports/generated_reports/`, and can be included in the Quarto book.
#'
#' Plots are written to:
#' `quarto/resources/plots/`, organized by type (e.g., `probes/`, `go/`, `kegg/`)
#'
#' HTML tables are generated via `write_all_comparison_tables()`, located in:
#' `R/helpers/write_all_comparison_tables.R`
#'
#' @param summaries_combined_df Data frame containing summary results
#' @param complexity_df Data frame of complexity statistics
#' @param entropy_df Data frame of entropy statistics
#' @param res_hu35ksuba Results object for hu35ksuba chip
#' @param res_hu6800 Results object for hu6800 chip
#' @param template_path Path to the Quarto template `.qmd` file
#' @param rds_output_dir Directory to save RDS files (default: "output/global_cancer/RData")
#' @param plot_output_dir Directory to save plots (default: "quarto/resources/plots")
#' @param qmd_output_dir Directory to save generated `.qmd` files (default: "quarto/reports/generated_reports")
#'
#' @return NULL (invisible), used for side effects
run_all_reports <- function(summaries_combined_df,
                            complexity_df,
                            entropy_df,
                            res_hu35ksuba,
                            res_hu6800,
                            template_path = here::here("quarto/reports/comparison_report_template.qmd"),
                            rds_output_dir,
                            plot_output_dir = here::here("quarto/resources/plots"),
                            table_output_dir = here::here("quarto/resources/tables"),
                            qmd_output_dir = here::here("quarto/reports/generated_reports")) {
  library(glue)
  
  # Update directory location
  rds_output_dir <- file.path(rds_output_dir, "comparison_data")
  
  source(here::here("R/helpers/sanitize_comparison.R"))
  source(here::here("R/helpers/prepare_comparison_data.R"))
  source(here::here("R/helpers/probe_helpers.R"))
  source(here::here("R/visualizations/generate_comparison_plots.R"))
  source(here::here("R/helpers/write_all_comparison_tables.R"))
  source(here::here("R/tables/write_probe_count_tables_html.R"))
  
  args <- list(
    summaries_combined_df = summaries_combined_df,
    complexity_df = complexity_df,
    entropy_df = entropy_df,
    res_hu35ksuba = res_hu35ksuba,
    res_hu6800 = res_hu6800
  )
  
  missing <- names(args)[vapply(args, is.null, logical(1))]
  
  if (length(missing) > 0) {
    stop("❌ Missing required data objects: ",
         paste(missing, collapse = ", "))
  }
  
  dir.create(qmd_output_dir,
             recursive = TRUE,
             showWarnings = FALSE)
  dir.create(plot_output_dir,
             recursive = TRUE,
             showWarnings = FALSE)
  dir.create(table_output_dir,
             recursive = TRUE,
             showWarnings = FALSE)
  
  all_comparisons <- unique(summaries_combined_df$comparison)
  
  # ---- Write global probe-count summary tables ----
  # Probe-count tables summarize filtered probe availability across comparisons
  # and are therefore generated once, outside the per-comparison loop.
  write_probe_count_tables_html(
    res_hu35ksuba = res_hu35ksuba,
    res_hu6800 = res_hu6800,
    output_dir = "quarto/resources/tables/probes",
    data_dir = rds_output_dir
  )
  
  for (cmp in all_comparisons) {
    clean_cmp <- sanitize_comparison(cmp)
    message("→ Generating .qmd for: ", cmp)
    
    rds_path <- prepare_comparison_data(
      comparison            = cmp,
      summaries_combined_df = summaries_combined_df,
      complexity_df         = complexity_df,
      entropy_df            = entropy_df,
      res_hu35ksuba         = res_hu35ksuba,
      res_hu6800            = res_hu6800,
      output_dir            = rds_output_dir
    )
    
    generate_comparison_plots(
      comparison       = cmp,
      complexity_df    = complexity_df,
      entropy_df       = entropy_df,
      plot_utils_path  = "R/helpers/plot_utils.R",
      output_dir       = plot_output_dir
    )
    
    # Generate HTML tables
    write_all_comparison_tables(
      comparison    = cmp,
      complexity_df = complexity_df,
      entropy_df    = entropy_df,
      output_dir    = table_output_dir
    )
    
    # # Generate probe tables
    # write_probe_count_tables_html(
    #   res_hu35ksuba = res_hu35ksuba,
    #   res_hu6800 = res_hu6800,
    #   output_dir = "quarto/resources/tables/probes",
    #   data_dir = data_dir
    # )
    
    Sys.setenv(TMPDIR = normalizePath(rds_output_dir, mustWork = TRUE))
    
    template_lines <- readLines(template_path)
    header_end <- which(trimws(template_lines) == "---")[2]
    if (is.na(header_end)) {
      stop("❌ Could not find the second YAML delimiter (---) in the template.")
    }
    
    param_yaml <- glue(
      "---\n",
      "title: \"Comparison Report: {cmp}\"\n",
      "author: \"Ali M. Al-Timimi, PhD\"\n",
      "format: html\n",
      "params:\n",
      "  data_file: \"{normalizePath(rds_path, winslash = '/') }\"\n",
      "execute:\n",
      "  eval: true\n",
      "---"
    )
    
    new_lines <- c(param_yaml, template_lines[(header_end + 1):length(template_lines)])
    
    output_qmd <- file.path(qmd_output_dir,
                            glue("comparison_report_{clean_cmp}.qmd"))
    writeLines(new_lines, output_qmd)
    message("✔ Wrote ", output_qmd, "\n")
  }
  
  invisible(NULL)
}
