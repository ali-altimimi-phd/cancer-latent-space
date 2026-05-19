# ------------------------------------------------------------------------------
# File: 03_run_report_pipeline.R
# Purpose: Execute the reporting pipeline by orchestrating summary exports,
#   optional per-comparison report generation, optional GO clustering outputs,
#   and optional Quarto rendering from aggregated analysis results.
# Role: Pipeline driver (entry point)
# Pipeline: Reporting
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Run Global Cancer Reporting Pipeline
#'
#' Orchestrates report-stage outputs from analysis-stage products, including
#' summary exports, optional per-comparison report generation, optional GO
#' clustering outputs, and optional Quarto rendering.
#'
#' @details
#' This function assumes that the analysis pipeline has already produced cleaned
#' pairwise comparison results and, when enabled, summary and filtered-probe
#' objects required for downstream reporting artifacts.
#'
#' @note Requires datasets created by the analysis pipeline.
#' @seealso \code{R/config/global_cancer/report_config.R}
#' @return Invisibly returns \code{NULL}; writes reporting outputs to disk.
run_report_pipeline <- function() {
  # ---- Config ----
  source(here::here("R/config/global_cancer/report_config.R"),
         local = FALSE)
  
  # ---- Logging ----
  source(here::here("R/helpers/pipeline_logger.R"))
  logger <- start_log(reports_pipeline_logfile)
  logger$log("🚀 Starting Global Cancer report pipeline")
  
  # ---- Stage 1: Load required datasets ----
  source(here::here("R/helpers/load_results_into_global.R"))
  
  if (isTRUE(run_summary_exports) ||
      isTRUE(run_comparison_qmd_generation) ||
      isTRUE(run_go_clustering)) {
    logger$log("🔀 Loading cleaned pairwise comparison results...")
    load_cleaned_results(aggregate_dir, overwrite = TRUE)
  }
  
  if (isTRUE(run_comparison_qmd_generation) ||
      isTRUE(run_go_clustering)) {
    logger$log("🔀 Loading summary results...")
    load_summaries_df(summaries_dir, overwrite = TRUE)
  }
  
  if (isTRUE(run_comparison_qmd_generation)) {
    logger$log("🔀 Loading filtered probes...")
    load_filtered_results(filtered_probes_dir, overwrite = TRUE)
  }
  
  # ---- Validate loaded datasets ----
  validate_reporting_inputs <- function() {
    
    required_objects <- c()
    
    if (isTRUE(run_summary_exports) || isTRUE(run_go_clustering)) {
      required_objects <- c(required_objects, "entropy_df", "complexity_df")
    }
    
    if (isTRUE(run_comparison_qmd_generation) || isTRUE(run_go_clustering)) {
      required_objects <- c(required_objects, "summaries_combined_df")
    }
    
    if (isTRUE(run_comparison_qmd_generation)) {
      required_objects <- c(required_objects, "res_hu35ksuba", "res_hu6800")
    }
    
    missing <- required_objects[
      !sapply(required_objects, exists, envir = .GlobalEnv)
    ]
    
    if (length(missing) > 0) {
      stop(sprintf(
        "Missing required reporting object(s): %s",
        paste(missing, collapse = ", ")
      ))
    }
  }
  
  validate_reporting_inputs()

  # ---- Stage 2: Create summary exports ----
  if (isTRUE(run_summary_exports)) {
    logger$log("📄 Generating summary exports...")
    source(here::here("R/reports/summarize_pairwise_results.R"))
    
    entropy_report    <- summarize_pairwise_results(entropy_df, engine = "entropy")
    complexity_report <- summarize_pairwise_results(complexity_df, engine = "complexity")
    
    if (!dir.exists(reports_dir)) {
      stop("Reports directory does not exist: ", reports_dir)
    }
    
    test_file <- file.path(reports_dir, ".write_test")
    ok <- tryCatch({
      writeLines("test", test_file)
      unlink(test_file)
      TRUE
    }, error = function(e) FALSE)
    
    if (!ok) {
      stop("Reports directory is not writable: ", reports_dir)
    }
    
    readr::write_lines(entropy_report,
                       file.path(reports_dir, "entropy_summary.txt"))
    readr::write_lines(complexity_report,
                       file.path(reports_dir, "complexity_summary.txt"))
    
    readr::write_csv(entropy_df,
                     file.path(reports_dir, "entropy_cleaned.csv"))
    readr::write_csv(complexity_df,
                     file.path(reports_dir, "complexity_cleaned.csv"))
    
    saveRDS(entropy_df,
            file.path(reports_dir, "entropy_cleaned.rds"))
    saveRDS(complexity_df,
            file.path(reports_dir, "complexity_cleaned.rds"))
    
    if (isTRUE(run_pairwise_summary_html)) {
      logger$log("⚠️ Rendering legacy pairwise summary report...")
      rmarkdown::render(
        input = here::here("R/legacy/reports/pairwise_summary_report.Rmd"),
        output_file = "pairwise_summary_report.html",
        output_dir = reports_dir
      )
    }
    
    logger$log("✅ Summary exports saved.")
  }

  # ---- Stage 3: GO semantic summarization ----
  if (isTRUE(run_go_clustering)) {
    logger$log(sprintf("🧬 Running GO semantic summarization... (mode = %s)", go_mode))
    
    source(here::here("R/wrappers/run_go_clustering.R"))
    
    run_go_clustering_main(
      summaries_combined_df = summaries_combined_df,
      complexity_df = complexity_df,
      entropy_df = entropy_df,
      output_dir_base = here::here("quarto", "resources", "tables", "clusters"),
      go_mode = go_mode,
      similarity_cutoff = go_similarity_cutoff,
      logger = logger
    )   
    
    logger$log("✅ GO semantic summarization complete.")
  }
  
  # ---- Stage 4: Generate comparison-level QMD reports ----
  
  if (isTRUE(run_comparison_qmd_generation)) {
    source(here::here("R/wrappers/run_all_reports.R"))
    
    run_all_reports(
      summaries_combined_df = summaries_combined_df,
      complexity_df = complexity_df,
      entropy_df = entropy_df,
      res_hu35ksuba = res_hu35ksuba,
      res_hu6800 = res_hu6800,
      rds_output_dir = data_dir
    )
  }
  
  # ---- Stage 5: Render comparison reports from .qmd template ----
  if (run_quarto) {
    old <- getwd()
    setwd(here::here("quarto"))
    unlink("_freeze", recursive = TRUE, force = TRUE)
    quarto::quarto_render(".")
    setwd(old)
  }

  # ---- Final message ----
  logger$log("🎉 Report pipeline completed successfully.")
}

if (sys.nframe() == 0) {
  run_report_pipeline()
}
