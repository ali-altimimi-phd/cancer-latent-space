# ------------------------------------------------------------------------------
# File: 03_run_biological_enrichment_pipeline.R
# Purpose: Execute the biological enrichment pipeline
# Role: Top-level biological enrichment runner
# Pipeline: Biological_enrichment
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------
NULL
#' Run Biological enrichment Pipeline for Global Cancer Microarray Data
#'
#'
#' This script is intended to be run **after** the global-cancer-complexity structural_inference pipeline.

run_biological_enrichment_pipeline <- function() {
  # ---- Stage 1: Configuration + Logging ----
  source(here::here("R/config/global_cancer/biological_enrichment_config.R"), local = FALSE)
  
  source(here::here("R/helpers/pipeline_logger.R"))
  logger <- start_log(biological_enrichment_logfile)
  logger$log("🚀 Starting Global Cancer biological enrichment pipeline...")
  
  # ---- Stage 2: Load existing structural-inference inputs ----
  
  logger$log(
    "📦 Loading structural-inference input artifacts...",
    section = "INPUTS"
  )
  
  source(load_pipeline_inputs_helper)
  
  load_pipeline_inputs(
    matrices_path = matrices_path,
    filtered_probes_dir = structural_filtered_probes_dir,
    chips = chips,
    filter_regimes = filter_regimes,
    require_matrix_maps = TRUE,
    require_filtered_probes = TRUE,
    logger = logger,
    overwrite = TRUE
  )
  
  logger$log(
    "✅ Structural-inference input artifacts successfully loaded.",
    section = "INPUTS"
  )
  
  
  # ---- Stage 3: Load biological annotations ----
  
  logger$log(
    "🧬 Loading biological annotation resources...",
    section = "ANNOTATIONS"
  )
  
  source(load_biological_annotations_helper)
  
  load_biological_annotations(
    annotations_path = annotations_path,
    logger = logger,
    overwrite = TRUE
  )
  
  logger$log(
    "✅ Biological annotation resources successfully loaded.",
    section = "ANNOTATIONS"
  )
  
  # ---- Stage 4: Pairwise comparisons ----
  
  if (isTRUE(run_pairwise)) {
    
    logger$log(
      "🔀 Starting biological pairwise gene-set comparisons...",
      section = "PAIRWISE"
    )
    
    source(here::here("R/wrappers/run_biological_pairwise_comparisons.R"))
    
    biological_pairwise_results <- run_biological_pairwise_comparisons(
      chips = chips,
      annotations = annotations,
      engines = biological_engines,
      gene_set_modes = gene_set_modes,
      filter_regimes = filter_regimes,
      min_gene_set_probes = min_gene_set_probes,
      filtered_probes_dir = structural_filtered_probes_dir,
      output_dir = biological_enrichment_rdata_dir,
      
      pairwise_logs_dir = file.path(
        biological_enrichment_logs_dir,
        "pairwise_detail"
      ),
      
      biological_resampling_seed = biological_resampling_seed,
      
      run_complexity_permutation = run_complexity_permutation,
      complexity_n_perm = complexity_n_perm,
      complexity_permutation_metric = complexity_permutation_metric,
      complexity_permutation_unit = complexity_permutation_unit,
      
      run_complexity_bootstrap = run_complexity_bootstrap,
      complexity_n_boot = complexity_n_boot,
      complexity_bootstrap_metric = complexity_bootstrap_metric,
      complexity_bootstrap_unit = complexity_bootstrap_unit,
      complexity_covariance_space = complexity_covariance_space,
      
      run_entropy_permutation = run_entropy_permutation,
      entropy_n_perm = entropy_n_perm,
      entropy_permutation_metric = entropy_permutation_metric,
      entropy_permutation_unit = entropy_permutation_unit,
      
      run_entropy_bootstrap = run_entropy_bootstrap,
      entropy_n_boot = entropy_n_boot,
      entropy_bootstrap_metric = entropy_bootstrap_metric,
      entropy_bootstrap_unit = entropy_bootstrap_unit,
      entropy_covariance_space = entropy_covariance_space,
      
      logger = logger
    )
    
    saveRDS(
      biological_pairwise_results,
      file.path(
        biological_enrichment_rdata_dir,
        "biological_pairwise_results_all.rds"
      )
    )
    
    logger$log(
      "✅ Biological pairwise gene-set comparisons completed.",
      section = "PAIRWISE"
    )
    
  } else {
    
    logger$log(
      "⏭️ Skipping pairwise biological comparisons.",
      section = "PAIRWISE"
    )
  }
  
  # ---- Stage 5: Aggregation + gene set annotation ----
  if (run_aggregator) {
    logger$log("🔀 Checking for pairwise comparison results...")
    source(here::here("R/helpers/load_results_into_global.R"))
    load_comparison_results(data_dir, overwrite = TRUE)
    
    logger$log("📦 Ensuring annotations are loaded...")
    source(here::here("R/helpers/load_annotations.R"))
    load_annotations_if_needed(logger = logger)
    
    logger$log("🔀 Starting results aggregation...")
    source(here::here("R/aggregate/aggregate_engine_results_by_engine.R"))
    source(here::here("R/helpers/gene_set_tools.R"))
    source(here::here("R/helpers/clean_aggregated_results.R"))
    
    agg_complexity <<- aggregate_engine_results_by_engine("complexity")
    agg_entropy    <<- aggregate_engine_results_by_engine("entropy")
    
    agg_complexity <<- attach_gene_set_names(agg_complexity, annotations)
    agg_entropy    <<- attach_gene_set_names(agg_entropy, annotations)
    
    complexity_df <<- clean_aggregated_results(agg_complexity)
    entropy_df    <<- clean_aggregated_results(agg_entropy)
    
    readr::write_csv(complexity_df,
                     file.path(aggregate_dir, "complexity_aggregated_results.csv"))
    readr::write_csv(entropy_df,
                     file.path(aggregate_dir, "entropy_aggregated_results.csv"))
    
    saveRDS(complexity_df,
            file.path(aggregate_dir, "complexity_aggregated_results.rds"))
    saveRDS(entropy_df,
            file.path(aggregate_dir, "entropy_aggregated_results.rds"))
    
    logger$log("✅ Aggregated complexity/entropy results saved.")
  }
  
  # ---- Stage 7: Summarize comparison-level patterns ----
  if (run_comparison_summary) {
    logger$log("🔀 Checking for cleaned pairwise comparison results...")
    source(here::here("R/helpers/load_results_into_global.R"))
    load_cleaned_results(aggregate_dir, overwrite = TRUE)
    
    logger$log("📊 Summarizing entropy + complexity across comparisons...")
    
    source("R/summarize/summarize_complexity.R")
    source("R/summarize/summarize_entropy.R")
    source("R/summarize/combine_entropy_complexity_summaries.R")
    
    summary_complexity_df <<- recalculate_complexity_summary(complexity_df)
    summary_entropy_df    <<- recalculate_entropy_summary(entropy_df)
    
    summaries_combined_df <<- combine_entropy_complexity_summaries(
      summary_complexity_df, summary_entropy_df
    )
    
    saveRDS(summaries_combined_df,
            file.path(summaries_dir, "summaries_combined_df.rds"))
    readr::write_csv(summaries_combined_df,
                     file.path(summaries_dir, "summaries_combined_df.csv"))
    
    logger$log("✅ Comparison-level summary table saved.")
  }

  # ---- Final message ----
  logger$log("🎉 Biological enrichment pipeline completed successfully.", section = "PIPELINE")
}

if (sys.nframe() == 0) {
  run_biological_enrichment_pipeline()
}
