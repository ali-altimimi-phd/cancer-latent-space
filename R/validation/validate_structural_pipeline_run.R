# ==============================================================================
# File: validate_structural_pipeline_run.R
# Purpose: Validate structural inference pipeline outputs
# Role: Post-run validation for structural, latent, synthesis, and DuckDB outputs
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ==============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(dplyr)
  library(readr)
  library(tibble)
})

# ------------------------------------------------------------------------------
# Main validator
# ------------------------------------------------------------------------------

validate_structural_pipeline_run <- function(
    structural_rdata_dir,
    structural_tables_dir,
    structural_synthesis_table_dir,
    warehouse_db_path,
    run_mp_engine = TRUE,
    run_complexity_engine = TRUE,
    run_entropy_engine = TRUE,
    run_latent_engine = FALSE,
    build_structural_phenotype_table = TRUE,
    ingest_structural_results_to_duckdb = FALSE,
    latent_tables_dir = file.path(structural_tables_dir, "latent"),
    latent_chip_id = "hu35ksuba",
    latent_filter_regime = "variance_global",
    logger = NULL
) {
  
  log_msg <- function(..., section = "VALIDATION") {
    msg <- paste0(...)
    if (!is.null(logger) && is.function(logger$log)) {
      logger$log(msg, section = section)
    } else {
      message(msg)
    }
  }
  
  fail <- function(...) {
    stop(paste0(...), call. = FALSE)
  }
  
  check_file <- function(path, label) {
    if (!file.exists(path)) {
      fail("Missing required ", label, ": ", path)
    }
    log_msg("✅ Found ", label, ": ", path)
    invisible(TRUE)
  }
  
  log_msg("🔎 Validating structural pipeline outputs...")
  
  # ---------------------------------------------------------------------------
  # Engine RDS outputs
  # ---------------------------------------------------------------------------
  
  if (isTRUE(run_complexity_engine)) {
    check_file(
      file.path(structural_rdata_dir, "complexity_results.rds"),
      "complexity results RDS"
    )
  }
  
  if (isTRUE(run_entropy_engine)) {
    check_file(
      file.path(structural_rdata_dir, "entropy_results.rds"),
      "entropy results RDS"
    )
  }
  
  if (isTRUE(run_mp_engine)) {
    check_file(
      file.path(structural_rdata_dir, "mp_spectral_results.rds"),
      "MP spectral results RDS"
    )
  }
  
  # ---------------------------------------------------------------------------
  # Latent engine outputs
  # ---------------------------------------------------------------------------
  
  if (isTRUE(run_latent_engine)) {
    
    latent_comparison_metrics_csv <- file.path(
      latent_tables_dir,
      "latent_comparison_metrics.csv"
    )
    
    latent_sample_coordinates_csv <- file.path(
      latent_tables_dir,
      "latent_sample_coordinates.csv"
    )
    
    vae_training_summary_csv <- file.path(
      latent_tables_dir,
      "vae_training_summary.csv"
    )
    
    check_file(
      latent_comparison_metrics_csv,
      "latent comparison metrics CSV"
    )
    
    check_file(
      latent_sample_coordinates_csv,
      "latent sample coordinates CSV"
    )
    
    check_file(
      vae_training_summary_csv,
      "VAE training summary CSV"
    )
    
    latent_metrics <- readr::read_csv(
      latent_comparison_metrics_csv,
      show_col_types = FALSE
    )
    
    required_latent_cols <- c(
      "chip",
      "filter_regime",
      "comparison",
      "group",
      "latent_model_id"
    )
    
    missing_latent_cols <- setdiff(
      required_latent_cols,
      names(latent_metrics)
    )
    
    if (length(missing_latent_cols) > 0) {
      fail(
        "Latent comparison metrics are missing required columns: ",
        paste(missing_latent_cols, collapse = ", ")
      )
    }
    
    if (!latent_chip_id %in% unique(latent_metrics$chip)) {
      fail(
        "Expected latent chip not found in latent metrics: ",
        latent_chip_id
      )
    }
    
    if (!latent_filter_regime %in% unique(latent_metrics$filter_regime)) {
      fail(
        "Expected latent filter regime not found in latent metrics: ",
        latent_filter_regime
      )
    }
    
    if (!"centroid_distance" %in% names(latent_metrics)) {
      fail("Latent comparison metrics are missing centroid_distance.")
    }
    
    log_msg(
      "✅ Latent metrics validated: ",
      nrow(latent_metrics),
      " rows; chip=",
      latent_chip_id,
      "; filter_regime=",
      latent_filter_regime
    )
  }
  
  # ---------------------------------------------------------------------------
  # Synthesis outputs
  # ---------------------------------------------------------------------------
  
  if (isTRUE(build_structural_phenotype_table)) {
    
    synthesis_files <- c(
      "structural_phenotype_long.csv",
      "structural_phenotype_wide.csv",
      "structural_phenotype_heatmap_long.csv",
      "structural_phenotype_summary.csv"
    )
    
    for (nm in synthesis_files) {
      check_file(
        file.path(structural_synthesis_table_dir, nm),
        paste("synthesis table", nm)
      )
    }
    
    phenotype_long <- readr::read_csv(
      file.path(
        structural_synthesis_table_dir,
        "structural_phenotype_long.csv"
      ),
      show_col_types = FALSE
    )
    
    if (isTRUE(run_latent_engine)) {
      latent_rows <- phenotype_long |>
        dplyr::filter(
          .data$engine == "latent",
          .data$chip == latent_chip_id,
          .data$filter_regime == latent_filter_regime
        )
      
      if (nrow(latent_rows) == 0) {
        fail(
          "Synthesis long table contains no latent rows for chip=",
          latent_chip_id,
          ", filter_regime=",
          latent_filter_regime
        )
      }
      
      log_msg(
        "✅ Latent rows present in synthesis long table: ",
        nrow(latent_rows)
      )
    }
  }
  
  # ---------------------------------------------------------------------------
  # DuckDB warehouse outputs
  # ---------------------------------------------------------------------------
  
  if (isTRUE(ingest_structural_results_to_duckdb)) {
    
    check_file(
      warehouse_db_path,
      "DuckDB warehouse"
    )
    
    con <- DBI::dbConnect(
      duckdb::duckdb(),
      dbdir = warehouse_db_path,
      read_only = TRUE
    )
    
    on.exit({
      DBI::dbDisconnect(con, shutdown = TRUE)
    }, add = TRUE)
    
    existing_tables <- DBI::dbGetQuery(
      con,
      "
      SELECT table_name
      FROM information_schema.tables
      WHERE table_schema = 'main'
      "
    )$table_name
    
    expected_tables <- c()
    
    if (isTRUE(run_complexity_engine)) {
      expected_tables <- c(
        expected_tables,
        "structural_complexity_results_fact"
      )
    }
    
    if (isTRUE(run_entropy_engine)) {
      expected_tables <- c(
        expected_tables,
        "structural_entropy_results_fact"
      )
    }
    
    if (isTRUE(run_mp_engine)) {
      expected_tables <- c(
        expected_tables,
        "structural_mp_spectral_results_fact",
        "structural_mp_spectral_deltas_fact"
      )
    }
    
    if (isTRUE(run_latent_engine)) {
      expected_tables <- c(
        expected_tables,
        "latent_comparison_metrics_fact",
        "latent_sample_coordinates_fact",
        "vae_training_summary_fact"
      )
    }
    
    missing_tables <- setdiff(expected_tables, existing_tables)
    
    if (length(missing_tables) > 0) {
      fail(
        "DuckDB warehouse is missing expected tables: ",
        paste(missing_tables, collapse = ", ")
      )
    }
    
    log_msg(
      "✅ DuckDB expected tables present: ",
      paste(expected_tables, collapse = ", ")
    )
    
    if (isTRUE(run_latent_engine)) {
      latent_db_check <- DBI::dbGetQuery(
        con,
        sprintf(
          "
          SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT comparison) AS n_comparisons
          FROM latent_comparison_metrics_fact
          WHERE chip = '%s'
            AND filter_regime = '%s'
          ",
          latent_chip_id,
          latent_filter_regime
        )
      )
      
      if (latent_db_check$n_rows[[1]] == 0) {
        fail(
          "DuckDB latent_comparison_metrics_fact contains no rows for chip=",
          latent_chip_id,
          ", filter_regime=",
          latent_filter_regime
        )
      }
      
      log_msg(
        "✅ DuckDB latent metrics validated: ",
        latent_db_check$n_rows[[1]],
        " rows; ",
        latent_db_check$n_comparisons[[1]],
        " comparisons"
      )
    }
  }
  
  log_msg("✅ Structural pipeline validation completed successfully.")
  
  invisible(TRUE)
}