# ==============================================================================
# File: ingest_structural_results_to_duckdb.R
# Purpose: Ingest structural inference results into DuckDB
# Role: Structural warehouse ingestor
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ==============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(dplyr)
  library(tibble)
  library(fs)
  library(readr)
  library(here)
})

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

ingest_structural_summary_results <- function(file_path, expected_engine = NULL) {
  x <- readRDS(file_path)
  
  if (!"summary" %in% names(x)) {
    stop("Structural RDS does not contain `summary`: ", file_path)
  }
  
  out <- x$summary |>
    dplyr::mutate(
      source_file    = fs::path_file(file_path),
      analysis_scope = "whole_filtered_chip",
      .before = 1
    )
  
  if (!is.null(expected_engine)) {
    bad_engine <- out |>
      dplyr::filter(.data$engine != expected_engine)
    
    if (nrow(bad_engine) > 0) {
      stop(
        "Unexpected engine values in ",
        fs::path_file(file_path),
        ". Expected engine: ",
        expected_engine
      )
    }
  }
  
  out
}

ingest_structural_mp_results <- function(file_path) {
  x <- readRDS(file_path)
  
  if (!"summary" %in% names(x)) {
    stop("MP spectral RDS does not contain `summary`: ", file_path)
  }
  
  if (!"deltas" %in% names(x)) {
    stop("MP spectral RDS does not contain `deltas`: ", file_path)
  }
  
  summary <- x$summary |>
    dplyr::mutate(
      source_file    = fs::path_file(file_path),
      analysis_scope = "whole_filtered_chip",
      engine         = "mp_spectral",
      gene_set_mode  = "FULL",
      gene_set_name  = "FULL",
      .before = 1
    )
  
  deltas <- x$deltas |>
    dplyr::mutate(
      source_file    = fs::path_file(file_path),
      analysis_scope = "whole_filtered_chip",
      engine         = "mp_spectral",
      gene_set_mode  = "FULL",
      gene_set_name  = "FULL",
      .before = 1
    )
  
  list(summary = summary, deltas = deltas)
}

write_fact_table <- function(con,
                             table_name,
                             data,
                             warehouse_dir,
                             write_csv_exports = TRUE) {
  DBI::dbWriteTable(
    con,
    table_name,
    data,
    overwrite = TRUE
  )
  
  if (isTRUE(write_csv_exports)) {
    readr::write_csv(
      data,
      file.path(warehouse_dir, paste0(table_name, ".csv"))
    )
  }
  
  invisible(TRUE)
}

# ------------------------------------------------------------------------------
# Main ingestion function
# ------------------------------------------------------------------------------

ingest_structural_results_to_duckdb <- function(
    study_name = "global_cancer",
    structural_output_dir = here::here(
      "output", study_name, "structural_inference"
    ),
    warehouse_dir = here::here(
      "output", study_name, "warehouse"
    ),
    db_path = file.path(
      warehouse_dir, "global_cancer_results.duckdb"
    ),
    write_csv_exports = TRUE,
    logger = NULL
) {
  
  log_msg <- function(..., section = "DUCKDB") {
    msg <- paste0(...)
    if (!is.null(logger) && is.function(logger$log)) {
      logger$log(msg, section = section)
    } else {
      message(msg)
    }
  }
  
  fs::dir_create(warehouse_dir)
  
  structural_rdata_dir <- file.path(structural_output_dir, "RData")
  structural_tables_dir <- file.path(structural_output_dir, "tables")
  latent_tables_dir <- file.path(structural_tables_dir, "latent")
  
  structural_complexity_file <- file.path(
    structural_rdata_dir,
    "complexity_results.rds"
  )
  
  structural_entropy_file <- file.path(
    structural_rdata_dir,
    "entropy_results.rds"
  )
  
  structural_mp_file <- file.path(
    structural_rdata_dir,
    "mp_spectral_results.rds"
  )
  
  latent_comparison_metrics_file <- file.path(
    latent_tables_dir,
    "latent_comparison_metrics.csv"
  )
  
  latent_sample_coordinates_file <- file.path(
    latent_tables_dir,
    "latent_sample_coordinates.csv"
  )
  
  latent_training_summary_file <- file.path(
    latent_tables_dir,
    "vae_training_summary.csv"
  )
  
  ingested_tables <- character()
  skipped_files <- character()
  
  row_counts <- tibble::tibble(
    table_name = character(),
    n_rows = integer()
  )
  
  log_msg("🦆 Opening DuckDB warehouse: ", db_path)
  
  con <- DBI::dbConnect(
    duckdb::duckdb(),
    dbdir = db_path,
    read_only = FALSE
  )
  
  on.exit({
    DBI::dbDisconnect(con, shutdown = TRUE)
  }, add = TRUE)
  
  # ---- Complexity -------------------------------------------------------------
  
  if (file.exists(structural_complexity_file)) {
    
    log_msg("📦 Ingesting structural complexity results...")
    
    structural_complexity_results_fact <- ingest_structural_summary_results(
      structural_complexity_file,
      expected_engine = "complexity"
    )
    
    write_fact_table(
      con = con,
      table_name = "structural_complexity_results_fact",
      data = structural_complexity_results_fact,
      warehouse_dir = warehouse_dir,
      write_csv_exports = write_csv_exports
    )
    
    ingested_tables <- c(
      ingested_tables,
      "structural_complexity_results_fact"
    )
    
    row_counts <- dplyr::bind_rows(
      row_counts,
      tibble::tibble(
        table_name = "structural_complexity_results_fact",
        n_rows = nrow(structural_complexity_results_fact)
      )
    )
    
    log_msg(
      "✅ Structural complexity rows: ",
      nrow(structural_complexity_results_fact)
    )
    
  } else {
    
    skipped_files <- c(skipped_files, structural_complexity_file)
    
    log_msg(
      "⏭️ Skipping complexity ingestion; file not found: ",
      structural_complexity_file
    )
  }
  
  # ---- Entropy ---------------------------------------------------------------
  
  if (file.exists(structural_entropy_file)) {
    
    log_msg("📦 Ingesting structural entropy results...")
    
    structural_entropy_results_fact <- ingest_structural_summary_results(
      structural_entropy_file,
      expected_engine = "entropy"
    )
    
    write_fact_table(
      con = con,
      table_name = "structural_entropy_results_fact",
      data = structural_entropy_results_fact,
      warehouse_dir = warehouse_dir,
      write_csv_exports = write_csv_exports
    )
    
    ingested_tables <- c(
      ingested_tables,
      "structural_entropy_results_fact"
    )
    
    row_counts <- dplyr::bind_rows(
      row_counts,
      tibble::tibble(
        table_name = "structural_entropy_results_fact",
        n_rows = nrow(structural_entropy_results_fact)
      )
    )
    
    log_msg(
      "✅ Structural entropy rows: ",
      nrow(structural_entropy_results_fact)
    )
    
  } else {
    
    skipped_files <- c(skipped_files, structural_entropy_file)
    
    log_msg(
      "⏭️ Skipping entropy ingestion; file not found: ",
      structural_entropy_file
    )
  }
  
  # ---- MP spectral -----------------------------------------------------------
  
  if (file.exists(structural_mp_file)) {
    
    log_msg("📦 Ingesting structural MP spectral results...")
    
    structural_mp <- ingest_structural_mp_results(structural_mp_file)
    
    structural_mp_spectral_results_fact <- structural_mp$summary
    structural_mp_spectral_deltas_fact  <- structural_mp$deltas
    
    write_fact_table(
      con = con,
      table_name = "structural_mp_spectral_results_fact",
      data = structural_mp_spectral_results_fact,
      warehouse_dir = warehouse_dir,
      write_csv_exports = write_csv_exports
    )
    
    write_fact_table(
      con = con,
      table_name = "structural_mp_spectral_deltas_fact",
      data = structural_mp_spectral_deltas_fact,
      warehouse_dir = warehouse_dir,
      write_csv_exports = write_csv_exports
    )
    
    ingested_tables <- c(
      ingested_tables,
      "structural_mp_spectral_results_fact",
      "structural_mp_spectral_deltas_fact"
    )
    
    row_counts <- dplyr::bind_rows(
      row_counts,
      tibble::tibble(
        table_name = "structural_mp_spectral_results_fact",
        n_rows = nrow(structural_mp_spectral_results_fact)
      ),
      tibble::tibble(
        table_name = "structural_mp_spectral_deltas_fact",
        n_rows = nrow(structural_mp_spectral_deltas_fact)
      )
    )
    
    log_msg(
      "✅ Structural MP condition-level rows: ",
      nrow(structural_mp_spectral_results_fact)
    )
    
    log_msg(
      "✅ Structural MP delta rows: ",
      nrow(structural_mp_spectral_deltas_fact)
    )
    
  } else {
    
    skipped_files <- c(skipped_files, structural_mp_file)
    
    log_msg(
      "⏭️ Skipping MP spectral ingestion; file not found: ",
      structural_mp_file
    )
  }
  
  # ---- Latent comparison metrics ---------------------------------------------
  
  if (file.exists(latent_comparison_metrics_file)) {
    
    log_msg("📦 Ingesting latent comparison metrics...")
    
    latent_comparison_metrics_fact <- readr::read_csv(
      latent_comparison_metrics_file,
      show_col_types = FALSE
    ) |>
      dplyr::mutate(
        source_file    = fs::path_file(latent_comparison_metrics_file),
        analysis_scope = "latent_space",
        engine         = "latent",
        .before = 1
      )
    
    write_fact_table(
      con = con,
      table_name = "latent_comparison_metrics_fact",
      data = latent_comparison_metrics_fact,
      warehouse_dir = warehouse_dir,
      write_csv_exports = write_csv_exports
    )
    
    ingested_tables <- c(
      ingested_tables,
      "latent_comparison_metrics_fact"
    )
    
    row_counts <- dplyr::bind_rows(
      row_counts,
      tibble::tibble(
        table_name = "latent_comparison_metrics_fact",
        n_rows = nrow(latent_comparison_metrics_fact)
      )
    )
    
    log_msg(
      "✅ Latent comparison metric rows: ",
      nrow(latent_comparison_metrics_fact)
    )
    
  } else {
    
    skipped_files <- c(skipped_files, latent_comparison_metrics_file)
    
    log_msg(
      "⏭️ Skipping latent comparison metrics ingestion; file not found: ",
      latent_comparison_metrics_file
    )
  }
  
  # ---- Latent sample coordinates ---------------------------------------------
  
  if (file.exists(latent_sample_coordinates_file)) {
    
    log_msg("📦 Ingesting latent sample coordinates...")
    
    latent_sample_coordinates_fact <- readr::read_csv(
      latent_sample_coordinates_file,
      show_col_types = FALSE
    ) |>
      dplyr::mutate(
        source_file    = fs::path_file(latent_sample_coordinates_file),
        analysis_scope = "latent_space",
        engine         = "latent",
        .before = 1
      )
    
    write_fact_table(
      con = con,
      table_name = "latent_sample_coordinates_fact",
      data = latent_sample_coordinates_fact,
      warehouse_dir = warehouse_dir,
      write_csv_exports = write_csv_exports
    )
    
    ingested_tables <- c(
      ingested_tables,
      "latent_sample_coordinates_fact"
    )
    
    row_counts <- dplyr::bind_rows(
      row_counts,
      tibble::tibble(
        table_name = "latent_sample_coordinates_fact",
        n_rows = nrow(latent_sample_coordinates_fact)
      )
    )
    
    log_msg(
      "✅ Latent sample coordinate rows: ",
      nrow(latent_sample_coordinates_fact)
    )
    
  } else {
    
    skipped_files <- c(skipped_files, latent_sample_coordinates_file)
    
    log_msg(
      "⏭️ Skipping latent sample coordinates ingestion; file not found: ",
      latent_sample_coordinates_file
    )
  }
  
  # ---- Latent training summary -----------------------------------------------
  
  if (file.exists(latent_training_summary_file)) {
    
    log_msg("📦 Ingesting VAE training summary...")
    
    vae_training_summary_fact <- readr::read_csv(
      latent_training_summary_file,
      show_col_types = FALSE
    ) |>
      dplyr::mutate(
        source_file    = fs::path_file(latent_training_summary_file),
        analysis_scope = "latent_space",
        engine         = "latent",
        .before = 1
      )
    
    write_fact_table(
      con = con,
      table_name = "vae_training_summary_fact",
      data = vae_training_summary_fact,
      warehouse_dir = warehouse_dir,
      write_csv_exports = write_csv_exports
    )
    
    ingested_tables <- c(
      ingested_tables,
      "vae_training_summary_fact"
    )
    
    row_counts <- dplyr::bind_rows(
      row_counts,
      tibble::tibble(
        table_name = "vae_training_summary_fact",
        n_rows = nrow(vae_training_summary_fact)
      )
    )
    
    log_msg(
      "✅ VAE training summary rows: ",
      nrow(vae_training_summary_fact)
    )
    
  } else {
    
    skipped_files <- c(skipped_files, latent_training_summary_file)
    
    log_msg(
      "⏭️ Skipping VAE training summary ingestion; file not found: ",
      latent_training_summary_file
    )
  }
  
  # ---- Validation views ------------------------------------------------------
  
  if ("structural_complexity_results_fact" %in% ingested_tables) {
    DBI::dbExecute(con, "
    CREATE OR REPLACE VIEW vw_structural_complexity_result_counts AS
    SELECT
      engine,
      chip,
      filter_regime,
      gene_set_mode,
      COUNT(*) AS n_rows,
      COUNT(DISTINCT comparison) AS n_comparisons
    FROM structural_complexity_results_fact
    GROUP BY engine, chip, filter_regime, gene_set_mode
    ORDER BY chip, filter_regime, gene_set_mode
    ")
  }
  
  if ("structural_entropy_results_fact" %in% ingested_tables) {
    DBI::dbExecute(con, "
    CREATE OR REPLACE VIEW vw_structural_entropy_result_counts AS
    SELECT
      engine,
      chip,
      filter_regime,
      gene_set_mode,
      COUNT(*) AS n_rows,
      COUNT(DISTINCT comparison) AS n_comparisons
    FROM structural_entropy_results_fact
    GROUP BY engine, chip, filter_regime, gene_set_mode
    ORDER BY chip, filter_regime, gene_set_mode
    ")
  }
  
  if ("structural_mp_spectral_results_fact" %in% ingested_tables) {
    DBI::dbExecute(con, "
    CREATE OR REPLACE VIEW vw_structural_mp_spectral_result_counts AS
    SELECT
      engine,
      chip,
      filter_regime,
      gene_set_mode,
      condition,
      COUNT(*) AS n_rows,
      COUNT(DISTINCT comparison) AS n_comparisons
    FROM structural_mp_spectral_results_fact
    GROUP BY engine, chip, filter_regime, gene_set_mode, condition
    ORDER BY chip, filter_regime, condition
    ")
  }
  
  if ("structural_mp_spectral_deltas_fact" %in% ingested_tables) {
    DBI::dbExecute(con, "
    CREATE OR REPLACE VIEW vw_structural_mp_spectral_delta_counts AS
    SELECT
      engine,
      chip,
      filter_regime,
      gene_set_mode,
      COUNT(*) AS n_rows,
      COUNT(DISTINCT comparison) AS n_comparisons
    FROM structural_mp_spectral_deltas_fact
    GROUP BY engine, chip, filter_regime, gene_set_mode
    ORDER BY chip, filter_regime
    ")
  }
  
  if ("latent_comparison_metrics_fact" %in% ingested_tables) {
    DBI::dbExecute(con, "
    CREATE OR REPLACE VIEW vw_latent_comparison_metric_counts AS
    SELECT
      engine,
      chip,
      filter_regime,
      latent_model_id,
      COUNT(*) AS n_rows,
      COUNT(DISTINCT comparison) AS n_comparisons
    FROM latent_comparison_metrics_fact
    GROUP BY engine, chip, filter_regime, latent_model_id
    ORDER BY chip, filter_regime, latent_model_id
    ")
  }
  
  if ("latent_sample_coordinates_fact" %in% ingested_tables) {
    DBI::dbExecute(con, "
    CREATE OR REPLACE VIEW vw_latent_sample_coordinate_counts AS
    SELECT
      engine,
      chip,
      filter_regime,
      latent_model_id,
      COUNT(*) AS n_rows,
      COUNT(DISTINCT sample_id) AS n_samples
    FROM latent_sample_coordinates_fact
    GROUP BY engine, chip, filter_regime, latent_model_id
    ORDER BY chip, filter_regime, latent_model_id
    ")
  }
  
  # ---- Inventory view --------------------------------------------------------
  
  inventory_parts <- character()
  
  if ("structural_complexity_results_fact" %in% ingested_tables) {
    inventory_parts <- c(inventory_parts, "
      SELECT
        'structural_complexity_results_fact' AS table_name,
        engine,
        chip,
        filter_regime,
        gene_set_mode,
        COUNT(*) AS n_rows,
        COUNT(DISTINCT comparison) AS n_comparisons
      FROM structural_complexity_results_fact
      GROUP BY engine, chip, filter_regime, gene_set_mode
    ")
  }
  
  if ("structural_entropy_results_fact" %in% ingested_tables) {
    inventory_parts <- c(inventory_parts, "
      SELECT
        'structural_entropy_results_fact' AS table_name,
        engine,
        chip,
        filter_regime,
        gene_set_mode,
        COUNT(*) AS n_rows,
        COUNT(DISTINCT comparison) AS n_comparisons
      FROM structural_entropy_results_fact
      GROUP BY engine, chip, filter_regime, gene_set_mode
    ")
  }
  
  if ("structural_mp_spectral_deltas_fact" %in% ingested_tables) {
    inventory_parts <- c(inventory_parts, "
      SELECT
        'structural_mp_spectral_deltas_fact' AS table_name,
        engine,
        chip,
        filter_regime,
        gene_set_mode,
        COUNT(*) AS n_rows,
        COUNT(DISTINCT comparison) AS n_comparisons
      FROM structural_mp_spectral_deltas_fact
      GROUP BY engine, chip, filter_regime, gene_set_mode
    ")
  }
  
  if ("latent_comparison_metrics_fact" %in% ingested_tables) {
    inventory_parts <- c(inventory_parts, "
      SELECT
        'latent_comparison_metrics_fact' AS table_name,
        engine,
        chip,
        filter_regime,
        'LATENT' AS gene_set_mode,
        COUNT(*) AS n_rows,
        COUNT(DISTINCT comparison) AS n_comparisons
      FROM latent_comparison_metrics_fact
      GROUP BY engine, chip, filter_regime
    ")
  }
  
  if (length(inventory_parts) > 0) {
    DBI::dbExecute(
      con,
      paste0(
        "CREATE OR REPLACE VIEW vw_structural_result_inventory AS ",
        paste(inventory_parts, collapse = "\nUNION ALL\n"),
        "\nORDER BY table_name, chip, filter_regime"
      )
    )
    
    inventory <- DBI::dbGetQuery(
      con,
      "SELECT * FROM vw_structural_result_inventory"
    )
    
    log_msg("📊 Structural result inventory:")
    print(inventory)
    
  } else {
    inventory <- tibble::tibble()
    log_msg("⚠️ No structural result files were ingested.")
  }
  
  DBI::dbExecute(con, "CHECKPOINT")
  
  log_msg("✅ Structural DuckDB ingestion completed.")
  log_msg("📁 Database: ", db_path)
  
  invisible(list(
    db_path = db_path,
    warehouse_dir = warehouse_dir,
    ingested_tables = ingested_tables,
    skipped_files = skipped_files,
    row_counts = row_counts,
    inventory = inventory
  ))
}

# ------------------------------------------------------------------------------
# Standalone execution
# ------------------------------------------------------------------------------

if (sys.nframe() == 0) {
  ingest_structural_results_to_duckdb()
}