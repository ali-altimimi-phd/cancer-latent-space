# ==============================================================================
# File: ingest_preprocessing_annotations_to_duckdb.R
# Purpose: Ingest preprocessing annotation artifacts into DuckDB
# Role: Preprocessing warehouse orchestration ingestor
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
  library(fs)
  library(here)
})

# ------------------------------------------------------------------------------
# Main ingestion function
# ------------------------------------------------------------------------------

ingest_preprocessing_annotations_to_duckdb <- function(
    study_name = "global_cancer",
    annotation_path = here::here(
      "data", study_name, "processed", "RData", "annotations", "full_chip_annotations.rds"
    ),
    go_semantic_dir = here::here(
      "data", study_name, "processed", "RData", "annotations", "go_semantic_layer"
    ),
    warehouse_dir = here::here(
      "output", study_name, "warehouse"
    ),
    db_path = file.path(
      warehouse_dir, "global_cancer_results.duckdb"
    ),
    ingest_gene_set_annotations = TRUE,
    ingest_go_semantic_layer = TRUE,
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
  
  ingested_tables <- character()
  skipped_stages <- character()
  row_counts <- tibble::tibble(
    table_name = character(),
    n_rows = integer()
  )
  results <- list()
  
  # ---- Gene-set annotation dimension -----------------------------------------
  
  if (isTRUE(ingest_gene_set_annotations)) {
    log_msg("🦆 Ingesting gene-set annotation metadata...")
    
    source(
      here::here(
        "R", "warehouse", "ingest", "ingest_gene_set_annotations_to_duckdb.R"
      ),
      local = TRUE
    )
    
    results$gene_set_annotations <- ingest_gene_set_annotations_to_duckdb(
      study_name = study_name,
      annotation_path = annotation_path,
      warehouse_dir = warehouse_dir,
      db_path = db_path,
      write_csv_exports = write_csv_exports,
      logger = logger
    )
    
    ingested_tables <- c(
      ingested_tables,
      results$gene_set_annotations$ingested_tables
    )
    
    row_counts <- dplyr::bind_rows(
      row_counts,
      results$gene_set_annotations$row_counts
    )
    
  } else {
    skipped_stages <- c(skipped_stages, "gene_set_annotations")
    log_msg("⏭️ Gene-set annotation DuckDB ingestion skipped.")
  }
  
  # ---- GO semantic layer ------------------------------------------------------
  
  if (isTRUE(ingest_go_semantic_layer)) {
    log_msg("🦆 Ingesting GO semantic annotation layer...")
    
    source(
      here::here(
        "R", "warehouse", "ingest", "ingest_go_semantic_layer_to_duckdb.R"
      ),
      local = TRUE
    )
    
    results$go_semantic_layer <- ingest_go_semantic_layer_to_duckdb(
      study_name = study_name,
      go_semantic_dir = go_semantic_dir,
      warehouse_dir = warehouse_dir,
      db_path = db_path,
      write_csv_exports = write_csv_exports,
      logger = logger
    )
    
    ingested_tables <- c(
      ingested_tables,
      results$go_semantic_layer$ingested_tables
    )
    
    row_counts <- dplyr::bind_rows(
      row_counts,
      results$go_semantic_layer$row_counts
    )
    
  } else {
    skipped_stages <- c(skipped_stages, "go_semantic_layer")
    log_msg("⏭️ GO semantic layer DuckDB ingestion skipped.")
  }
  
  log_msg("✅ Preprocessing annotation DuckDB ingestion completed.")
  log_msg("📁 Database: ", db_path)
  
  invisible(list(
    db_path = db_path,
    warehouse_dir = warehouse_dir,
    ingested_tables = ingested_tables,
    skipped_stages = skipped_stages,
    row_counts = row_counts,
    results = results
  ))
}

# ------------------------------------------------------------------------------
# Standalone execution
# ------------------------------------------------------------------------------

if (sys.nframe() == 0) {
  ingest_preprocessing_annotations_to_duckdb()
}
