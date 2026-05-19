# ==============================================================================
# File: ingest_go_semantic_layer_to_duckdb.R
# Purpose: Ingest p-value-neutral GO semantic annotation layer into DuckDB
# Role: Preprocessing warehouse ingestor
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ==============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(readr)
  library(dplyr)
  library(tibble)
  library(fs)
  library(here)
})

# ------------------------------------------------------------------------------
# Main ingestion function
# ------------------------------------------------------------------------------

ingest_go_semantic_layer_to_duckdb <- function(
    study_name = "global_cancer",
    go_semantic_dir = here::here(
      "data", study_name, "processed", "RData", "annotations", "go_semantic_layer"
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
  
  required_files <- c(
    "go_semantic_universe.csv",
    "go_semantic_term_dim.csv",
    "go_semantic_cluster_membership.csv",
    "go_semantic_cluster_dim.csv",
    "go_semantic_provenance.csv"
  )
  
  missing_files <- required_files[
    !file.exists(file.path(go_semantic_dir, required_files))
  ]
  
  if (length(missing_files) > 0) {
    stop(
      "Missing GO semantic layer files in ", go_semantic_dir, ": ",
      paste(missing_files, collapse = ", "),
      call. = FALSE
    )
  }
  
  log_msg("📦 Loading GO semantic layer from: ", go_semantic_dir)
  
  go_semantic_universe <- readr::read_csv(
    file.path(go_semantic_dir, "go_semantic_universe.csv"),
    show_col_types = FALSE
  )
  
  go_semantic_term_dim <- readr::read_csv(
    file.path(go_semantic_dir, "go_semantic_term_dim.csv"),
    show_col_types = FALSE
  )
  
  go_semantic_cluster_membership <- readr::read_csv(
    file.path(go_semantic_dir, "go_semantic_cluster_membership.csv"),
    show_col_types = FALSE
  )
  
  go_semantic_cluster_dim <- readr::read_csv(
    file.path(go_semantic_dir, "go_semantic_cluster_dim.csv"),
    show_col_types = FALSE
  )
  
  go_semantic_provenance <- readr::read_csv(
    file.path(go_semantic_dir, "go_semantic_provenance.csv"),
    show_col_types = FALSE
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
  
  DBI::dbWriteTable(con, "go_semantic_universe", go_semantic_universe, overwrite = TRUE)
  DBI::dbWriteTable(con, "go_semantic_term_dim", go_semantic_term_dim, overwrite = TRUE)
  DBI::dbWriteTable(con, "go_semantic_cluster_membership", go_semantic_cluster_membership, overwrite = TRUE)
  DBI::dbWriteTable(con, "go_semantic_cluster_dim", go_semantic_cluster_dim, overwrite = TRUE)
  DBI::dbWriteTable(con, "go_semantic_provenance", go_semantic_provenance, overwrite = TRUE)
  
  DBI::dbExecute(con, "
  CREATE INDEX IF NOT EXISTS idx_go_semantic_universe_key
  ON go_semantic_universe(chip, ontology, go_id)
  ")
  
  DBI::dbExecute(con, "
  CREATE INDEX IF NOT EXISTS idx_go_semantic_term_dim_key
  ON go_semantic_term_dim(ontology, go_id)
  ")
  
  DBI::dbExecute(con, "
  CREATE INDEX IF NOT EXISTS idx_go_semantic_membership_key
  ON go_semantic_cluster_membership(ontology, go_id, semantic_cluster_id)
  ")
  
  DBI::dbExecute(con, "
  CREATE INDEX IF NOT EXISTS idx_go_semantic_cluster_dim_key
  ON go_semantic_cluster_dim(ontology, semantic_cluster_id)
  ")
  
  DBI::dbExecute(con, "
  CREATE OR REPLACE VIEW vw_gene_set_annotation_with_go_semantic_clusters AS
  SELECT
    g.*,
    m.semantic_cluster_id,
    m.semantic_block_id,
    m.semantic_block_name,
    m.cluster_method,
    m.semantic_similarity_cutoff,
    m.semantic_block_n_terms
  FROM gene_set_annotation_dim g
  LEFT JOIN go_semantic_cluster_membership m
    ON g.gene_set_normalized = m.go_id
   AND g.go_ontology = m.ontology
  WHERE g.gene_set_mode LIKE 'GO_%'
  ")
  
  DBI::dbExecute(con, "
  CREATE OR REPLACE VIEW vw_go_semantic_cluster_summary AS
  SELECT
    c.ontology,
    c.semantic_cluster_id,
    c.semantic_block_id,
    c.semantic_block_name,
    c.root_node,
    c.n_terms,
    c.representative_go_ids,
    c.representative_terms,
    c.cluster_method,
    c.semantic_similarity_cutoff,
    c.semantic_block_n_terms,
    c.preblock_method,
    c.preblock_go_id
  FROM go_semantic_cluster_dim c
  ")
  
  if (isTRUE(write_csv_exports)) {
    readr::write_csv(go_semantic_universe, file.path(warehouse_dir, "go_semantic_universe.csv"))
    readr::write_csv(go_semantic_term_dim, file.path(warehouse_dir, "go_semantic_term_dim.csv"))
    readr::write_csv(go_semantic_cluster_membership, file.path(warehouse_dir, "go_semantic_cluster_membership.csv"))
    readr::write_csv(go_semantic_cluster_dim, file.path(warehouse_dir, "go_semantic_cluster_dim.csv"))
    readr::write_csv(go_semantic_provenance, file.path(warehouse_dir, "go_semantic_provenance.csv"))
  }
  
  membership_inventory <- DBI::dbGetQuery(con, "
  SELECT ontology, cluster_method, COUNT(*) AS n_terms
  FROM go_semantic_cluster_membership
  GROUP BY ontology, cluster_method
  ORDER BY ontology, cluster_method
  ")
  
  annotation_coverage <- DBI::dbGetQuery(con, "
  SELECT
    COUNT(*) AS n_go_gene_sets,
    COUNT(semantic_cluster_id) AS n_with_semantic_cluster,
    COUNT(*) - COUNT(semantic_cluster_id) AS n_without_semantic_cluster
  FROM vw_gene_set_annotation_with_go_semantic_clusters
  ")
  
  cluster_inventory <- DBI::dbGetQuery(con, "
  SELECT
    ontology,
    COUNT(*) AS n_clusters,
    AVG(n_terms) AS mean_terms_per_cluster,
    MAX(n_terms) AS max_terms_per_cluster
  FROM go_semantic_cluster_dim
  GROUP BY ontology
  ORDER BY ontology
  ")
  
  log_msg("📊 GO semantic membership inventory:")
  print(membership_inventory)
  
  log_msg("📊 GO semantic annotation coverage:")
  print(annotation_coverage)
  
  log_msg("📊 GO semantic cluster inventory:")
  print(cluster_inventory)
  
  log_msg("✅ GO semantic layer DuckDB ingestion completed.")
  
  invisible(list(
    db_path = db_path,
    warehouse_dir = warehouse_dir,
    ingested_tables = c(
      "go_semantic_universe",
      "go_semantic_term_dim",
      "go_semantic_cluster_membership",
      "go_semantic_cluster_dim",
      "go_semantic_provenance"
    ),
    row_counts = tibble::tibble(
      table_name = c(
        "go_semantic_universe",
        "go_semantic_term_dim",
        "go_semantic_cluster_membership",
        "go_semantic_cluster_dim",
        "go_semantic_provenance"
      ),
      n_rows = c(
        nrow(go_semantic_universe),
        nrow(go_semantic_term_dim),
        nrow(go_semantic_cluster_membership),
        nrow(go_semantic_cluster_dim),
        nrow(go_semantic_provenance)
      )
    ),
    membership_inventory = membership_inventory,
    annotation_coverage = annotation_coverage,
    cluster_inventory = cluster_inventory
  ))
}

# ------------------------------------------------------------------------------
# Standalone execution
# ------------------------------------------------------------------------------

if (sys.nframe() == 0) {
  ingest_go_semantic_layer_to_duckdb()
}
