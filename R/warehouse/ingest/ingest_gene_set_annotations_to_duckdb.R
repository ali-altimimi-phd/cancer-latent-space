# ==============================================================================
# File: ingest_gene_set_annotations_to_duckdb.R
# Purpose: Ingest gene-set annotation metadata into DuckDB
# Role: Preprocessing warehouse ingestor
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ==============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(dplyr)
  library(purrr)
  library(stringr)
  library(tibble)
  library(fs)
  library(here)
})

# ------------------------------------------------------------------------------
# Build gene_set_annotation_dim for one chip
# ------------------------------------------------------------------------------

build_gene_set_annotation_dim_one_chip <- function(chip_annotations, chip_id) {
  annotation_table <- chip_annotations$annotation_table
  
  go_names <- annotation_table |>
    dplyr::filter(!is.na(.data$go_id)) |>
    dplyr::distinct(
      go_id,
      gene_set_name = go_name,
      root_node
    )
  
  go_dim <- chip_annotations$go_counts |>
    dplyr::left_join(go_names, by = "go_id") |>
    dplyr::transmute(
      chip = chip_id,
      gene_set_mode = paste0("GO_", .data$go_ontology),
      gene_set_id = .data$go_id,
      gene_set_normalized = .data$go_id,
      gene_set_name = .data$gene_set_name,
      go_ontology = .data$go_ontology,
      root_node = .data$root_node,
      n_probes = .data$n_probes
    )
  
  kegg_names <- annotation_table |>
    dplyr::filter(!is.na(.data$PATH)) |>
    dplyr::distinct(
      gene_set_id = PATH,
      gene_set_name = path_name
    )
  
  kegg_dim <- chip_annotations$kegg_counts |>
    dplyr::left_join(kegg_names, by = c("PATH" = "gene_set_id")) |>
    dplyr::transmute(
      chip = chip_id,
      gene_set_mode = "KEGG",
      gene_set_id = PATH,
      gene_set_normalized = PATH,
      gene_set_name = gene_set_name,
      go_ontology = NA_character_,
      root_node = NA_character_,
      n_probes = n_probes
    )
  
  msig_names <- annotation_table |>
    dplyr::filter(!is.na(.data$gs_name)) |>
    dplyr::distinct(
      gene_set_id = gs_name,
      gene_set_name = gs_name
    )
  
  msig_dim <- chip_annotations$msig_counts |>
    dplyr::left_join(msig_names, by = c("gs_name" = "gene_set_id")) |>
    dplyr::transmute(
      chip = chip_id,
      gene_set_mode = "MSIGDB",
      gene_set_id = gs_name,
      gene_set_normalized = gs_name,
      gene_set_name = gene_set_name,
      go_ontology = NA_character_,
      root_node = NA_character_,
      n_probes = n_probes
    )
  
  dplyr::bind_rows(go_dim, kegg_dim, msig_dim) |>
    dplyr::mutate(
      gene_set_name = dplyr::coalesce(.data$gene_set_name, .data$gene_set_id)
    ) |>
    dplyr::distinct()
}

# ------------------------------------------------------------------------------
# Main ingestion function
# ------------------------------------------------------------------------------

ingest_gene_set_annotations_to_duckdb <- function(
    study_name = "global_cancer",
    annotation_path = here::here(
      "data", study_name, "processed", "RData", "annotations", "full_chip_annotations.rds"
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
  
  if (!file.exists(annotation_path)) {
    stop("Annotation file not found: ", annotation_path, call. = FALSE)
  }
  
  log_msg("📦 Loading gene-set annotations: ", annotation_path)
  full_chip_annotations <- readRDS(annotation_path)
  
  gene_set_annotation_dim <- purrr::imap_dfr(
    full_chip_annotations,
    build_gene_set_annotation_dim_one_chip
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
  
  DBI::dbExecute(con, "DROP TABLE IF EXISTS gene_set_annotation_dim")
  
  DBI::dbWriteTable(
    con,
    "gene_set_annotation_dim",
    gene_set_annotation_dim,
    overwrite = TRUE
  )
  
  DBI::dbExecute(con, "
  CREATE INDEX IF NOT EXISTS idx_gene_set_annotation_dim_key
  ON gene_set_annotation_dim(chip, gene_set_mode, gene_set_normalized)
  ")
  
  if (isTRUE(write_csv_exports)) {
    readr::write_csv(
      gene_set_annotation_dim,
      file.path(warehouse_dir, "gene_set_annotation_dim.csv")
    )
  }
  
  inventory <- DBI::dbGetQuery(con, "
  SELECT
    chip,
    gene_set_mode,
    COUNT(*) AS n_gene_sets,
    SUM(n_probes) AS total_probe_memberships
  FROM gene_set_annotation_dim
  GROUP BY chip, gene_set_mode
  ORDER BY chip, gene_set_mode
  ")
  
  log_msg("📊 Gene-set annotation inventory:")
  print(inventory)
  
  log_msg("✅ Gene-set annotation DuckDB ingestion completed.")
  
  invisible(list(
    db_path = db_path,
    warehouse_dir = warehouse_dir,
    ingested_tables = "gene_set_annotation_dim",
    row_counts = tibble::tibble(
      table_name = "gene_set_annotation_dim",
      n_rows = nrow(gene_set_annotation_dim)
    ),
    inventory = inventory
  ))
}

# ------------------------------------------------------------------------------
# Standalone execution
# ------------------------------------------------------------------------------

if (sys.nframe() == 0) {
  ingest_gene_set_annotations_to_duckdb()
}
