# ==============================================================================
# File: build_global_cancer_results_duckdb.R
# Purpose: Build DuckDB warehouse for Global Cancer result layers
# Role: Warehouse builder / aggregation replacement
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ==============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(dplyr)
  library(purrr)
  library(tibble)
  library(stringr)
  library(fs)
  library(readr)
  library(here)
})

# ==============================================================================
# Configuration
# ==============================================================================

biological_rdata_dir <- here::here(
  "output", "global_cancer", "biological_enrichment", "RData"
)

warehouse_dir <- here::here(
  "output", "global_cancer", "warehouse"
)

db_path <- file.path(
  warehouse_dir,
  "global_cancer_results.duckdb"
)

fs::dir_create(warehouse_dir)

# ---- Metadata authority -------------------------------------------------------
# Current setting:
#   "filename" = use parsed filename metadata as canonical.
#
# Reason:
#   entropy biological result rows currently report gene_set_mode/gene_set_name
#   as FULL/FULL. Until that engine-side bug is fixed, filename-derived metadata
#   is authoritative.
#
# Later:
#   after complexity and entropy both write correct internal metadata, switch to:
#     metadata_authority <- "internal"
#
metadata_authority <- "filename"
# metadata_authority <- "internal"

drop_large_columns <- TRUE

drop_cols <- c(
  "perm_dist",
  "boot_1",
  "boot_2",
  "per_sample_1",
  "per_sample_2",
  "null_distribution",
  "bootstrap_distribution"
)

# ==============================================================================
# Filename parser
# ==============================================================================

parse_biological_filename <- function(file_path) {
  file_name <- fs::path_file(file_path)
  
  m <- stringr::str_match(
    file_name,
    paste0(
      "^(complexity|entropy)_results_",
      "(hu35ksuba|hu6800)_",
      "(limma|variance_top3000)_",
      "(go_bp|go_mf|kegg|msigdb)\\.rds$"
    )
  )
  
  if (any(is.na(m))) {
    stop("Could not parse biological result filename: ", file_name)
  }
  
  tibble::tibble(
    source_file    = file_name,
    analysis_scope = "gene_set_subset",
    engine         = m[, 2],
    chip           = m[, 3],
    filter_regime  = m[, 4],
    gene_set_mode  = toupper(m[, 5])
  )
}

# ==============================================================================
# Metadata helpers
# ==============================================================================

apply_canonical_metadata <- function(df,
                                     meta,
                                     comparison_name = NULL,
                                     gene_set_id = NULL) {
  if (metadata_authority == "filename") {
    
    # TEMPORARY PATH:
    # Use filename-derived metadata as canonical.
    #
    # This deliberately removes internally reported metadata fields that may
    # currently be wrong in entropy results, especially gene_set_mode/gene_set_name.
    #
    # After fixing entropy result metadata, switch metadata_authority to "internal"
    # and eventually remove/deprecate this filename override block.
    
    df |>
      dplyr::select(
        -dplyr::any_of(c(
          "chip",
          "engine",
          "filter_regime",
          "gene_set_mode",
          "gene_set_name",
          "comparison"
        ))
      ) |>
      dplyr::mutate(
        source_file    = meta$source_file,
        analysis_scope = meta$analysis_scope,
        engine         = meta$engine,
        chip           = meta$chip,
        filter_regime  = meta$filter_regime,
        gene_set_mode  = meta$gene_set_mode,
        comparison     = comparison_name,
        gene_set_id    = gene_set_id,
        gene_set_name  = gene_set_id,
        .before = 1
      )
    
  } else if (metadata_authority == "internal") {
    
    # FUTURE PATH:
    # Use engine-produced metadata fields from each result row.
    #
    # Enable only after complexity and entropy both write correct internal
    # gene_set_mode and gene_set_name values.
    
    df |>
      dplyr::mutate(
        source_file    = meta$source_file,
        analysis_scope = meta$analysis_scope,
        gene_set_id    = gene_set_id,
        .before = 1
      )
    
  } else {
    stop("Invalid metadata_authority: ", metadata_authority)
  }
}

apply_canonical_skipped_metadata <- function(df, meta) {
  if (metadata_authority == "filename") {
    
    df |>
      dplyr::select(
        -dplyr::any_of(c(
          "chip",
          "engine",
          "filter_regime",
          "gene_set_mode"
        ))
      ) |>
      dplyr::mutate(
        source_file    = meta$source_file,
        analysis_scope = meta$analysis_scope,
        engine         = meta$engine,
        chip           = meta$chip,
        filter_regime  = meta$filter_regime,
        gene_set_mode  = meta$gene_set_mode,
        .before = 1
      )
    
  } else if (metadata_authority == "internal") {
    
    df |>
      dplyr::mutate(
        source_file    = meta$source_file,
        analysis_scope = meta$analysis_scope,
        .before = 1
      )
    
  } else {
    stop("Invalid metadata_authority: ", metadata_authority)
  }
}

# ==============================================================================
# Biological result flatteners
# ==============================================================================

safe_drop_large_columns <- function(df) {
  if (!drop_large_columns) {
    return(df)
  }
  
  df |>
    dplyr::select(-dplyr::any_of(drop_cols))
}

flatten_biological_result_file <- function(file_path) {
  meta <- parse_biological_filename(file_path)
  
  x <- readRDS(file_path)
  
  if (!"results" %in% names(x)) {
    stop("RDS file does not contain a `results` element: ", file_path)
  }
  
  purrr::imap_dfr(x$results, function(res_by_set, comparison_name) {
    if (!is.list(res_by_set)) return(NULL)
    
    purrr::imap_dfr(res_by_set, function(entry, gene_set_id) {
      if (is.null(entry)) return(NULL)
      if (!inherits(entry, "data.frame")) return(NULL)
      
      entry |>
        safe_drop_large_columns() |>
        apply_canonical_metadata(
          meta            = meta,
          comparison_name = comparison_name,
          gene_set_id     = gene_set_id
        )
    })
  })
}

extract_biological_skipped_file <- function(file_path) {
  meta <- parse_biological_filename(file_path)
  
  x <- readRDS(file_path)
  
  if (!"skipped" %in% names(x)) {
    return(tibble::tibble())
  }
  
  skipped <- x$skipped
  
  if (!inherits(skipped, "data.frame") || nrow(skipped) == 0) {
    return(tibble::tibble())
  }
  
  skipped |>
    apply_canonical_skipped_metadata(meta = meta)
}

# ==============================================================================
# Locate biological RDS files
# ==============================================================================

message("🔎 Locating biological enrichment RDS files...")

biological_files <- list.files(
  biological_rdata_dir,
  pattern = paste0(
    "^(complexity|entropy)_results_",
    "(hu35ksuba|hu6800)_",
    "(limma|variance_top3000)_",
    "(go_bp|go_mf|kegg|msigdb)\\.rds$"
  ),
  full.names = TRUE
)

if (length(biological_files) == 0) {
  stop("No biological result RDS files found in: ", biological_rdata_dir)
}

message("📦 Biological result files found: ", length(biological_files))

# ==============================================================================
# Flatten biological results and skipped records
# ==============================================================================

message("🔀 Flattening biological analyzed result rows...")

biological_gene_set_results_fact <- purrr::map_dfr(
  biological_files,
  flatten_biological_result_file
)

message("✅ Biological analyzed rows flattened: ",
        nrow(biological_gene_set_results_fact))

message("🔀 Flattening biological skipped rows...")

biological_gene_set_skipped_fact <- purrr::map_dfr(
  biological_files,
  extract_biological_skipped_file
)

message("✅ Biological skipped rows flattened: ",
        nrow(biological_gene_set_skipped_fact))

# ==============================================================================
# Build DuckDB warehouse
# ==============================================================================

message("🦆 Opening DuckDB warehouse: ", db_path)

con <- DBI::dbConnect(
  duckdb::duckdb(),
  dbdir = db_path,
  read_only = FALSE
)

on.exit({
  DBI::dbDisconnect(con, shutdown = TRUE)
}, add = TRUE)

DBI::dbWriteTable(
  con,
  "biological_gene_set_results_fact",
  biological_gene_set_results_fact,
  overwrite = TRUE
)

DBI::dbWriteTable(
  con,
  "biological_gene_set_skipped_fact",
  biological_gene_set_skipped_fact,
  overwrite = TRUE
)

# ==============================================================================
# Views
# ==============================================================================

DBI::dbExecute(con, "
CREATE OR REPLACE VIEW vw_biological_result_counts AS
SELECT
  engine,
  chip,
  filter_regime,
  gene_set_mode,
  COUNT(*) AS n_analyzed_rows,
  COUNT(DISTINCT comparison) AS n_comparisons,
  COUNT(DISTINCT gene_set_id) AS n_gene_sets
FROM biological_gene_set_results_fact
GROUP BY
  engine,
  chip,
  filter_regime,
  gene_set_mode
ORDER BY
  engine,
  chip,
  filter_regime,
  gene_set_mode
")

DBI::dbExecute(con, "
CREATE OR REPLACE VIEW vw_biological_skipped_counts AS
SELECT
  engine,
  chip,
  filter_regime,
  gene_set_mode,
  reason,
  COUNT(*) AS n_skipped_rows,
  COUNT(DISTINCT comparison) AS n_comparisons,
  COUNT(DISTINCT gene_set_name) AS n_gene_sets
FROM biological_gene_set_skipped_fact
GROUP BY
  engine,
  chip,
  filter_regime,
  gene_set_mode,
  reason
ORDER BY
  engine,
  chip,
  filter_regime,
  gene_set_mode,
  n_skipped_rows DESC
")

DBI::dbExecute(con, "
CREATE OR REPLACE VIEW vw_biological_complexity_results AS
SELECT *
FROM biological_gene_set_results_fact
WHERE engine = 'complexity'
")

DBI::dbExecute(con, "
CREATE OR REPLACE VIEW vw_biological_entropy_results AS
SELECT *
FROM biological_gene_set_results_fact
WHERE engine = 'entropy'
")

DBI::dbExecute(con, "
CREATE OR REPLACE VIEW vw_biological_gene_set_throughput AS
SELECT
  r.engine,
  r.chip,
  r.filter_regime,
  r.gene_set_mode,
  r.n_analyzed_rows,
  COALESCE(s.n_skipped_rows, 0) AS n_skipped_rows,
  r.n_analyzed_rows + COALESCE(s.n_skipped_rows, 0) AS n_total_rows
FROM vw_biological_result_counts r
LEFT JOIN (
  SELECT
    engine,
    chip,
    filter_regime,
    gene_set_mode,
    SUM(n_skipped_rows) AS n_skipped_rows
  FROM vw_biological_skipped_counts
  GROUP BY
    engine,
    chip,
    filter_regime,
    gene_set_mode
) s
ON
  r.engine = s.engine AND
  r.chip = s.chip AND
  r.filter_regime = s.filter_regime AND
  r.gene_set_mode = s.gene_set_mode
ORDER BY
  r.engine,
  r.chip,
  r.filter_regime,
  r.gene_set_mode
")

DBI::dbExecute(con, "
CREATE OR REPLACE TABLE warehouse_tables_manifest AS
SELECT
  'biological_gene_set_results_fact' AS table_name,
  'gene_set_subset' AS analysis_scope,
  'Successful GO_BP / GO_MF / KEGG / MSIGDB biological enrichment results' AS description
UNION ALL
SELECT
  'biological_gene_set_skipped_fact' AS table_name,
  'gene_set_subset' AS analysis_scope,
  'Skipped biological gene-set rows with admissibility failure reasons' AS description
")

# ==============================================================================
# CSV exports for inspection
# ==============================================================================

readr::write_csv(
  biological_gene_set_results_fact,
  file.path(warehouse_dir, "biological_gene_set_results_fact.csv")
)

readr::write_csv(
  biological_gene_set_skipped_fact,
  file.path(warehouse_dir, "biological_gene_set_skipped_fact.csv")
)

# ==============================================================================
# Validation printout
# ==============================================================================

message("📊 Biological result count summary:")
print(
  DBI::dbGetQuery(
    con,
    "SELECT * FROM vw_biological_result_counts"
  )
)

message("📊 Biological throughput summary:")
print(
  DBI::dbGetQuery(
    con,
    "SELECT * FROM vw_biological_gene_set_throughput"
  )
)

message("✅ DuckDB warehouse built successfully.")
message("📁 Database: ", db_path)
message("📄 Results CSV: ",
        file.path(warehouse_dir, "biological_gene_set_results_fact.csv"))
message("📄 Skipped CSV: ",
        file.path(warehouse_dir, "biological_gene_set_skipped_fact.csv"))