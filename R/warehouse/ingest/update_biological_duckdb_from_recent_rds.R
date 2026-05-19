# ==============================================================================
# File: update_biological_duckdb_from_recent_rds.R
# Purpose: Partially update biological DuckDB fact tables from selected RDS files
# Role: Biological warehouse partial updater
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

study_name <- "global_cancer"

biological_rdata_dir <- here::here(
  "output", study_name, "biological_enrichment", "RData"
)

warehouse_dir <- here::here(
  "output", study_name, "warehouse"
)

db_path <- file.path(
  warehouse_dir,
  "global_cancer_results.duckdb"
)

# ---- Partial-update controls --------------------------------------------------
#
# Use this script after a partial biological rerun. It updates only the selected
# biological result files in DuckDB and preserves all other rows already present
# in the biological warehouse facts.
#
# Example:
#   updated_gene_set_modes <- c("KEGG", "MSIGDB")
#
# This matches a partial rerun where GO_BP / GO_MF were not regenerated.

updated_engines <- c(
  "complexity",
  "entropy"
)

updated_chips <- c(
  "hu35ksuba",
  "hu6800"
)

updated_filter_regimes <- c(
  "limma",
  "variance_top3000"
)

updated_gene_set_modes <- c(
  "KEGG",
  "MSIGDB"
)

# ---- Metadata authority -------------------------------------------------------
# Keep aligned with the full biological warehouse builder.
#
# Current setting:
#   "filename" = use parsed filename metadata as canonical.
#
# This remains useful while older entropy biological result rows may contain
# stale internal metadata such as gene_set_mode/gene_set_name.

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
# Locate only the regenerated biological RDS files
# ==============================================================================

message("🔎 Locating selected regenerated biological enrichment RDS files...")

all_candidate_files <- list.files(
  biological_rdata_dir,
  pattern = paste0(
    "^(complexity|entropy)_results_",
    "(hu35ksuba|hu6800)_",
    "(limma|variance_top3000)_",
    "(go_bp|go_mf|kegg|msigdb)\\.rds$"
  ),
  full.names = TRUE
)

if (length(all_candidate_files) == 0) {
  stop("No biological result RDS files found in: ", biological_rdata_dir)
}

candidate_meta <- purrr::map_dfr(all_candidate_files, parse_biological_filename) |>
  dplyr::mutate(file_path = all_candidate_files)

selected_meta <- candidate_meta |>
  dplyr::filter(
    .data$engine %in% updated_engines,
    .data$chip %in% updated_chips,
    .data$filter_regime %in% updated_filter_regimes,
    .data$gene_set_mode %in% updated_gene_set_modes
  )

if (nrow(selected_meta) == 0) {
  stop("No selected biological RDS files matched the partial-update filters.")
}

selected_files <- selected_meta$file_path
selected_source_files <- selected_meta$source_file

message("📦 Selected biological result files: ", length(selected_files))
print(
  selected_meta |>
    dplyr::select(
      source_file,
      engine,
      chip,
      filter_regime,
      gene_set_mode
    ) |>
    dplyr::arrange(
      engine,
      chip,
      filter_regime,
      gene_set_mode
    )
)

# ==============================================================================
# Flatten selected results and skipped records
# ==============================================================================

message("🔀 Flattening selected biological analyzed result rows...")

selected_biological_gene_set_results_fact <- purrr::map_dfr(
  selected_files,
  flatten_biological_result_file
)

message("✅ Selected biological analyzed rows flattened: ",
        nrow(selected_biological_gene_set_results_fact))

message("🔀 Flattening selected biological skipped rows...")

selected_biological_gene_set_skipped_fact <- purrr::map_dfr(
  selected_files,
  extract_biological_skipped_file
)

message("✅ Selected biological skipped rows flattened: ",
        nrow(selected_biological_gene_set_skipped_fact))

# ==============================================================================
# Open DuckDB and partially replace selected rows
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

required_tables <- c(
  "biological_gene_set_results_fact",
  "biological_gene_set_skipped_fact"
)

missing_tables <- setdiff(required_tables, DBI::dbListTables(con))

if (length(missing_tables) > 0) {
  stop(
    "Required biological fact table(s) missing from DuckDB: ",
    paste(missing_tables, collapse = ", "),
    ". Run the full biological warehouse builder once before using this partial updater."
  )
}

# ---- Delete existing rows for exactly the regenerated files -------------------

quoted_source_files <- paste(
  DBI::dbQuoteString(con, selected_source_files),
  collapse = ", "
)

message("🧹 Removing old rows for selected source files from biological facts...")

deleted_results <- DBI::dbExecute(
  con,
  paste0(
    "DELETE FROM biological_gene_set_results_fact ",
    "WHERE source_file IN (", quoted_source_files, ")"
  )
)

deleted_skipped <- DBI::dbExecute(
  con,
  paste0(
    "DELETE FROM biological_gene_set_skipped_fact ",
    "WHERE source_file IN (", quoted_source_files, ")"
  )
)

message("🧹 Deleted result rows: ", deleted_results)
message("🧹 Deleted skipped rows: ", deleted_skipped)

# ---- Append regenerated rows --------------------------------------------------

message("➕ Appending regenerated biological result rows...")

DBI::dbWriteTable(
  con,
  "biological_gene_set_results_fact",
  selected_biological_gene_set_results_fact,
  append = TRUE
)

message("➕ Appending regenerated biological skipped rows...")

if (nrow(selected_biological_gene_set_skipped_fact) > 0) {
  DBI::dbWriteTable(
    con,
    "biological_gene_set_skipped_fact",
    selected_biological_gene_set_skipped_fact,
    append = TRUE
  )
}

# ==============================================================================
# Refresh biological summary views
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

message("📄 Re-exporting biological fact CSVs for inspection...")

readr::write_csv(
  DBI::dbReadTable(con, "biological_gene_set_results_fact"),
  file.path(warehouse_dir, "biological_gene_set_results_fact.csv")
)

readr::write_csv(
  DBI::dbReadTable(con, "biological_gene_set_skipped_fact"),
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

message("📊 Updated p-value availability for selected source files:")
print(
  DBI::dbGetQuery(
    con,
    paste0(
      "
      SELECT
        source_file,
        engine,
        chip,
        filter_regime,
        gene_set_mode,
        COUNT(*) AS n_rows,
        SUM(CASE WHEN p_perm IS NOT NULL THEN 1 ELSE 0 END) AS n_p_perm,
        SUM(CASE WHEN p_perm_shannon IS NOT NULL THEN 1 ELSE 0 END) AS n_p_perm_shannon,
        SUM(CASE WHEN p_perm_spectral IS NOT NULL THEN 1 ELSE 0 END) AS n_p_perm_spectral
      FROM biological_gene_set_results_fact
      WHERE source_file IN (", quoted_source_files, ")
      GROUP BY
        source_file,
        engine,
        chip,
        filter_regime,
        gene_set_mode
      ORDER BY
        engine,
        chip,
        filter_regime,
        gene_set_mode
      "
    )
  )
)

message("✅ Partial biological DuckDB update completed successfully.")
message("📁 Database: ", db_path)
