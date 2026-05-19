# ==============================================================================
# File: diagnose_gene_set_probe_counts_from_duckdb.R
# Purpose: Export gene-set probe-count diagnostics from DuckDB
# Role: Standalone diagnostic for choosing min_gene_set_probes
# ==============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(dplyr)
  library(readr)
  library(here)
})

# ---- Paths --------------------------------------------------------------------

duckdb_path <- here::here(
  "output",
  "global_cancer",
  "warehouse",
  "global_cancer_results.duckdb"
)

output_dir <- here::here(
  "output",
  "global_cancer",
  "diagnostics",
  "gene_set_probe_counts"
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

message("📂 Project root: ", here::here())
message("🦆 DuckDB path: ", duckdb_path)

# ---- Connect ------------------------------------------------------------------

con <- DBI::dbConnect(
  duckdb::duckdb(),
  dbdir = duckdb_path,
  read_only = TRUE
)

on.exit({
  DBI::dbDisconnect(con, shutdown = TRUE)
}, add = TRUE)

# ---- Validate source table -----------------------------------------------------

source_table <- "gene_set_annotation_dim"

if (!source_table %in% DBI::dbListTables(con)) {
  stop("Required table not found: ", source_table)
}

required_cols <- c(
  "chip",
  "gene_set_mode",
  "gene_set_id",
  "gene_set_name",
  "n_probes"
)

available_cols <- DBI::dbListFields(con, source_table)
missing_cols <- setdiff(required_cols, available_cols)

if (length(missing_cols) > 0) {
  stop(
    "Missing required columns in ", source_table, ": ",
    paste(missing_cols, collapse = ", ")
  )
}

# ---- Detailed table ------------------------------------------------------------

gene_set_probe_counts <- DBI::dbGetQuery(
  con,
  "
  SELECT
      chip,
      gene_set_mode,
      gene_set_id,
      gene_set_name,
      go_ontology,
      root_node,
      n_probes
  FROM gene_set_annotation_dim
  WHERE
      chip IS NOT NULL
      AND gene_set_mode IN ('GO_BP', 'GO_MF', 'KEGG', 'MSIGDB')
      AND n_probes IS NOT NULL
  ORDER BY
      gene_set_mode,
      chip,
      n_probes DESC,
      gene_set_name
  "
)

# ---- Summary table -------------------------------------------------------------

gene_set_probe_summary <- DBI::dbGetQuery(
  con,
  "
  SELECT
      chip,
      gene_set_mode,
      COUNT(*) AS n_gene_sets,

      MIN(n_probes) AS min_probes,
      quantile_cont(n_probes, 0.25) AS q1,
      median(n_probes) AS median,
      AVG(n_probes) AS mean,
      quantile_cont(n_probes, 0.75) AS q3,
      MAX(n_probes) AS max_probes,

      SUM(CASE WHEN n_probes >= 5 THEN 1 ELSE 0 END) AS n_ge_5,
      100.0 * AVG(CASE WHEN n_probes >= 5 THEN 1 ELSE 0 END) AS pct_ge_5,

      SUM(CASE WHEN n_probes >= 8 THEN 1 ELSE 0 END) AS n_ge_8,
      100.0 * AVG(CASE WHEN n_probes >= 8 THEN 1 ELSE 0 END) AS pct_ge_8,

      SUM(CASE WHEN n_probes >= 10 THEN 1 ELSE 0 END) AS n_ge_10,
      100.0 * AVG(CASE WHEN n_probes >= 10 THEN 1 ELSE 0 END) AS pct_ge_10,

      SUM(CASE WHEN n_probes >= 15 THEN 1 ELSE 0 END) AS n_ge_15,
      100.0 * AVG(CASE WHEN n_probes >= 15 THEN 1 ELSE 0 END) AS pct_ge_15,

      SUM(CASE WHEN n_probes >= 20 THEN 1 ELSE 0 END) AS n_ge_20,
      100.0 * AVG(CASE WHEN n_probes >= 20 THEN 1 ELSE 0 END) AS pct_ge_20

  FROM gene_set_annotation_dim
  WHERE
      chip IS NOT NULL
      AND gene_set_mode IN ('GO_BP', 'GO_MF', 'KEGG', 'MSIGDB')
      AND n_probes IS NOT NULL
  GROUP BY
      chip,
      gene_set_mode
  ORDER BY
      gene_set_mode,
      chip
  "
)

# ---- Threshold suggestions -----------------------------------------------------

threshold_suggestions <- gene_set_probe_summary |>
  group_by(gene_set_mode) |>
  summarise(
    n_chip_rows = n(),
    
    min_q1_across_chips = min(q1, na.rm = TRUE),
    min_median_across_chips = min(median, na.rm = TRUE),
    mean_median_across_chips = mean(median, na.rm = TRUE),
    
    min_pct_ge_5 = min(pct_ge_5, na.rm = TRUE),
    min_pct_ge_8 = min(pct_ge_8, na.rm = TRUE),
    min_pct_ge_10 = min(pct_ge_10, na.rm = TRUE),
    min_pct_ge_15 = min(pct_ge_15, na.rm = TRUE),
    min_pct_ge_20 = min(pct_ge_20, na.rm = TRUE),
    
    suggested_from_min_q1 = pmax(5, floor(min_q1_across_chips)),
    suggested_from_min_median = pmax(5, floor(min_median_across_chips)),
    suggested_from_mean_median = pmax(5, floor(mean_median_across_chips)),
    
    .groups = "drop"
  ) |>
  arrange(gene_set_mode)

# ---- Export -------------------------------------------------------------------

probe_counts_path <- file.path(
  output_dir,
  "gene_set_probe_counts_by_chip_mode.csv"
)

summary_path <- file.path(
  output_dir,
  "gene_set_probe_count_summary_by_chip_mode.csv"
)

threshold_path <- file.path(
  output_dir,
  "gene_set_probe_threshold_suggestions_by_mode.csv"
)

readr::write_csv(gene_set_probe_counts, probe_counts_path)
readr::write_csv(gene_set_probe_summary, summary_path)
readr::write_csv(threshold_suggestions, threshold_path)

message("✅ Wrote detailed probe counts: ", probe_counts_path)
message("✅ Wrote summary table: ", summary_path)
message("✅ Wrote threshold suggestions: ", threshold_path)

message("\n📊 Gene-set probe-count summary:")
print(gene_set_probe_summary)

message("\n🧭 Threshold suggestions:")
print(threshold_suggestions)