# ==============================================================================
# File: create_biological_reporting_views.R
# Purpose: Build biological reporting views in DuckDB
# Role: Connection-agnostic view module for biological enrichment reporting
# ==============================================================================

# This file intentionally does not open or close a DuckDB connection when sourced.
# Call create_biological_reporting_views(con) from pipeline, reporting, or
# warehouse-refresh scripts.

# ---- Internal helpers ---------------------------------------------------------

.assert_relation_exists <- function(con, relation_name) {
  relation_exists <- DBI::dbGetQuery(
    con,
    sprintf(
      "
      SELECT COUNT(*) AS n
      FROM information_schema.tables
      WHERE table_name = %s
      ",
      DBI::dbQuoteString(con, relation_name)
    )
  )$n[[1]] > 0
  
  if (!relation_exists) {
    stop("Required DuckDB relation is missing: ",
         relation_name,
         call. = FALSE)
  }
  
  invisible(TRUE)
}

# ---- Create views -------------------------------------------------------------

#' Create biological reporting views
#'
#' Builds the semantic DuckDB views used by biological enrichment reporting.
#' The function is idempotent and connection-agnostic: the caller owns the
#' DuckDB connection lifecycle.
#'
#' Required input relations:
#'   - biological_gene_set_results_fact
#'   - gene_set_annotation_dim
#'
#' Created views:
#'   - vw_biological_gene_set_results_annotated
#'   - vw_report_biological_aggregated_results
#'   - vw_report_biological_top_complexity
#'   - vw_report_biological_top_entropy
#'   - vw_report_biological_significant_terms
#'
#' @param con Open DBI DuckDB connection.
#' @param create_fdr Logical; if TRUE, create p-report/FDR convenience fields.
#' @return Invisibly returns TRUE.
create_biological_reporting_views <- function(con, create_fdr = TRUE) {
  .assert_relation_exists(con, "biological_gene_set_results_fact")
  .assert_relation_exists(con, "gene_set_annotation_dim")
  
  # ---- Annotated biological results view -------------------------------------
  
  DBI::dbExecute(
    con,
    "
  CREATE OR REPLACE VIEW vw_biological_gene_set_results_annotated AS
  WITH r_norm AS (
    SELECT
      r.*,
      CASE
        WHEN regexp_matches(r.gene_set_id, '^GO\\.')
          THEN regexp_replace(r.gene_set_id, '^GO\\.', 'GO:')
        WHEN regexp_matches(r.gene_set_id, '^X[0-9]{5}$')
          THEN regexp_replace(r.gene_set_id, '^X', '')
        ELSE r.gene_set_id
      END AS gene_set_normalized
    FROM biological_gene_set_results_fact r
  )
  SELECT
    r_norm.*,
    a.gene_set_name AS annotation_gene_set_name,
    a.go_ontology,
    a.root_node,
    a.n_probes AS annotated_n_probes
  FROM r_norm
  LEFT JOIN gene_set_annotation_dim a
    ON r_norm.chip = a.chip
   AND r_norm.gene_set_mode = a.gene_set_mode
   AND r_norm.gene_set_normalized = a.gene_set_normalized
  "
  )
  
  # ---- Cleaned biological reporting view -------------------------------------
  # p_report is the reporting-level biological term p-value.
  #   complexity: p_perm
  #   entropy:    p_perm_spectral preferred, then p_perm_shannon, then p_perm
  # This keeps structural-comparison p-values conceptually separate from
  # gene-set-level p-values used for biological enrichment reporting.
  
  DBI::dbExecute(
    con,
    "
  CREATE OR REPLACE VIEW vw_report_biological_aggregated_results AS
  SELECT
    *,
  COALESCE(annotation_gene_set_name, gene_set_name, gene_set_id)
  AS report_gene_set_name,
    CASE
      WHEN engine = 'entropy'
        THEN COALESCE(p_perm_spectral, p_perm_shannon, p_perm)
      ELSE p_perm
    END AS p_report,
    CASE
      WHEN engine = 'entropy'
        THEN COALESCE(spectral_direction, direction)
      ELSE direction
    END AS report_direction
  FROM vw_biological_gene_set_results_annotated
  "
  )
  
  if (isTRUE(create_fdr)) {
    DBI::dbExecute(
      con,
      "
    CREATE OR REPLACE VIEW vw_report_biological_aggregated_results_fdr AS
    WITH base AS (
      SELECT
        *,
        ROW_NUMBER() OVER (
          PARTITION BY chip, filter_regime, gene_set_mode, engine
          ORDER BY p_report
        ) AS p_report_rank,
        COUNT(*) OVER (
          PARTITION BY chip, filter_regime, gene_set_mode, engine
        ) AS n_report_tests
      FROM vw_report_biological_aggregated_results
      WHERE p_report IS NOT NULL
    ),
    bh AS (
      SELECT
        *,
        p_report * n_report_tests / p_report_rank AS p_report_bh_raw
      FROM base
    )
    SELECT
      *,
      LEAST(
        1.0,
        MIN(p_report_bh_raw) OVER (
          PARTITION BY chip, filter_regime, gene_set_mode, engine
          ORDER BY p_report_rank DESC
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
      ) AS p_report_fdr
    FROM bh
    "
    )
  }
  
  # ---- Descriptor-ranked reporting views -------------------------------------
  
  DBI::dbExecute(
    con,
    "
  CREATE OR REPLACE VIEW vw_report_biological_top_complexity AS
  SELECT *
  FROM vw_report_biological_aggregated_results
  WHERE engine = 'complexity'
  ORDER BY ABS(kappa_delta) DESC NULLS LAST
  "
  )
  
  DBI::dbExecute(
    con,
    "
  CREATE OR REPLACE VIEW vw_report_biological_top_entropy AS
  SELECT *
  FROM vw_report_biological_aggregated_results
  WHERE engine = 'entropy'
  ORDER BY ABS(spectral_delta) DESC NULLS LAST
  "
  )
  
  # ---- Significant term convenience view -------------------------------------
  
  DBI::dbExecute(
    con,
    "
  CREATE OR REPLACE VIEW vw_report_biological_significant_terms AS
  SELECT *
  FROM vw_report_biological_aggregated_results
  WHERE p_report IS NOT NULL
    AND p_report <= 0.05
  "
  )
  
  invisible(TRUE)
}

# ---- Validation ---------------------------------------------------------------

#' Validate biological reporting views
#'
#' Returns a named list of small validation tables. Printing is intentionally
#' left to the caller.
#'
#' @param con Open DBI DuckDB connection.
#' @return Named list of data frames.
validate_biological_reporting_views <- function(con) {
  .assert_relation_exists(con, "vw_report_biological_aggregated_results")
  
  list(
    annotation_coverage = DBI::dbGetQuery(
      con,
      "
      SELECT
        gene_set_mode,
        COUNT(*) AS n_rows,
        SUM(CASE WHEN annotation_gene_set_name IS NOT NULL THEN 1 ELSE 0 END)
          AS n_annotated,
        SUM(CASE WHEN annotation_gene_set_name IS NULL THEN 1 ELSE 0 END)
          AS n_missing_annotation
      FROM vw_biological_gene_set_results_annotated
      GROUP BY gene_set_mode
      ORDER BY gene_set_mode
    "
    ),
    
    reporting_view_rows = DBI::dbGetQuery(
      con,
      "
      SELECT
        engine,
        gene_set_mode,
        COUNT(*) AS n_rows,
        SUM(CASE WHEN report_gene_set_name IS NOT NULL THEN 1 ELSE 0 END)
          AS n_named,
        SUM(CASE WHEN p_report IS NOT NULL THEN 1 ELSE 0 END)
          AS n_with_p_report
      FROM vw_report_biological_aggregated_results
      GROUP BY engine, gene_set_mode
      ORDER BY engine, gene_set_mode
    "
    ),
    
    pvalue_availability = DBI::dbGetQuery(
      con,
      "
      SELECT
        engine,
        COUNT(*) AS n_rows,
        SUM(CASE WHEN p_perm IS NOT NULL THEN 1 ELSE 0 END)
          AS n_p_perm,
        SUM(CASE WHEN p_perm_shannon IS NOT NULL THEN 1 ELSE 0 END)
          AS n_p_perm_shannon,
        SUM(CASE WHEN p_perm_spectral IS NOT NULL THEN 1 ELSE 0 END)
          AS n_p_perm_spectral,
        SUM(CASE WHEN p_report IS NOT NULL THEN 1 ELSE 0 END)
          AS n_p_report
      FROM vw_report_biological_aggregated_results
      GROUP BY engine
      ORDER BY engine
    "
    )
  )
}
