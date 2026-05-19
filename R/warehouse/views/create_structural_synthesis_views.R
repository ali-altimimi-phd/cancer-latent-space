# ==============================================================================
# File: create_structural_synthesis_views.R
# Purpose: Create DuckDB synthesis views for structural + latent phenotype tables
# Role: Structural synthesis view builder
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ==============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(here)
})

study_name <- "global_cancer"

warehouse_dir <- here::here(
  "output", study_name, "warehouse"
)

db_path <- file.path(
  warehouse_dir,
  "global_cancer_results.duckdb"
)

if (!file.exists(db_path)) {
  stop("DuckDB warehouse not found: ", db_path)
}

con <- DBI::dbConnect(
  duckdb::duckdb(),
  dbdir = db_path,
  read_only = FALSE
)

on.exit({
  DBI::dbDisconnect(con, shutdown = TRUE)
}, add = TRUE)

# ==============================================================================
# 1. Long descriptor-first synthesis view
# ==============================================================================

DBI::dbExecute(con, "
CREATE OR REPLACE VIEW vw_structural_phenotype_long AS

-- Complexity: kappa_delta
SELECT
  chip,
  filter_regime,
  \"group\",
  comparison,
  engine,
  'kappa_delta' AS metric,
  kappa_delta AS value,
  CASE
    WHEN kappa_delta > 0 THEN 'tumor_greater'
    WHEN kappa_delta < 0 THEN 'tumor_lower'
    WHEN kappa_delta = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  'permutation_available' AS inference_status,
  p_perm AS p_value,
  p_perm_fdr AS p_fdr,
  'permutation' AS resampling_method,
  source_file,
  'complexity_engine' AS source_layer,
  NULL AS notes
FROM structural_complexity_results_fact

UNION ALL

-- Complexity: effrank_delta
SELECT
  chip,
  filter_regime,
  \"group\",
  comparison,
  engine,
  'effrank_delta' AS metric,
  effrank_delta AS value,
  CASE
    WHEN effrank_delta > 0 THEN 'tumor_greater'
    WHEN effrank_delta < 0 THEN 'tumor_lower'
    WHEN effrank_delta = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  'permutation_available' AS inference_status,
  p_perm AS p_value,
  p_perm_fdr AS p_fdr,
  'permutation' AS resampling_method,
  source_file,
  'complexity_engine' AS source_layer,
  NULL AS notes
FROM structural_complexity_results_fact

UNION ALL

-- Complexity: sparsity_delta
SELECT
  chip,
  filter_regime,
  \"group\",
  comparison,
  engine,
  'sparsity_delta' AS metric,
  sparsity_delta AS value,
  CASE
    WHEN sparsity_delta > 0 THEN 'tumor_greater'
    WHEN sparsity_delta < 0 THEN 'tumor_lower'
    WHEN sparsity_delta = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  'descriptive' AS inference_status,
  NULL AS p_value,
  NULL AS p_fdr,
  NULL AS resampling_method,
  source_file,
  'complexity_engine' AS source_layer,
  NULL AS notes
FROM structural_complexity_results_fact

UNION ALL

-- Complexity: composite_kappa_delta
SELECT
  chip,
  filter_regime,
  \"group\",
  comparison,
  engine,
  'composite_kappa_delta' AS metric,
  composite_kappa_delta AS value,
  CASE
    WHEN composite_kappa_delta > 0 THEN 'tumor_greater'
    WHEN composite_kappa_delta < 0 THEN 'tumor_lower'
    WHEN composite_kappa_delta = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  'descriptive' AS inference_status,
  NULL AS p_value,
  NULL AS p_fdr,
  NULL AS resampling_method,
  source_file,
  'complexity_engine' AS source_layer,
  NULL AS notes
FROM structural_complexity_results_fact

UNION ALL

-- Entropy: shannon_delta
SELECT
  chip,
  filter_regime,
  \"group\",
  comparison,
  engine,
  'shannon_delta' AS metric,
  shannon_delta AS value,
  CASE
    WHEN shannon_delta > 0 THEN 'tumor_greater'
    WHEN shannon_delta < 0 THEN 'tumor_lower'
    WHEN shannon_delta = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  CASE
    WHEN p_perm_shannon IS NULL THEN 'descriptive'
    ELSE 'permutation_available'
  END AS inference_status,
  p_perm_shannon AS p_value,
  p_perm_shannon_fdr AS p_fdr,
  CASE
    WHEN p_perm_shannon IS NULL THEN NULL
    ELSE 'permutation'
  END AS resampling_method,
  source_file,
  'entropy_engine' AS source_layer,
  NULL AS notes
FROM structural_entropy_results_fact

UNION ALL

-- Entropy: spectral_delta
SELECT
  chip,
  filter_regime,
  \"group\",
  comparison,
  engine,
  'spectral_delta' AS metric,
  spectral_delta AS value,
  CASE
    WHEN spectral_delta > 0 THEN 'tumor_greater'
    WHEN spectral_delta < 0 THEN 'tumor_lower'
    WHEN spectral_delta = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  CASE
    WHEN p_perm_spectral IS NULL THEN 'descriptive'
    ELSE 'permutation_available'
  END AS inference_status,
  p_perm_spectral AS p_value,
  p_perm_spectral_fdr AS p_fdr,
  CASE
    WHEN p_perm_spectral IS NULL THEN NULL
    ELSE 'permutation'
  END AS resampling_method,
  source_file,
  'entropy_engine' AS source_layer,
  NULL AS notes
FROM structural_entropy_results_fact

UNION ALL

-- MP: spectral_entropy_delta
SELECT
  chip,
  filter_regime,
  \"group\",
  comparison,
  engine,
  'spectral_entropy_delta' AS metric,
  spectral_entropy_delta AS value,
  CASE
    WHEN spectral_entropy_delta > 0 THEN 'tumor_greater'
    WHEN spectral_entropy_delta < 0 THEN 'tumor_lower'
    WHEN spectral_entropy_delta = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  'descriptive_only' AS inference_status,
  NULL AS p_value,
  NULL AS p_fdr,
  NULL AS resampling_method,
  source_file,
  'mp_engine' AS source_layer,
  'MP engine currently descriptive only.' AS notes
FROM structural_mp_spectral_deltas_fact

UNION ALL

-- MP: participation_ratio_delta
SELECT
  chip,
  filter_regime,
  \"group\",
  comparison,
  engine,
  'participation_ratio_delta' AS metric,
  participation_ratio_delta AS value,
  CASE
    WHEN participation_ratio_delta > 0 THEN 'tumor_greater'
    WHEN participation_ratio_delta < 0 THEN 'tumor_lower'
    WHEN participation_ratio_delta = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  'descriptive_only' AS inference_status,
  NULL AS p_value,
  NULL AS p_fdr,
  NULL AS resampling_method,
  source_file,
  'mp_engine' AS source_layer,
  'MP engine currently descriptive only.' AS notes
FROM structural_mp_spectral_deltas_fact

UNION ALL

-- MP: largest_eigenvalue_fraction_delta
SELECT
  chip,
  filter_regime,
  \"group\",
  comparison,
  engine,
  'largest_eigenvalue_fraction_delta' AS metric,
  largest_eigenvalue_fraction_delta AS value,
  CASE
    WHEN largest_eigenvalue_fraction_delta > 0 THEN 'tumor_greater'
    WHEN largest_eigenvalue_fraction_delta < 0 THEN 'tumor_lower'
    WHEN largest_eigenvalue_fraction_delta = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  'descriptive_only' AS inference_status,
  NULL AS p_value,
  NULL AS p_fdr,
  NULL AS resampling_method,
  source_file,
  'mp_engine' AS source_layer,
  'MP engine currently descriptive only.' AS notes
FROM structural_mp_spectral_deltas_fact

UNION ALL

-- MP: excess_spectral_mass_delta
SELECT
  chip,
  filter_regime,
  \"group\",
  comparison,
  engine,
  'excess_spectral_mass_delta' AS metric,
  excess_spectral_mass_delta AS value,
  CASE
    WHEN excess_spectral_mass_delta > 0 THEN 'tumor_greater'
    WHEN excess_spectral_mass_delta < 0 THEN 'tumor_lower'
    WHEN excess_spectral_mass_delta = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  'descriptive_only' AS inference_status,
  NULL AS p_value,
  NULL AS p_fdr,
  NULL AS resampling_method,
  source_file,
  'mp_engine' AS source_layer,
  'MP engine currently descriptive only.' AS notes
FROM structural_mp_spectral_deltas_fact

UNION ALL

-- Latent: latent_pr_delta
SELECT
  chip,
  filter_regime,
  NULL AS \"group\",
  comparison,
  engine,
  'latent_pr_delta' AS metric,
  pr_delta AS value,
  CASE
    WHEN pr_delta > 0 THEN 'tumor_greater'
    WHEN pr_delta < 0 THEN 'tumor_lower'
    WHEN pr_delta = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  'external_joined' AS inference_status,
  NULL AS p_value,
  NULL AS p_fdr,
  NULL AS resampling_method,
  source_file,
  'python_latent_geometry' AS source_layer,
  latent_model_id AS notes
FROM latent_comparison_metrics_fact
WHERE pr_delta IS NOT NULL

UNION ALL

-- Latent: latent_eig_entropy_delta
SELECT
  chip,
  filter_regime,
  NULL AS \"group\",
  comparison,
  engine,
  'latent_eig_entropy_delta' AS metric,
  eig_entropy_delta AS value,
  CASE
    WHEN eig_entropy_delta > 0 THEN 'tumor_greater'
    WHEN eig_entropy_delta < 0 THEN 'tumor_lower'
    WHEN eig_entropy_delta = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  'external_joined' AS inference_status,
  NULL AS p_value,
  NULL AS p_fdr,
  NULL AS resampling_method,
  source_file,
  'python_latent_geometry' AS source_layer,
  latent_model_id AS notes
FROM latent_comparison_metrics_fact
WHERE eig_entropy_delta IS NOT NULL

UNION ALL

-- Latent: centroid_distance
SELECT
  chip,
  filter_regime,
  NULL AS \"group\",
  comparison,
  engine,
  'centroid_distance' AS metric,
  centroid_distance AS value,
  CASE
    WHEN centroid_distance > 0 THEN 'tumor_greater'
    WHEN centroid_distance < 0 THEN 'tumor_lower'
    WHEN centroid_distance = 0 THEN 'no_change'
    ELSE NULL
  END AS direction,
  'external_joined' AS inference_status,
  NULL AS p_value,
  NULL AS p_fdr,
  NULL AS resampling_method,
  source_file,
  'python_latent_geometry' AS source_layer,
  latent_model_id AS notes
FROM latent_comparison_metrics_fact
WHERE centroid_distance IS NOT NULL
")

# ==============================================================================
# 2. Wide synthesis view
# ==============================================================================

DBI::dbExecute(con, "
CREATE OR REPLACE VIEW vw_structural_phenotype_wide AS
SELECT
  c.chip,
  c.filter_regime,
  c.\"group\",
  c.comparison,

  c.kappa_delta AS complexity__kappa_delta,
  c.effrank_delta AS complexity__effrank_delta,
  c.sparsity_delta AS complexity__sparsity_delta,
  c.composite_kappa_delta AS complexity__composite_kappa_delta,
  c.p_perm AS complexity__p_perm,
  c.p_perm_fdr AS complexity__p_perm_fdr,

  e.shannon_delta AS entropy__shannon_delta,
  e.spectral_delta AS entropy__spectral_delta,
  e.p_perm_shannon AS entropy__p_perm_shannon,
  e.p_perm_shannon_fdr AS entropy__p_perm_shannon_fdr,
  e.p_perm_spectral AS entropy__p_perm_spectral,
  e.p_perm_spectral_fdr AS entropy__p_perm_spectral_fdr,

  mp.spectral_entropy_delta AS mp__spectral_entropy_delta,
  mp.participation_ratio_delta AS mp__participation_ratio_delta,
  mp.largest_eigenvalue_fraction_delta AS mp__largest_eigenvalue_fraction_delta,
  mp.excess_spectral_mass_delta AS mp__excess_spectral_mass_delta,
  mp.n_spikes_delta AS mp__n_spikes_delta,
  mp.spike_fraction_delta AS mp__spike_fraction_delta,

  l.latent_model_id AS latent__model_id,
  l.pr_delta AS latent__pr_delta,
  l.eig_entropy_delta AS latent__eig_entropy_delta,
  l.centroid_distance AS latent__centroid_distance

FROM structural_complexity_results_fact c

LEFT JOIN structural_entropy_results_fact e
ON
  c.chip = e.chip AND
  c.filter_regime = e.filter_regime AND
  c.comparison = e.comparison

LEFT JOIN structural_mp_spectral_deltas_fact mp
ON
  c.chip = mp.chip AND
  c.filter_regime = mp.filter_regime AND
  c.comparison = mp.comparison

LEFT JOIN latent_comparison_metrics_fact l
ON
  c.chip = l.chip AND
  c.filter_regime = l.filter_regime AND
  c.comparison = l.comparison
")

# ==============================================================================
# 3. Latent-overlay wide view
# ==============================================================================

DBI::dbExecute(con, "
CREATE OR REPLACE VIEW vw_structural_phenotype_wide_with_latent_overlay AS
SELECT
  c.chip,
  c.filter_regime,
  c.\"group\",
  c.comparison,

  c.kappa_delta AS complexity__kappa_delta,
  c.effrank_delta AS complexity__effrank_delta,
  c.sparsity_delta AS complexity__sparsity_delta,
  c.composite_kappa_delta AS complexity__composite_kappa_delta,

  e.shannon_delta AS entropy__shannon_delta,
  e.spectral_delta AS entropy__spectral_delta,

  mp.spectral_entropy_delta AS mp__spectral_entropy_delta,
  mp.participation_ratio_delta AS mp__participation_ratio_delta,
  mp.largest_eigenvalue_fraction_delta AS mp__largest_eigenvalue_fraction_delta,
  mp.excess_spectral_mass_delta AS mp__excess_spectral_mass_delta,

  l.latent_model_id AS latent__model_id,
  l.chip AS latent__chip,
  l.filter_regime AS latent__filter_regime,
  l.pr_delta AS latent__pr_delta,
  l.eig_entropy_delta AS latent__eig_entropy_delta,
  l.centroid_distance AS latent__centroid_distance

FROM structural_complexity_results_fact c

LEFT JOIN structural_entropy_results_fact e
ON
  c.chip = e.chip AND
  c.filter_regime = e.filter_regime AND
  c.comparison = e.comparison

LEFT JOIN structural_mp_spectral_deltas_fact mp
ON
  c.chip = mp.chip AND
  c.filter_regime = mp.filter_regime AND
  c.comparison = mp.comparison

LEFT JOIN latent_comparison_metrics_fact l
ON
  c.comparison = l.comparison
")

# ==============================================================================
# 4. Summary view
# ==============================================================================

DBI::dbExecute(con, "
CREATE OR REPLACE VIEW vw_structural_phenotype_summary AS
SELECT
  chip,
  filter_regime,
  engine,
  metric,
  inference_status,
  COUNT(DISTINCT comparison) AS n_comparisons,
  COUNT(value) AS n_values,
  AVG(value) AS mean_value,
  MEDIAN(value) AS median_value,
  STDDEV_SAMP(value) AS sd_value,
  SUM(CASE WHEN direction = 'tumor_greater' THEN 1 ELSE 0 END) AS n_tumor_greater,
  SUM(CASE WHEN direction = 'tumor_lower' THEN 1 ELSE 0 END) AS n_tumor_lower,
  SUM(CASE WHEN direction = 'no_change' THEN 1 ELSE 0 END) AS n_no_change
FROM vw_structural_phenotype_long
GROUP BY
  chip,
  filter_regime,
  engine,
  metric,
  inference_status
ORDER BY
  chip,
  filter_regime,
  engine,
  metric
")

# ==============================================================================
# 5. Heatmap-ready z-score view
# ==============================================================================

DBI::dbExecute(con, "
CREATE OR REPLACE VIEW vw_structural_phenotype_heatmap_long AS
SELECT
  *,
  CASE
    WHEN STDDEV_SAMP(value) OVER (PARTITION BY engine, metric) IS NULL THEN NULL
    WHEN STDDEV_SAMP(value) OVER (PARTITION BY engine, metric) = 0 THEN NULL
    ELSE
      (
        value - AVG(value) OVER (PARTITION BY engine, metric)
      ) / STDDEV_SAMP(value) OVER (PARTITION BY engine, metric)
  END AS z_score
FROM vw_structural_phenotype_long
")

# ==============================================================================
# 6. Inventory view
# ==============================================================================

DBI::dbExecute(con, "
CREATE OR REPLACE VIEW vw_structural_synthesis_inventory AS
SELECT
  'vw_structural_phenotype_long' AS view_name,
  COUNT(*) AS n_rows
FROM vw_structural_phenotype_long

UNION ALL

SELECT
  'vw_structural_phenotype_wide' AS view_name,
  COUNT(*) AS n_rows
FROM vw_structural_phenotype_wide

UNION ALL

SELECT
  'vw_structural_phenotype_wide_with_latent_overlay' AS view_name,
  COUNT(*) AS n_rows
FROM vw_structural_phenotype_wide_with_latent_overlay

UNION ALL

SELECT
  'vw_structural_phenotype_summary' AS view_name,
  COUNT(*) AS n_rows
FROM vw_structural_phenotype_summary

UNION ALL

SELECT
  'vw_structural_phenotype_heatmap_long' AS view_name,
  COUNT(*) AS n_rows
FROM vw_structural_phenotype_heatmap_long
")

# ==============================================================================
# 7. Stale schema validation
# ==============================================================================

stale_views <- DBI::dbGetQuery(con, "
SELECT view_name, sql
FROM duckdb_views()
WHERE sql ILIKE '%latent_model_chip%'
   OR sql ILIKE '%latent_model_filter_regime%'
   OR sql ILIKE '%latent_model_scope%'
")

if (nrow(stale_views) > 0) {
  print(stale_views)
  stop(
    'Stale latent schema references remain in structural synthesis views.',
    call. = FALSE
  )
}

message('📊 Structural synthesis view inventory:')
print(
  DBI::dbGetQuery(
    con,
    'SELECT * FROM vw_structural_synthesis_inventory'
  )
)

message('✅ Structural synthesis views created successfully.')
message('📁 Database: ', db_path)