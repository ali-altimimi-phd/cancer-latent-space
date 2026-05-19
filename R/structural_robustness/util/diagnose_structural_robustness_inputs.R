# ==============================================================================
# Diagnose Structural Robustness Inputs
# ==============================================================================
#
# Purpose:
#   Validate DuckDB structural synthesis views before running the structural
#   robustness layer.
#
# Checks:
#   - view existence
#   - required columns
#   - candidate robustness axes
#   - missingness
#   - duplicate rows
#   - regime coverage
#
# ==============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(dplyr)
  library(here)
})


# ==============================================================================
# 1. Configuration
# ==============================================================================

study_name <- "global_cancer"

warehouse_path <- here::here(
  "output",
  study_name,
  "warehouse",
  "global_cancer_results.duckdb"
)

source_view <- "vw_structural_phenotype_wide_with_latent_overlay"


# ==============================================================================
# 2. Connect
# ==============================================================================

cat("\n🔌 Connecting to DuckDB...\n")

con <- DBI::dbConnect(
  duckdb::duckdb(),
  dbdir = warehouse_path,
  read_only = TRUE
)


# ==============================================================================
# 3. Check view existence
# ==============================================================================

available_tables <- DBI::dbGetQuery(con, "
SELECT table_name
FROM information_schema.tables
")

if (!(source_view %in% available_tables$table_name)) {
  
  stop(
    "\n❌ Source view not found:\n",
    source_view,
    "\n"
  )
}

cat("\n✅ Found source view:\n")
cat(source_view, "\n")


# ==============================================================================
# 4. Inspect columns
# ==============================================================================

cat("\n📋 Inspecting columns...\n")

columns_df <- DBI::dbGetQuery(
  con,
  paste0("PRAGMA table_info('", source_view, "')")
)

# print(columns_df, n = Inf)
print(as.data.frame(columns_df))


# ==============================================================================
# 5. Candidate robustness axes
# ==============================================================================

candidate_complexity <- c(
  "composite_kappa_delta",
  "kappa_delta",
  "effrank_delta",
  "complexity_delta"
)

candidate_entropy <- c(
  "mp_spectral_entropy_delta",
  "spectral_entropy_delta",
  "entropy_delta"
)

available_cols <- columns_df$name

complexity_found <- intersect(candidate_complexity, available_cols)
entropy_found <- intersect(candidate_entropy, available_cols)

cat("\n🧠 Candidate complexity axes:\n")
print(complexity_found)

cat("\n🧠 Candidate entropy/MP axes:\n")
print(entropy_found)


# ==============================================================================
# 6. Basic data preview
# ==============================================================================

preview_query <- paste0("
SELECT *
FROM ", source_view, "
LIMIT 10
")

preview_df <- DBI::dbGetQuery(con, preview_query)

cat("\n👀 Preview rows:\n")
print(preview_df)


# ==============================================================================
# 7. Coverage diagnostics
# ==============================================================================

coverage_query <- paste0("
SELECT
  comparison,
  COUNT(*) AS n_rows,
  COUNT(DISTINCT chip) AS n_chips,
  COUNT(DISTINCT filter_regime) AS n_filters
FROM ", source_view, "
GROUP BY comparison
ORDER BY comparison
")

coverage_df <- DBI::dbGetQuery(con, coverage_query)

cat("\n📊 Coverage diagnostics:\n")
# print(coverage_df, n = Inf)
print(as.data.frame(coverage_df))


# ==============================================================================
# 8. Duplicate structure check
# ==============================================================================

duplicate_query <- paste0("
SELECT
  comparison,
  chip,
  filter_regime,
  COUNT(*) AS n
FROM ", source_view, "
GROUP BY
  comparison,
  chip,
  filter_regime
HAVING COUNT(*) > 1
")

duplicate_df <- DBI::dbGetQuery(con, duplicate_query)

cat("\n🧬 Duplicate assignment check:\n")

if (nrow(duplicate_df) == 0) {
  
  cat("✅ No duplicate comparison/chip/filter rows found.\n")
  
} else {
  
  print(duplicate_df, n = Inf)
}


# ==============================================================================
# 9. Missingness diagnostics
# ==============================================================================

if (length(complexity_found) > 0 &&
    length(entropy_found) > 0) {
  
  x_col <- complexity_found[1]
  y_col <- entropy_found[1]
  
  missing_query <- paste0("
  SELECT
    SUM(CASE WHEN ", x_col, " IS NULL THEN 1 ELSE 0 END)
      AS missing_x,

    SUM(CASE WHEN ", y_col, " IS NULL THEN 1 ELSE 0 END)
      AS missing_y,

    COUNT(*) AS total_rows
  FROM ", source_view
  )
  
  missing_df <- DBI::dbGetQuery(con, missing_query)
  
  cat("\n🩺 Missingness diagnostics:\n")
  print(missing_df)
}


# ==============================================================================
# 10. Disconnect
# ==============================================================================

DBI::dbDisconnect(con, shutdown = TRUE)

cat("\n✅ Structural robustness diagnostic complete.\n")