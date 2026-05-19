# ==============================================================================
# File: R/structural_concordance/inspect_concordance_and_build_archetypes.R
# Purpose:
#   Inspect structural quadrant stability and build an archetype summary table
#   from the structural concordance DuckDB tables.
#
# Run after:
#   source("R/structural_concordance/run_structural_concordance.R")
# ==============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(dplyr)
  library(readr)
  library(here)
})

# ------------------------------------------------------------------------------
# 1. Connect to DuckDB warehouse
# ------------------------------------------------------------------------------

db_path <- here::here(
  "output",
  "global_cancer",
  "warehouse",
  "global_cancer_results.duckdb"
)

message("🦆 Connecting to DuckDB warehouse: ", db_path)

con <- DBI::dbConnect(
  duckdb::duckdb(),
  dbdir = db_path,
  read_only = FALSE
)

on.exit({
  DBI::dbDisconnect(con, shutdown = TRUE)
}, add = TRUE)

# ------------------------------------------------------------------------------
# 2. Check concordance inventory
# ------------------------------------------------------------------------------

message("📦 Concordance inventory")

concordance_inventory <- DBI::dbGetQuery(con, "
SELECT *
FROM structural_concordance_inventory
ORDER BY table_name
")

print(concordance_inventory)

# ------------------------------------------------------------------------------
# 3. Inspect quadrant assignment distribution
# ------------------------------------------------------------------------------

message("🧭 Quadrant distribution by confidence")

quadrant_distribution <- DBI::dbGetQuery(con, "
SELECT
  quadrant,
  quadrant_confidence,
  COUNT(*) AS n
FROM structural_quadrant_assignments
GROUP BY quadrant, quadrant_confidence
ORDER BY quadrant, quadrant_confidence
")

print(quadrant_distribution)

# ------------------------------------------------------------------------------
# 4. Inspect stability across chips and filter regimes
# ------------------------------------------------------------------------------

message("🧪 Quadrant stability table")

quadrant_stability <- DBI::dbGetQuery(con, "
SELECT *
FROM structural_quadrant_stability
ORDER BY
  modal_quadrant_fraction DESC,
  n_unique_quadrants ASC,
  comparison
")

print(tibble::as_tibble(quadrant_stability), n = Inf)

# ------------------------------------------------------------------------------
# 5. Build archetype summary table
# ------------------------------------------------------------------------------

message("🏷️ Building structural archetype summary table")

archetype_summary <- quadrant_stability |>
  mutate(
    archetype_stability = case_when(
      stable_all_assignments & n_chips >= 2 & n_filter_regimes >= 3 ~
        "cross_chip_cross_filter_stable",
      
      stable_all_assignments ~
        "stable_among_available_assignments",
      
      modal_quadrant_fraction >= 0.75 ~
        "majority_stable",
      
      TRUE ~
        "unstable_mixed"
    ),
    
    archetype_label = case_when(
      archetype_stability == "cross_chip_cross_filter_stable" ~
        paste0("stable_Q", modal_quadrant),
      
      archetype_stability == "stable_among_available_assignments" ~
        paste0("available_stable_Q", modal_quadrant),
      
      archetype_stability == "majority_stable" ~
        paste0("majority_Q", modal_quadrant),
      
      TRUE ~
        "unstable_mixed"
    ),
    
    archetype_confidence = case_when(
      archetype_stability == "cross_chip_cross_filter_stable" &
        high_confidence_fraction >= 0.75 ~ "high",
      
      archetype_stability %in% c(
        "cross_chip_cross_filter_stable",
        "stable_among_available_assignments",
        "majority_stable"
      ) &
        high_confidence_fraction >= 0.33 ~ "moderate",
      
      archetype_stability %in% c(
        "cross_chip_cross_filter_stable",
        "stable_among_available_assignments",
        "majority_stable"
      ) ~ "low",
      
      TRUE ~ "unstable"
    ),
    
    archetype_interpretation = case_when(
      modal_quadrant == "I" ~
        "coordinated structural expansion",
      
      modal_quadrant == "II" ~
        "entropy-dominant disorganization with reduced complexity",
      
      modal_quadrant == "III" ~
        "coordinated structural compression",
      
      modal_quadrant == "IV" ~
        "complexity-dominant constrained reorganization",
      
      TRUE ~
        "mixed or boundary structural behavior"
    )
  ) |>
  select(
    group,
    comparison,
    archetype_label,
    archetype_stability,
    archetype_confidence,
    archetype_interpretation,
    modal_quadrant,
    modal_quadrant_fraction,
    stable_all_assignments,
    n_unique_quadrants,
    n_assignments,
    n_chips,
    n_filter_regimes,
    chips_seen,
    filter_regimes_seen,
    high_confidence_fraction,
    median_distance_from_boundary_z,
    stability_class,
    everything()
  )

print(tibble::as_tibble(archetype_summary), n = Inf)

# ------------------------------------------------------------------------------
# 6. Write archetype summary to CSV/RDS
# ------------------------------------------------------------------------------

output_dir <- here::here(
  "output",
  "global_cancer",
  "structural_inference",
  "tables",
  "concordance",
  "tables"
)

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

csv_path <- file.path(output_dir, "structural_archetype_summary.csv")
rds_path <- here::here(
  "output",
  "global_cancer",
  "structural_inference",
  "tables",
  "concordance",
  "RData",
  "structural_archetype_summary.rds"
)

readr::write_csv(archetype_summary, csv_path)
saveRDS(archetype_summary, rds_path)

message("💾 Wrote CSV: ", csv_path)
message("💾 Wrote RDS: ", rds_path)

# ------------------------------------------------------------------------------
# 7. Materialize archetype table into DuckDB
# ------------------------------------------------------------------------------

DBI::dbWriteTable(
  con,
  "structural_archetype_summary",
  archetype_summary,
  overwrite = TRUE
)

message("🦆 Materialized DuckDB table: structural_archetype_summary")

# ------------------------------------------------------------------------------
# 8. Quick archetype counts
# ------------------------------------------------------------------------------

message("📊 Archetype counts")

archetype_counts <- DBI::dbGetQuery(con, "
SELECT
  archetype_label,
  archetype_stability,
  archetype_confidence,
  COUNT(*) AS n
FROM structural_archetype_summary
GROUP BY
  archetype_label,
  archetype_stability,
  archetype_confidence
ORDER BY
  archetype_stability,
  archetype_label,
  archetype_confidence
")

print(archetype_counts)

message("✅ Archetype inspection complete.")