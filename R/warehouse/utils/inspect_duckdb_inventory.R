# ==============================================================================
# File: inspect_duckdb_inventory.R
# Purpose: Inspect DuckDB warehouse contents
# Role: Warehouse inventory / diagnostic utility
# Project: Global Cancer Structural Inference Framework
# ==============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(dplyr)
  library(here)
  library(tibble)
})

# ---- Paths --------------------------------------------------------------------

duckdb_path <- here::here(
  "output",
  "global_cancer",
  "warehouse",
  "global_cancer_results.duckdb"
)

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

# ---- Inventory ----------------------------------------------------------------

tables <- DBI::dbListTables(con)

message("\n📦 Tables/views found: ", length(tables))

inventory_tbl <- purrr::map_dfr(
  tables,
  function(tbl_name) {
    
    n_rows <- tryCatch(
      {
        DBI::dbGetQuery(
          con,
          paste0(
            "SELECT COUNT(*) AS n FROM ",
            tbl_name
          )
        )$n[[1]]
      },
      error = function(e) {
        NA_integer_
      }
    )
    
    tibble(
      table_name = tbl_name,
      n_rows = n_rows
    )
  }
)

inventory_tbl <- inventory_tbl |>
  arrange(table_name)

message("\n🦆 DuckDB inventory:")
print(inventory_tbl, n = Inf)

# ---- Optional: show columns ---------------------------------------------------

message("\n📑 Column inventory:")

column_inventory <- purrr::map_dfr(
  tables,
  function(tbl_name) {
    
    cols <- DBI::dbListFields(con, tbl_name)
    
    tibble(
      table_name = tbl_name,
      column_name = cols
    )
  }
)

print(column_inventory, n = Inf)

# ---- Optional: classify by prefix ---------------------------------------------

message("\n🧭 Inventory by logical layer:")

layer_summary <- inventory_tbl |>
  mutate(
    layer = case_when(
      grepl("^structural_", table_name) ~ "structural",
      grepl("^biological_", table_name) ~ "biological",
      grepl("^latent_", table_name) ~ "latent",
      grepl("^diagnostic_", table_name) ~ "diagnostic",
      grepl("^vw_", table_name) ~ "views",
      TRUE ~ "other"
    )
  ) |>
  count(layer, name = "n_tables") |>
  arrange(layer)

print(layer_summary)

# ==============================================================================