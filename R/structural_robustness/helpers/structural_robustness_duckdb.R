# ==============================================================================
# Structural Robustness DuckDB Helpers
# ==============================================================================
#
# Purpose:
#   Query validated structural phenotype data from the DuckDB warehouse and write
#   robustness outputs back to the warehouse.
#
# ==============================================================================


connect_structural_warehouse <- function(warehouse_path) {
  DBI::dbConnect(duckdb::duckdb(), dbdir = warehouse_path, read_only = FALSE)
}


disconnect_structural_warehouse <- function(con) {
  DBI::dbDisconnect(con, shutdown = TRUE)
}


load_structural_robustness_input <- function(con,
                                             source_view = "vw_structural_phenotype_wide_with_latent_overlay") {
  
  available_cols <- DBI::dbGetQuery(
    con,
    paste0("PRAGMA table_info('", source_view, "')")
  )$name
  
  preferred_x <- c(
    "complexity__composite_kappa_delta",
    "complexity__kappa_delta",
    "complexity__effrank_delta"
  )
  
  preferred_y <- c(
    "mp__spectral_entropy_delta",
    "entropy__spectral_delta"
  )
  
  x_col <- preferred_x[preferred_x %in% available_cols][1]
  y_col <- preferred_y[preferred_y %in% available_cols][1]
  
  if (is.na(x_col) || is.na(y_col)) {
    stop(
      "Could not identify required robustness axes in ",
      source_view,
      ". Need one complexity axis and one MP/entropy spectral axis."
    )
  }
  
  query <- paste0(
    "
  SELECT
    chip,
    filter_regime,
    \"group\" AS group_label,
    comparison,
    ",
    x_col, " AS complexity_delta,
    ",
    y_col, " AS spectral_entropy_delta
  FROM ", source_view, "
  WHERE comparison IS NOT NULL
  "
  )

  
  DBI::dbGetQuery(con, query)
}


write_structural_robustness_tables <- function(con,
                                               boundary_assignments,
                                               robustness_summary,
                                               trajectory_summary,
                                               overwrite = TRUE) {
  
  DBI::dbWriteTable(
    con,
    "structural_robustness_boundary_assignments",
    boundary_assignments,
    overwrite = overwrite
  )
  
  DBI::dbWriteTable(
    con,
    "structural_robustness_summary",
    robustness_summary,
    overwrite = overwrite
  )
  
  DBI::dbWriteTable(
    con,
    "structural_robustness_trajectories",
    trajectory_summary,
    overwrite = overwrite
  )
  
  invisible(TRUE)
}