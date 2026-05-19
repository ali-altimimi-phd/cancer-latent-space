# ==============================================================================
# File: R/structural_concordance/helpers_structural_concordance.R
# Purpose: Shared helpers for structural concordance and quadrant assignment
# Project: Global Cancer Complexity / Structural Inference
# ------------------------------------------------------------------------------
# Notes:
#   - These helpers operate downstream of the structural synthesis layer.
#   - They assume concordance inputs are already harmonized by the synthesis
#     script or by DuckDB synthesis views.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
  library(rlang)
})

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

sc_log <- function(..., logger = NULL) {
  msg <- paste0(...)
  if (!is.null(logger) && is.function(logger$log)) {
    logger$log(msg)
  } else {
    message(msg)
  }
}

require_columns <- function(.data, required_cols, table_label = "input table") {
  missing_cols <- setdiff(required_cols, names(.data))
  if (length(missing_cols) > 0) {
    stop(
      table_label, " is missing required columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

optional_numeric_cols <- function(.data, cols) {
  cols[cols %in% names(.data) & vapply(.data[cols], is.numeric, logical(1))]
}

safe_divide <- function(x, y) {
  out <- rep(NA_real_, length(x))
  ok <- !is.na(x) & !is.na(y) & y != 0
  out[ok] <- x[ok] / y[ok]
  out
}

standardize_vector <- function(x, center_method = c("zero", "mean", "median")) {
  center_method <- match.arg(center_method)

  center <- switch(
    center_method,
    zero = 0,
    mean = mean(x, na.rm = TRUE),
    median = stats::median(x, na.rm = TRUE)
  )

  scale_value <- stats::sd(x, na.rm = TRUE)
  if (is.na(scale_value) || scale_value == 0) {
    scale_value <- 1
  }

  (x - center) / scale_value
}

signed_state <- function(x, tolerance = 0) {
  out <- rep(NA_character_, length(x))
  
  out[!is.na(x) & x > tolerance] <- "positive"
  out[!is.na(x) & x < -tolerance] <- "negative"
  out[!is.na(x) & abs(x) <= tolerance] <- "near_zero"
  
  out
}

assign_quadrant_label <- function(x, y, tolerance = 0) {
  out <- rep(NA_character_, length(x))
  
  valid <- !is.na(x) & !is.na(y)
  
  out[valid & (abs(x) <= tolerance | abs(y) <= tolerance)] <- "boundary"
  out[valid & x > tolerance & y > tolerance] <- "I"
  out[valid & x < -tolerance & y > tolerance] <- "II"
  out[valid & x < -tolerance & y < -tolerance] <- "III"
  out[valid & x > tolerance & y < -tolerance] <- "IV"
  
  out
}

default_quadrant_interpretation <- function(quadrant) {
  dplyr::case_when(
    quadrant == "I" ~ "coordinated structural expansion",
    quadrant == "II" ~ "entropy-dominant disorganization with reduced complexity",
    quadrant == "III" ~ "coordinated structural compression",
    quadrant == "IV" ~ "complexity-dominant constrained reorganization",
    quadrant == "boundary" ~ "near quadrant boundary",
    TRUE ~ NA_character_
  )
}

majority_label <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) {
    return(NA_character_)
  }
  tab <- sort(table(x), decreasing = TRUE)
  names(tab)[1]
}

majority_fraction <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) {
    return(NA_real_)
  }
  max(table(x)) / length(x)
}

create_output_dirs <- function(output_dir) {
  dirs <- c(
    output_dir,
    file.path(output_dir, "tables"),
    file.path(output_dir, "RData")
  )

  for (dir_i in dirs) {
    if (!dir.exists(dir_i)) {
      dir.create(dir_i, recursive = TRUE, showWarnings = FALSE)
    }
  }

  invisible(dirs)
}

write_concordance_artifacts <- function(tables,
                                        output_dir,
                                        write_csv = TRUE,
                                        write_rds = TRUE,
                                        logger = NULL) {
  create_output_dirs(output_dir)

  if (isTRUE(write_csv)) {
    purrr::iwalk(
      tables,
      function(tbl, nm) {
        path <- file.path(output_dir, "tables", paste0(nm, ".csv"))
        readr::write_csv(tbl, path)
        sc_log("💾 Wrote ", path, logger = logger)
      }
    )
  }

  if (isTRUE(write_rds)) {
    path <- file.path(output_dir, "RData", "structural_concordance_tables.rds")
    saveRDS(tables, path)
    sc_log("💾 Wrote ", path, logger = logger)
  }

  invisible(TRUE)
}

materialize_concordance_tables <- function(con,
                                           tables,
                                           table_prefix = "structural",
                                           logger = NULL) {
  stopifnot(!is.null(con))

  name_map <- c(
    canonical = paste0(table_prefix, "_concordance_canonical"),
    quadrant_assignments = paste0(table_prefix, "_quadrant_assignments"),
    engine_correlations = paste0(table_prefix, "_engine_correlations"),
    quadrant_stability = paste0(table_prefix, "_quadrant_stability"),
    concordance_inventory = paste0(table_prefix, "_concordance_inventory")
  )

  for (nm in intersect(names(name_map), names(tables))) {
    DBI::dbWriteTable(
      con,
      name = unname(name_map[[nm]]),
      value = tables[[nm]],
      overwrite = TRUE
    )
    sc_log("🦆 Materialized DuckDB table: ", unname(name_map[[nm]]), logger = logger)
  }

  invisible(name_map)
}
