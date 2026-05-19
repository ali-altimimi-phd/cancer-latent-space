# ==============================================================================
# File: R/structural_concordance/build_structural_concordance_tables.R
# Purpose: Build canonical structural concordance, quadrant, correlation, and
#          stability tables from DuckDB structural synthesis views.
# Project: Global Cancer Complexity / Structural Inference
# ------------------------------------------------------------------------------
# Notes:
#   - This is a downstream layer. It does not read raw engine RDS files.
#   - Preferred input is vw_structural_phenotype_wide_with_latent_overlay.
#   - The output is designed to become the computational bridge between
#     structural synthesis and biological interpretation.
# ==============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(dplyr)
  library(readr)
  library(tibble)
  library(purrr)
})

source_if_needed <- function(path) {
  if (file.exists(path)) {
    source(path)
  }
}

#' Build structural concordance tables
#'
#' @param db_path Path to the DuckDB warehouse.
#' @param output_dir Output directory for concordance tables.
#' @param synthesis_view DuckDB view/table used as canonical synthesis input.
#' @param x_metric Structural quadrant x-axis metric.
#' @param y_metric Structural quadrant y-axis metric.
#' @param center_method How to center quadrant axes for standardized distances.
#' @param boundary_tolerance Raw-value tolerance used to call a point boundary-like.
#' @param correlation_method Correlation method for metric concordance.
#' @param min_complete Minimum complete pairs required for a metric correlation.
#' @param materialize_to_duckdb If TRUE, writes concordance tables back to DuckDB.
#' @param write_csv If TRUE, writes CSV tables.
#' @param write_rds If TRUE, writes an RDS list of all concordance tables.
#' @param logger Optional pipeline logger with a $log method.
#'
#' @return Invisibly returns a named list of concordance tables.
build_structural_concordance_tables <- function(
    db_path = file.path(
      "output",
      "global_cancer",
      "warehouse",
      "global_cancer_results.duckdb"
    ),
    output_dir = file.path(
      "output",
      "global_cancer",
      "structural_inference",
      "tables",
      "concordance"
    ),
    synthesis_view = "vw_structural_phenotype_wide_with_latent_overlay",
    x_metric = "complexity__effrank_delta",
    y_metric = "mp__spectral_entropy_delta",
    center_method = "zero",
    boundary_tolerance = 0,
    correlation_method = "spearman",
    min_complete = 5,
    materialize_to_duckdb = TRUE,
    write_csv = TRUE,
    write_rds = TRUE,
    logger = NULL) {

  source_if_needed(file.path("R", "structural_concordance", "helpers_structural_concordance.R"))
  source_if_needed(file.path("R", "structural_concordance", "compute_quadrant_assignments.R"))
  source_if_needed(file.path("R", "structural_concordance", "compute_engine_correlations.R"))
  source_if_needed(file.path("R", "structural_concordance", "compute_quadrant_stability.R"))

  if (!file.exists(db_path)) {
    stop("DuckDB warehouse not found: ", db_path, call. = FALSE)
  }

  sc_log("🦆 Connecting to DuckDB warehouse: ", db_path, logger = logger)
  con <- DBI::dbConnect(duckdb::duckdb(), dbdir = db_path, read_only = FALSE)
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)

  available_tables <- DBI::dbListTables(con)
  if (!synthesis_view %in% available_tables) {
    stop(
      "Required synthesis view/table not found: ", synthesis_view,
      "\nAvailable tables/views include: ",
      paste(utils::head(available_tables, 20), collapse = ", "),
      call. = FALSE
    )
  }

  sc_log("📥 Reading synthesis view: ", synthesis_view, logger = logger)
  synthesis_tbl <- DBI::dbGetQuery(con, paste0("SELECT * FROM ", synthesis_view)) |>
    tibble::as_tibble()

  require_columns(
    synthesis_tbl,
    c("chip", "filter_regime", "group", "comparison"),
    table_label = synthesis_view
  )

  metric_candidates <- c(
    "complexity__kappa_delta",
    "complexity__effrank_delta",
    "complexity__sparsity_delta",
    "complexity__composite_kappa_delta",
    "entropy__shannon_delta",
    "entropy__spectral_delta",
    "mp__spectral_entropy_delta",
    "mp__participation_ratio_delta",
    "mp__largest_eigenvalue_fraction_delta",
    "mp__excess_spectral_mass_delta",
    "mp__n_spikes_delta",
    "mp__spike_fraction_delta",
    "latent__pr_delta",
    "latent__eig_entropy_delta",
    "latent__centroid_distance",
    "latent__anisotropy_delta",
    "latent__radius_delta"
  )

  available_metric_cols <- metric_candidates[metric_candidates %in% names(synthesis_tbl)]

  canonical_tbl <- synthesis_tbl |>
    select(
      chip,
      filter_regime,
      group,
      comparison,
      any_of(c("latent__model_id", "latent__chip", "latent__filter_regime")),
      any_of(available_metric_cols)
    )
  
  metric_availability_tbl <- canonical_tbl |>
    select(any_of(available_metric_cols))
  
  canonical_tbl <- canonical_tbl |>
    mutate(
      n_structural_metrics_available = if (ncol(metric_availability_tbl) == 0) {
        0L
      } else {
        rowSums(!is.na(metric_availability_tbl))
      }
    )
  
  # Add latent-only metrics not present in the overlay view, when available.
  if ("latent_comparison_metrics_fact" %in% available_tables) {
    latent_cols <- DBI::dbListFields(con, "latent_comparison_metrics_fact")

    supplemental_latent_cols <- c(
      "chip",
      "filter_regime",
      "comparison",
      "latent_model_id",
      "anisotropy_delta",
      "radius_delta"
    )

    if (all(supplemental_latent_cols %in% latent_cols)) {
      latent_supp <- DBI::dbGetQuery(
        con,
        paste(
          "SELECT chip AS latent_join_chip,",
          "filter_regime AS latent_join_filter_regime,",
          "comparison,",
          "latent_model_id AS latent__model_id_supplemental,",
          "anisotropy_delta AS latent__anisotropy_delta_supplemental,",
          "radius_delta AS latent__radius_delta_supplemental",
          "FROM latent_comparison_metrics_fact"
        )
      ) |>
        tibble::as_tibble()

      if (!"latent__model_id" %in% names(canonical_tbl)) {
        canonical_tbl$latent__model_id <- NA_character_
      }
      if (!"latent__anisotropy_delta" %in% names(canonical_tbl)) {
        canonical_tbl$latent__anisotropy_delta <- NA_real_
      }
      if (!"latent__radius_delta" %in% names(canonical_tbl)) {
        canonical_tbl$latent__radius_delta <- NA_real_
      }

      canonical_tbl <- canonical_tbl |>
        left_join(latent_supp, by = "comparison") |>
        mutate(
          latent__model_id = dplyr::coalesce(
            latent__model_id,
            latent__model_id_supplemental
          ),
          latent__anisotropy_delta = dplyr::coalesce(
            latent__anisotropy_delta,
            latent__anisotropy_delta_supplemental
          ),
          latent__radius_delta = dplyr::coalesce(
            latent__radius_delta,
            latent__radius_delta_supplemental
          )
        ) |>
        select(
          -any_of(c(
            "latent_join_chip",
            "latent_join_filter_regime",
            "latent__model_id_supplemental",
            "latent__anisotropy_delta_supplemental",
            "latent__radius_delta_supplemental"
          ))
        )
    }
  }

  structural_direction_metrics <- c(
    complexity = "complexity__effrank_delta",
    entropy = "entropy__spectral_delta",
    mp = "mp__spectral_entropy_delta",
    latent = "latent__eig_entropy_delta"
  )

  available_direction_metrics <- structural_direction_metrics[
    structural_direction_metrics %in% names(canonical_tbl)
  ]

  if (length(available_direction_metrics) > 0) {
    direction_tbl <- canonical_tbl |>
      transmute(
        chip,
        filter_regime,
        group,
        comparison,
        !!!setNames(
          lapply(
            available_direction_metrics,
            function(col_i) signed_state(canonical_tbl[[col_i]])
          ),
          paste0(names(available_direction_metrics), "_direction")
        )
      )

    direction_cols <- setdiff(names(direction_tbl), c("chip", "filter_regime", "group", "comparison"))

    direction_tbl <- direction_tbl |>
      rowwise() |>
      mutate(
        n_direction_metrics_available = sum(!is.na(c_across(all_of(direction_cols)))),
        n_positive_direction_metrics = sum(c_across(all_of(direction_cols)) == "positive", na.rm = TRUE),
        n_negative_direction_metrics = sum(c_across(all_of(direction_cols)) == "negative", na.rm = TRUE),
        n_near_zero_direction_metrics = sum(c_across(all_of(direction_cols)) == "near_zero", na.rm = TRUE),
        engine_sign_majority = case_when(
          n_direction_metrics_available == 0 ~ NA_character_,
          n_positive_direction_metrics >= n_negative_direction_metrics &
            n_positive_direction_metrics >= n_near_zero_direction_metrics ~ "positive",
          n_negative_direction_metrics >= n_positive_direction_metrics &
            n_negative_direction_metrics >= n_near_zero_direction_metrics ~ "negative",
          TRUE ~ "near_zero"
        ),
        engine_sign_agreement_fraction = safe_divide(
          pmax(
            n_positive_direction_metrics,
            n_negative_direction_metrics,
            n_near_zero_direction_metrics
          ),
          n_direction_metrics_available
        )
      ) |>
      ungroup()

    canonical_tbl <- canonical_tbl |>
      left_join(
        direction_tbl,
        by = c("chip", "filter_regime", "group", "comparison")
      )
  }

  quadrant_tbl <- compute_quadrant_assignments(
    canonical_tbl = canonical_tbl,
    x_metric = x_metric,
    y_metric = y_metric,
    center_method = center_method,
    boundary_tolerance = boundary_tolerance,
    logger = logger
  )

  correlation_tbl <- compute_engine_correlations(
    canonical_tbl = canonical_tbl,
    method = correlation_method,
    min_complete = min_complete,
    logger = logger
  )

  stability_tbl <- compute_quadrant_stability(
    quadrant_tbl = quadrant_tbl,
    logger = logger
  )

  inventory_tbl <- tibble(
    table_name = c(
      "canonical",
      "quadrant_assignments",
      "engine_correlations",
      "quadrant_stability"
    ),
    n_rows = c(
      nrow(canonical_tbl),
      nrow(quadrant_tbl),
      nrow(correlation_tbl),
      nrow(stability_tbl)
    ),
    source = synthesis_view,
    created_at = as.character(Sys.time())
  )

  tables <- list(
    canonical = canonical_tbl,
    quadrant_assignments = quadrant_tbl,
    engine_correlations = correlation_tbl,
    quadrant_stability = stability_tbl,
    concordance_inventory = inventory_tbl
  )

  write_concordance_artifacts(
    tables = tables,
    output_dir = output_dir,
    write_csv = write_csv,
    write_rds = write_rds,
    logger = logger
  )

  if (isTRUE(materialize_to_duckdb)) {
    materialize_concordance_tables(
      con = con,
      tables = tables,
      table_prefix = "structural",
      logger = logger
    )
  }

  sc_log("✅ Structural concordance build complete.", logger = logger)

  invisible(tables)
}
