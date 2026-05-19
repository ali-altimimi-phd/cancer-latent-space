# ==============================================================================
# File: R/structural_concordance/compute_quadrant_assignments.R
# Purpose: Assign structural quadrants from canonical concordance table
# ==============================================================================

compute_quadrant_assignments <- function(canonical_tbl,
                                         x_metric = "complexity__effrank_delta",
                                         y_metric = "mp__spectral_entropy_delta",
                                         center_method = "zero",
                                         boundary_tolerance = 0,
                                         logger = NULL) {
  require_columns(
    canonical_tbl,
    c("chip", "filter_regime", "group", "comparison", x_metric, y_metric),
    table_label = "canonical concordance table"
  )
  
  sc_log(
    "🧭 Assigning quadrants using x = ", x_metric,
    " and y = ", y_metric, "...",
    logger = logger
  )
  
  x_raw <- canonical_tbl[[x_metric]]
  y_raw <- canonical_tbl[[y_metric]]
  
  x_z <- standardize_vector(x_raw, center_method = center_method)
  y_z <- standardize_vector(y_raw, center_method = center_method)
  
  quadrant <- assign_quadrant_label(
    x = x_z,
    y = y_z,
    tolerance = boundary_tolerance
  )
  
  canonical_tbl |>
    dplyr::mutate(
      quadrant_x_metric = x_metric,
      quadrant_y_metric = y_metric,
      quadrant_x_raw = x_raw,
      quadrant_y_raw = y_raw,
      quadrant_x_z = x_z,
      quadrant_y_z = y_z,
      quadrant = quadrant,
      quadrant_interpretation = default_quadrant_interpretation(quadrant),
      quadrant_x_state = signed_state(x_z, tolerance = boundary_tolerance),
      quadrant_y_state = signed_state(y_z, tolerance = boundary_tolerance),
      distance_from_origin_raw = sqrt(x_raw^2 + y_raw^2),
      distance_from_origin_z = sqrt(x_z^2 + y_z^2),
      distance_from_boundary_raw = pmin(abs(x_raw), abs(y_raw)),
      distance_from_boundary_z = pmin(abs(x_z), abs(y_z)),
      quadrant_confidence = dplyr::case_when(
        is.na(quadrant) ~ NA_character_,
        quadrant == "boundary" ~ "boundary",
        distance_from_boundary_z >= 1 ~ "high",
        distance_from_boundary_z >= 0.5 ~ "moderate",
        TRUE ~ "low"
      )
    ) |>
    dplyr::select(
      chip,
      filter_regime,
      group,
      comparison,
      quadrant,
      quadrant_interpretation,
      quadrant_confidence,
      quadrant_x_metric,
      quadrant_y_metric,
      quadrant_x_raw,
      quadrant_y_raw,
      quadrant_x_z,
      quadrant_y_z,
      quadrant_x_state,
      quadrant_y_state,
      distance_from_origin_raw,
      distance_from_origin_z,
      distance_from_boundary_raw,
      distance_from_boundary_z,
      dplyr::everything()
    )
}