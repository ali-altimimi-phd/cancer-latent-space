# ==============================================================================
# Structural Robustness Metrics
# ==============================================================================
#
# Purpose:
#   Helper functions for margin-aware structural robustness profiling.
#
#   These functions operate after structural concordance / validation and convert
#   quadrant assignments into geometric robustness diagnostics.
#
# Main concepts:
#   - boundary distance
#   - axis margin
#   - origin distance
#   - quadrant transition behavior
#   - chip/filter instability decomposition
#   - structural confidence class
#
# ==============================================================================


safe_scale_vector <- function(x) {
  if (all(is.na(x))) {
    return(rep(NA_real_, length(x)))
  }
  
  s <- stats::sd(x, na.rm = TRUE)
  
  if (is.na(s) || s == 0) {
    return(rep(0, length(x)))
  }
  
  as.numeric(scale(x))
}


infer_quadrant_from_xy <- function(x, y) {
  dplyr::case_when(
    is.na(x) | is.na(y) ~ NA_character_,
    x >= 0 & y >= 0 ~ "QI",
    x <  0 & y >= 0 ~ "QII",
    x <  0 & y <  0 ~ "QIII",
    x >= 0 & y <  0 ~ "QIV"
  )
}


add_boundary_distance_metrics <- function(df,
                                          x_col = "complexity_delta",
                                          y_col = "mp_spectral_entropy_delta",
                                          scale_within = c("global", "chip", "chip_filter")) {
  
  scale_within <- match.arg(scale_within)
  
  stopifnot(x_col %in% names(df))
  stopifnot(y_col %in% names(df))
  
  group_vars <- switch(
    scale_within,
    global = character(0),
    chip = "chip",
    chip_filter = c("chip", "filter_regime")
  )
  
  df %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(group_vars))) %>%
    dplyr::mutate(
      robustness_x = safe_scale_vector(.data[[x_col]]),
      robustness_y = safe_scale_vector(.data[[y_col]])
    ) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      robustness_quadrant = infer_quadrant_from_xy(robustness_x, robustness_y),
      
      distance_to_x_axis = abs(robustness_y),
      distance_to_y_axis = abs(robustness_x),
      distance_to_origin = sqrt(robustness_x^2 + robustness_y^2),
      
      quadrant_margin = pmin(abs(robustness_x), abs(robustness_y), na.rm = FALSE),
      
      x_sign = dplyr::case_when(
        is.na(robustness_x) ~ NA_character_,
        robustness_x >= 0 ~ "+",
        robustness_x < 0 ~ "-"
      ),
      
      y_sign = dplyr::case_when(
        is.na(robustness_y) ~ NA_character_,
        robustness_y >= 0 ~ "+",
        robustness_y < 0 ~ "-"
      )
    )
}

safe_mean <- function(x) {
  if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
}

safe_median <- function(x) {
  if (all(is.na(x))) NA_real_ else stats::median(x, na.rm = TRUE)
}

safe_max <- function(x) {
  if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE)
}

safe_min <- function(x) {
  if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE)
}

get_modal_quadrant <- function(x) {
  x <- x[!is.na(x)]
  
  if (length(x) == 0) {
    return(NA_character_)
  }
  
  names(sort(table(x), decreasing = TRUE))[1]
}

get_modal_fraction <- function(x) {
  x <- x[!is.na(x)]
  
  if (length(x) == 0) {
    return(NA_real_)
  }
  
  max(table(x)) / length(x)
}

summarise_margin_profile <- function(df) {
  df %>%
    dplyr::group_by(comparison) %>%
    dplyr::summarise(
      n_assignments = dplyr::n(),
      n_available_assignments = sum(!is.na(robustness_quadrant)),
      
      modal_quadrant = get_modal_quadrant(robustness_quadrant),
      modal_quadrant_fraction = get_modal_fraction(robustness_quadrant),
      
      mean_quadrant_margin = safe_mean(quadrant_margin),
      median_quadrant_margin = safe_median(quadrant_margin),
      min_quadrant_margin = safe_min(quadrant_margin),
      
      mean_distance_to_origin = safe_mean(distance_to_origin),
      median_distance_to_origin = safe_median(distance_to_origin),
      
      n_x_signs = dplyr::n_distinct(x_sign, na.rm = TRUE),
      n_y_signs = dplyr::n_distinct(y_sign, na.rm = TRUE),
      n_quadrants = dplyr::n_distinct(robustness_quadrant, na.rm = TRUE),
      
      .groups = "drop"
    )
}


compute_instability_decomposition <- function(df) {
  
  chip_instability <- df %>%
    dplyr::group_by(comparison, filter_regime) %>%
    dplyr::summarise(
      n_chip_quadrants = dplyr::n_distinct(robustness_quadrant, na.rm = TRUE),
      chip_instability = as.integer(n_chip_quadrants > 1),
      .groups = "drop"
    ) %>%
    dplyr::group_by(comparison) %>%
    dplyr::summarise(
      chip_instability_fraction = mean(chip_instability, na.rm = TRUE),
      .groups = "drop"
    )
  
  filter_instability <- df %>%
    dplyr::group_by(comparison, chip) %>%
    dplyr::summarise(
      n_filter_quadrants = dplyr::n_distinct(robustness_quadrant, na.rm = TRUE),
      filter_instability = as.integer(n_filter_quadrants > 1),
      .groups = "drop"
    ) %>%
    dplyr::group_by(comparison) %>%
    dplyr::summarise(
      filter_instability_fraction = mean(filter_instability, na.rm = TRUE),
      .groups = "drop"
    )
  
  sign_instability <- df %>%
    dplyr::group_by(comparison) %>%
    dplyr::summarise(
      x_sign_flips = as.integer(dplyr::n_distinct(x_sign, na.rm = TRUE) > 1),
      y_sign_flips = as.integer(dplyr::n_distinct(y_sign, na.rm = TRUE) > 1),
      both_axis_flips = as.integer(x_sign_flips == 1 & y_sign_flips == 1),
      .groups = "drop"
    )
  
  chip_instability %>%
    dplyr::full_join(filter_instability, by = "comparison") %>%
    dplyr::full_join(sign_instability, by = "comparison")
}


compute_cross_regime_trajectories <- function(df) {
  df %>%
    dplyr::arrange(comparison, chip, filter_regime) %>%
    dplyr::group_by(comparison, chip) %>%
    dplyr::summarise(
      n_regimes = dplyr::n(),
      
      trajectory_length = {
        x <- robustness_x
        y <- robustness_y
        
        if (length(x) < 2) {
          NA_real_
        } else {
          sum(sqrt(diff(x)^2 + diff(y)^2), na.rm = TRUE)
        }
      },
      
      net_displacement = {
        x <- robustness_x
        y <- robustness_y
        
        if (length(x) < 2) {
          NA_real_
        } else {
          sqrt((dplyr::last(x) - dplyr::first(x))^2 +
                 (dplyr::last(y) - dplyr::first(y))^2)
        }
      },
      
      boundary_crossing_count = sum(
        robustness_quadrant != dplyr::lag(robustness_quadrant),
        na.rm = TRUE
      ),
      
      quadrant_path = paste(robustness_quadrant, collapse = " -> "),
      
      .groups = "drop"
    ) %>%
    dplyr::group_by(comparison) %>%
    dplyr::summarise(
      mean_trajectory_length = safe_mean(trajectory_length),
      max_trajectory_length = safe_max(trajectory_length),
      
      mean_net_displacement = safe_mean(net_displacement),
      max_net_displacement = safe_max(net_displacement),
      
      total_boundary_crossings = sum(boundary_crossing_count, na.rm = TRUE),
      
      chip_quadrant_paths = paste(unique(quadrant_path[!is.na(quadrant_path)]), collapse = " | "),
      
      .groups = "drop"
    )
}


assign_margin_aware_class <- function(df,
                                      high_modal_cutoff = 0.80,
                                      high_margin_cutoff = 0.75,
                                      low_margin_cutoff = 0.25,
                                      weak_origin_cutoff = 0.50,
                                      high_trajectory_cutoff = 1.50) {
  
  df %>%
    dplyr::mutate(
      robustness_class = dplyr::case_when(
        is.na(modal_quadrant_fraction) ~
          "unavailable_structural_signal",
        
        mean_distance_to_origin < weak_origin_cutoff ~
          "weak_structural_signal",
        
        modal_quadrant_fraction >= high_modal_cutoff &
          mean_quadrant_margin >= high_margin_cutoff &
          max_trajectory_length < high_trajectory_cutoff ~
          "stable_core_archetype",
        
        modal_quadrant_fraction >= high_modal_cutoff &
          mean_quadrant_margin < high_margin_cutoff ~
          "stable_boundary_archetype",
        
        both_axis_flips == 1 ~
          "true_phase_flipper",
        
        x_sign_flips == 1 | y_sign_flips == 1 ~
          "single_axis_flipper",
        
        filter_instability_fraction > chip_instability_fraction ~
          "preprocessing_sensitive_manifold",
        
        chip_instability_fraction > filter_instability_fraction ~
          "platform_sensitive_manifold",
        
        mean_quadrant_margin <= low_margin_cutoff ~
          "boundary_rider",
        
        TRUE ~
          "mixed_or_intermediate"
      ),
      
      structural_confidence = dplyr::case_when(
        robustness_class == "stable_core_archetype" ~ "high",
        
        robustness_class %in% c(
          "stable_boundary_archetype",
          "boundary_rider",
          "mixed_or_intermediate"
        ) ~ "moderate",
        
        robustness_class %in% c(
          "unavailable_structural_signal",
          "weak_structural_signal"
        ) ~ "unavailable_or_weak",
        
        TRUE ~ "low"
      )
    )
}
