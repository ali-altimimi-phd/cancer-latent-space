#' Build Engine Discordance Map
#'
#' Converts canonical engine deltas into directional sign calls and quantifies
#' cross-engine agreement within each chip/filter/comparison combination.

suppressPackageStartupMessages({
  library(DBI)
  library(dplyr)
  library(tibble)
  library(readr)
})

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

metric_eps <- function(x, eps_mode = "fixed", eps_fixed = 1e-8, eps_scaled_factor = 0.10) {
  eps_mode <- match.arg(eps_mode, c("fixed", "scaled"))

  if (eps_mode == "fixed") {
    return(eps_fixed)
  }

  sx <- stats::sd(x, na.rm = TRUE)

  if (is.na(sx) || sx == 0) {
    return(eps_fixed)
  }

  eps_scaled_factor * sx
}

sign_call <- function(x, eps) {
  dplyr::case_when(
    is.na(x) ~ NA_character_,
    x > eps ~ "positive",
    x < -eps ~ "negative",
    TRUE ~ "zero"
  )
}

require_columns <- function(df, cols) {
  missing <- setdiff(cols, names(df))

  if (length(missing) > 0) {
    stop(
      "Required columns are missing from source view: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
}

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------

build_engine_discordance_map <- function(
    con,
    output_dir,
    source_view = "vw_structural_phenotype_wide_with_latent_overlay",
    eps_mode = "fixed",
    eps_fixed = 1e-8,
    eps_scaled_factor = 0.10,
    invert_latent_anisotropy = TRUE
) {

  message("Building cross-engine directional concordance table...")

  df <- DBI::dbGetQuery(con, paste0("SELECT * FROM ", source_view))

  id_cols <- c("chip", "filter_regime", "comparison", "group")

  metric_cols <- c(
    "complexity__effrank_delta",
    "entropy__spectral_delta",
    "mp__spectral_entropy_delta"
  )

  require_columns(df, c(id_cols, metric_cols))

  latent_available <- "latent__anisotropy_delta" %in% names(df)

  if (!latent_available) {
    warning(
      "latent__anisotropy_delta not found. Latent direction will be set to NA.",
      call. = FALSE
    )
    df$latent__anisotropy_delta <- NA_real_
  }

  eps_complexity <- metric_eps(
    df$complexity__effrank_delta,
    eps_mode = eps_mode,
    eps_fixed = eps_fixed,
    eps_scaled_factor = eps_scaled_factor
  )

  eps_entropy <- metric_eps(
    df$entropy__spectral_delta,
    eps_mode = eps_mode,
    eps_fixed = eps_fixed,
    eps_scaled_factor = eps_scaled_factor
  )

  eps_mp <- metric_eps(
    df$mp__spectral_entropy_delta,
    eps_mode = eps_mode,
    eps_fixed = eps_fixed,
    eps_scaled_factor = eps_scaled_factor
  )

  latent_metric_for_direction <- if (invert_latent_anisotropy) {
    -df$latent__anisotropy_delta
  } else {
    df$latent__anisotropy_delta
  }

  eps_latent <- metric_eps(
    latent_metric_for_direction,
    eps_mode = eps_mode,
    eps_fixed = eps_fixed,
    eps_scaled_factor = eps_scaled_factor
  )

  direction_df <- df |>
    dplyr::mutate(
      complexity_direction = sign_call(complexity__effrank_delta, eps_complexity),
      entropy_direction = sign_call(entropy__spectral_delta, eps_entropy),
      mp_direction = sign_call(mp__spectral_entropy_delta, eps_mp),
      latent_direction = sign_call(latent_metric_for_direction, eps_latent),
      direction_eps_mode = eps_mode,
      eps_complexity = eps_complexity,
      eps_entropy = eps_entropy,
      eps_mp = eps_mp,
      eps_latent = eps_latent,
      latent_anisotropy_inverted = invert_latent_anisotropy
    ) |>
    dplyr::select(
      dplyr::all_of(id_cols),
      complexity_direction,
      entropy_direction,
      mp_direction,
      latent_direction,
      direction_eps_mode,
      eps_complexity,
      eps_entropy,
      eps_mp,
      eps_latent,
      latent_anisotropy_inverted
    )

  concordance_df <- direction_df |>
    dplyr::rowwise() |>
    dplyr::mutate(
      directions = list(stats::na.omit(c(
        complexity_direction,
        entropy_direction,
        mp_direction,
        latent_direction
      ))),
      n_engines_available = length(directions),
      n_positive = sum(directions == "positive"),
      n_negative = sum(directions == "negative"),
      n_zero = sum(directions == "zero"),
      modal_direction = dplyr::case_when(
        n_engines_available == 0 ~ NA_character_,
        n_positive > n_negative & n_positive >= n_zero ~ "positive",
        n_negative > n_positive & n_negative >= n_zero ~ "negative",
        n_zero > n_positive & n_zero > n_negative ~ "zero",
        n_positive == n_negative & n_positive > 0 ~ "split",
        TRUE ~ "mixed"
      ),
      engine_sign_agreement_fraction = dplyr::case_when(
        n_engines_available == 0 ~ NA_real_,
        TRUE ~ max(n_positive, n_negative, n_zero) / n_engines_available
      ),
      discordance_class = dplyr::case_when(
        is.na(engine_sign_agreement_fraction) ~ "insufficient_data",
        engine_sign_agreement_fraction == 1 ~ "full_agreement",
        engine_sign_agreement_fraction >= 0.75 ~ "minor_discordance",
        engine_sign_agreement_fraction >= 0.50 ~ "major_discordance",
        TRUE ~ "severe_discordance"
      )
    ) |>
    dplyr::ungroup() |>
    dplyr::select(-directions) |>
    dplyr::arrange(engine_sign_agreement_fraction, comparison, chip, filter_regime)

  DBI::dbWriteTable(
    con,
    "cross_engine_directional_concordance",
    concordance_df,
    overwrite = TRUE
  )

  readr::write_csv(
    concordance_df,
    file.path(output_dir, "cross_engine_directional_concordance.csv")
  )

  message("Wrote table: cross_engine_directional_concordance")

  invisible(concordance_df)
}
