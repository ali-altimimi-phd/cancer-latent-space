# ==============================================================================
# Compute spectral deltas
# ==============================================================================
#
# Purpose:
#   Convert normal/tumor spectral summaries into paired tumor-minus-normal deltas.
#
# ==============================================================================

#' Compute normal-tumor spectral deltas
#'
#' @param spectral_summary Tibble containing normal/tumor spectral summaries.
#' @return Tibble with one row per chip/group/comparison/filter method.
compute_spectral_deltas <- function(spectral_summary) {
  spectral_summary |>
    dplyr::select(
      chip,
      group,
      comparison,
      filter_regime,
      condition,
      matrix_key,
      n_samples,
      n_features,
      q,
      mp_lambda_plus,
      largest_eigenvalue,
      largest_eigenvalue_fraction,
      n_spikes,
      spike_fraction,
      excess_spectral_mass,
      spectral_entropy,
      participation_ratio,
      status
    ) |>
    tidyr::pivot_wider(
      names_from = condition,
      values_from = c(
        matrix_key,
        n_samples,
        n_features,
        q,
        mp_lambda_plus,
        largest_eigenvalue,
        largest_eigenvalue_fraction,
        n_spikes,
        spike_fraction,
        excess_spectral_mass,
        spectral_entropy,
        participation_ratio,
        status
      )
    ) |>
    dplyr::mutate(
      paired_status = dplyr::case_when(
        status_normal == "ok" & status_tumor == "ok" ~ "ok",
        status_normal != "ok" & status_tumor == "ok" ~ "normal_skipped",
        status_normal == "ok" & status_tumor != "ok" ~ "tumor_skipped",
        status_normal != "ok" & status_tumor != "ok" ~ "both_skipped",
        TRUE ~ "unknown"
      ),
      largest_eigenvalue_delta =
        largest_eigenvalue_tumor - largest_eigenvalue_normal,

      largest_eigenvalue_fraction_delta =
        largest_eigenvalue_fraction_tumor - largest_eigenvalue_fraction_normal,

      n_spikes_delta =
        n_spikes_tumor - n_spikes_normal,

      spike_fraction_delta =
        spike_fraction_tumor - spike_fraction_normal,

      excess_spectral_mass_delta =
        excess_spectral_mass_tumor - excess_spectral_mass_normal,

      spectral_entropy_delta =
        spectral_entropy_tumor - spectral_entropy_normal,

      participation_ratio_delta =
        participation_ratio_tumor - participation_ratio_normal
    )
}
