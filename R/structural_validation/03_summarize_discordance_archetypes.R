#' Summarize Discordance Archetypes
#'
#' Classifies directional discordance into interpretable structural archetypes
#' and builds DuckDB summary views for inspection and reporting.

suppressPackageStartupMessages({
  library(DBI)
  library(dplyr)
  library(readr)
})

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------

summarize_discordance_archetypes <- function(
    con,
    output_dir,
    source_table = "cross_engine_directional_concordance"
) {

  message("Building cross-engine discordance archetype table...")

  concordance_df <- DBI::dbGetQuery(con, paste0("SELECT * FROM ", source_table))

  archetype_df <- concordance_df |>
    dplyr::mutate(
      discordance_archetype = dplyr::case_when(
        n_engines_available == 0 ~
          "insufficient_data",

        discordance_class == "full_agreement" ~
          "full_cross_engine_agreement",

        n_engines_available < 4 ~
          "missing_engine_overlay",

        complexity_direction == entropy_direction &
          !is.na(complexity_direction) &
          mp_direction != complexity_direction &
          latent_direction != complexity_direction ~
          "complexity_entropy_vs_mp_latent_bifurcation",

        complexity_direction == entropy_direction &
          !is.na(complexity_direction) &
          mp_direction != complexity_direction ~
          "mp_divergence_from_classical_spectrum",

        complexity_direction == entropy_direction &
          !is.na(complexity_direction) &
          latent_direction != complexity_direction ~
          "latent_divergence_from_classical_spectrum",

        complexity_direction != entropy_direction &
          !is.na(complexity_direction) &
          !is.na(entropy_direction) ~
          "complexity_entropy_internal_conflict",

        TRUE ~
          "mixed_or_unclassified_discordance"
      )
    ) |>
    dplyr::arrange(discordance_class, discordance_archetype, comparison, chip, filter_regime)

  DBI::dbWriteTable(
    con,
    "cross_engine_discordance_archetypes",
    archetype_df,
    overwrite = TRUE
  )

  readr::write_csv(
    archetype_df,
    file.path(output_dir, "cross_engine_discordance_archetypes.csv")
  )

  # --------------------------------------------------------------------------
  # Summary views
  # --------------------------------------------------------------------------

  DBI::dbExecute(con, "
    CREATE OR REPLACE VIEW vw_cross_engine_discordance_summary AS
    SELECT
      comparison,
      COUNT(*) AS n_runs,
      AVG(engine_sign_agreement_fraction) AS mean_agreement_fraction,
      MIN(engine_sign_agreement_fraction) AS min_agreement_fraction,
      SUM(CASE WHEN discordance_class != 'full_agreement' THEN 1 ELSE 0 END) AS n_discordant_runs
    FROM cross_engine_directional_concordance
    GROUP BY comparison
    ORDER BY mean_agreement_fraction ASC, n_discordant_runs DESC
  ")

  DBI::dbExecute(con, "
    CREATE OR REPLACE VIEW vw_cross_engine_archetype_summary AS
    SELECT
      comparison,
      discordance_archetype,
      COUNT(*) AS n
    FROM cross_engine_discordance_archetypes
    GROUP BY comparison, discordance_archetype
    ORDER BY comparison, n DESC
  ")

  DBI::dbExecute(con, "
    CREATE OR REPLACE VIEW vw_cross_engine_discordant_runs AS
    SELECT *
    FROM cross_engine_discordance_archetypes
    WHERE discordance_class != 'full_agreement'
    ORDER BY engine_sign_agreement_fraction ASC, comparison, chip, filter_regime
  ")

  summary_df <- DBI::dbGetQuery(con, "SELECT * FROM vw_cross_engine_discordance_summary")
  archetype_summary_df <- DBI::dbGetQuery(con, "SELECT * FROM vw_cross_engine_archetype_summary")
  discordant_runs_df <- DBI::dbGetQuery(con, "SELECT * FROM vw_cross_engine_discordant_runs")

  readr::write_csv(
    summary_df,
    file.path(output_dir, "vw_cross_engine_discordance_summary.csv")
  )

  readr::write_csv(
    archetype_summary_df,
    file.path(output_dir, "vw_cross_engine_archetype_summary.csv")
  )

  readr::write_csv(
    discordant_runs_df,
    file.path(output_dir, "vw_cross_engine_discordant_runs.csv")
  )

  message("Wrote table: cross_engine_discordance_archetypes")
  message("Created views: vw_cross_engine_discordance_summary, vw_cross_engine_archetype_summary, vw_cross_engine_discordant_runs")

  invisible(archetype_df)
}
