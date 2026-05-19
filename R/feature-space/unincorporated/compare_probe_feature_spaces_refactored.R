# ------------------------------------------------------------------------------
# File: compare_probe_feature_spaces.R
# Purpose: Compare analysis-selected top-N probe sets with latent-space features
# Role: Standalone diagnostic utility
# Pipeline: Standalone / Analysis diagnostics
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Feature-space overlap diagnostic
#'
#' @description
#' Compares comparison-specific analysis probe sets with the global probe set used
#' for latent-space / machine-learning input. This diagnostic is intended to test
#' whether the complexity/entropy analysis feature spaces and the latent feature
#' space are operating on similar probe subsets.
#'
#' IMPORTANT: Under the current project design, this comparison is scientifically
#' valid only when the analysis filtered-probe object was generated using
#' variance-based fixed-count selection:
#'
#'   filter_method = "variance"
#'   variance_selection_mode = "top_n"
#'   analysis_top_n = expected_top_n
#'
#' The latent-space feature set is currently implemented only for the hu35ksuba
#' chip in this workflow. The function therefore validates chip identity and
#' filtering provenance before computing overlap statistics.
#'
#' NOTE (Future Pipeline Integration)
#'
#' This script is intentionally kept as a standalone diagnostic utility for now.
#' Future integration as an optional analysis-pipeline stage should occur only
#' after:
#'   (1) latent-space preprocessing is generalized across supported chips;
#'   (2) fixed-count top-N analysis filtering is formally supported as a stable
#'       pipeline mode; and
#'   (3) downstream Quarto reports are updated to conditionally include this
#'       diagnostic only when its provenance requirements are satisfied.
#'
#' @param filtered_results Nested filtered-probe result object produced by
#'   `run_grouped_probe_filtering()` and containing `__metadata__`.
#' @param latent_features_path Path to CSV containing latent-space feature list.
#'   Must contain a `probe_id` column.
#' @param expected_chip_id Character chip/platform identifier currently supported
#'   by the latent-space feature file. Default is `"hu35ksuba"`.
#' @param expected_top_n Integer. Required top-N probe count for both analysis
#'   and latent feature spaces under the current diagnostic design.
#' @param output_csv Optional path for CSV output.
#' @param output_rds Optional path for RDS output.
#' @param include_probe_ids Logical. If `TRUE`, include semicolon-delimited
#'   overlapping probe IDs in the output table. Defaults to `FALSE` to avoid
#'   unnecessarily large output files.
#'
#' @return A tibble with one row per comparison and overlap statistics.
#' @export
compare_probe_feature_spaces <- function(filtered_results,
                                         latent_features_path,
                                         expected_chip_id = "hu35ksuba",
                                         expected_top_n = 3000L,
                                         output_csv = NULL,
                                         output_rds = NULL,
                                         include_probe_ids = FALSE) {
  # --------------------------------------------------------------------------
  # VALIDATION: Filtered-probe object provenance
  # --------------------------------------------------------------------------

  meta <- filtered_results$`__metadata__`

  # --------------------------------------------------------------------------
  # NOTE (Metadata Consistency – Future Fix)
  #
  # When filter_method = "limma", the field variance_selection_mode may still
  # contain values such as "top_n" because it is propagated from pipeline
  # configuration rather than reflecting the actual filtering method used.
  #
  # This is semantically misleading, since variance-based selection is not
  # applied in limma filtering and no top_n parameter is operative.
  #
  # In a future refactor, variance_selection_mode should be set to NULL (or
  # "not_applicable") whenever method != "variance", and top_n should likewise
  # be NULL in such cases.
  #
  # The current validation logic correctly ignores limma-based objects, but
  # this inconsistency may affect interpretability of metadata in diagnostics.
  # --------------------------------------------------------------------------
  
  if (is.null(meta)) {
    stop(
      "Filtered-probe object has no __metadata__ field. ",
      "Rerun the analysis pipeline after updating run_grouped_probe_filtering.R ",
      "to save provenance metadata.",
      call. = FALSE
    )
  }

  if (!identical(meta$chip_id, expected_chip_id)) {
    stop(
      sprintf(
        "This diagnostic currently supports only chip_id = '%s'. Found: '%s'.",
        expected_chip_id,
        meta$chip_id
      ),
      call. = FALSE
    )
  }

  valid_top_n <- !is.null(meta$parameters$top_n) &&
    as.integer(meta$parameters$top_n) == as.integer(expected_top_n)

  valid_top_n <- !is.null(meta$parameters$top_n) &&
    as.integer(meta$parameters$top_n) == as.integer(expected_top_n)
  
  is_valid_feature_space_object <-
    identical(meta$chip_id, expected_chip_id) &&
    identical(meta$method, "variance") &&
    identical(meta$variance_selection_mode, "top_n") &&
    valid_top_n
  
  if (!is_valid_feature_space_object) {
    message(
      paste0(
        "\nSkipping feature-space comparison.\n\n",
        "Reason: analysis filtered-probe object was not generated with:\n",
        "  filter_method = 'variance'\n",
        "  variance_selection_mode = 'top_n'\n",
        "  analysis_top_n = ", expected_top_n, "\n\n",
        "Current object metadata:\n",
        "  chip_id = ", meta$chip_id, "\n",
        "  method = ", meta$method, "\n",
        "  variance_selection_mode = ", meta$variance_selection_mode, "\n",
        "  top_n = ", meta$parameters$top_n, "\n\n",
        "To run this diagnostic, rerun the analysis pipeline with variance filtering ",
        "and ", expected_top_n, " top probes.\n"
      )
    )
    
    return(invisible(NULL))
  }

  # --------------------------------------------------------------------------
  # VALIDATION: Latent feature file
  # --------------------------------------------------------------------------

  if (!file.exists(latent_features_path)) {
    stop(sprintf("Latent feature file not found: %s", latent_features_path),
         call. = FALSE)
  }

  latent_features <- readr::read_csv(latent_features_path, show_col_types = FALSE)

  if (!"probe_id" %in% names(latent_features)) {
    stop("Latent feature file must contain a 'probe_id' column.", call. = FALSE)
  }

  latent_probes <- unique(as.character(latent_features$probe_id))
  latent_probes <- latent_probes[!is.na(latent_probes)]

  if (length(latent_probes) == 0L) {
    stop("No valid latent probes found in latent feature file.", call. = FALSE)
  }

  if (length(latent_probes) != as.integer(expected_top_n)) {
    warning(
      sprintf(
        "Latent feature file contains %s unique probes, but expected_top_n is %s.",
        length(latent_probes),
        expected_top_n
      ),
      call. = FALSE
    )
  }

  # --------------------------------------------------------------------------
  # COMPUTE: Per-comparison overlap statistics
  # --------------------------------------------------------------------------

  rows <- list()
  groups <- setdiff(names(filtered_results), c("__summary__", "__metadata__"))

  for (group in groups) {
    labels <- names(filtered_results[[group]])

    for (label in labels) {
      entry <- filtered_results[[group]][[label]]

      if (is.null(entry$filtered_probes) || length(entry$filtered_probes) == 0L) {
        next
      }

      analysis_probes <- unique(as.character(entry$filtered_probes))
      analysis_probes <- analysis_probes[!is.na(analysis_probes)]

      if (length(analysis_probes) == 0L) {
        next
      }

      overlap_probes <- intersect(analysis_probes, latent_probes)
      union_n <- length(union(analysis_probes, latent_probes))

      rows[[length(rows) + 1L]] <- tibble::tibble(
        chip_id = meta$chip_id,
        group = group,
        label = label,
        comparison = paste(entry$metadata$comparison, collapse = " / "),
        filter_method = meta$method,
        variance_selection_mode = meta$variance_selection_mode,
        analysis_top_n = as.integer(meta$parameters$top_n),
        latent_top_n = length(latent_probes),
        analysis_probe_count = length(analysis_probes),
        latent_probe_count = length(latent_probes),
        overlap_count = length(overlap_probes),
        overlap_fraction_analysis = length(overlap_probes) / length(analysis_probes),
        overlap_fraction_latent = length(overlap_probes) / length(latent_probes),
        jaccard = ifelse(union_n > 0L, length(overlap_probes) / union_n, NA_real_),
        overlap_probes = if (isTRUE(include_probe_ids)) {
          paste(overlap_probes, collapse = ";")
        } else {
          NA_character_
        }
      )
    }
  }

  out <- dplyr::bind_rows(rows)

  if (nrow(out) == 0L) {
    warning("No comparison rows were generated. Check filtered_probes entries.", call. = FALSE)
  }

  # --------------------------------------------------------------------------
  # SAVE: Optional output artifacts
  # --------------------------------------------------------------------------

  if (!is.null(output_csv)) {
    dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)
    readr::write_csv(out, output_csv)
  }

  if (!is.null(output_rds)) {
    dir.create(dirname(output_rds), recursive = TRUE, showWarnings = FALSE)
    saveRDS(out, output_rds)
  }

  out
}

# ------------------------------------------------------------------------------
# OPTIONAL STANDALONE EXECUTION TEMPLATE
# ------------------------------------------------------------------------------
filtered_results <- readRDS(here::here("output/global_cancer/RData/filtered_probes/filtered_probes_hu35ksuba.rds")
)

overlap_tbl <- compare_probe_feature_spaces(
  filtered_results = filtered_results,
  latent_features_path = here::here("data/global_cancer/processed/ml_inputs/hu35ksuba_top3000_variance_features.csv"),
  expected_chip_id = "hu35ksuba",
  expected_top_n = 3000L,
  # output_csv = (here::here("output/global_cancer/diagnostics/feature_space_overlap_hu35ksuba.csv")),
  # temporary fix for quarto
  output_csv = (here::here("quarto/resources/diagnostics/feature_space_overlap_hu35ksuba.csv")),
  output_rds = (here::here("output/global_cancer/diagnostics/feature_space_overlap_hu35ksuba.rds")),
  include_probe_ids = FALSE
)
