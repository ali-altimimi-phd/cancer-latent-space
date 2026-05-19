# ------------------------------------------------------------------------------
# File: generate_probe_feature_concordance.R
# Purpose: Compare latent-space top-variance input probes against comparison-
#   specific limma-filtered probes and generate concordance tables, decomposed
#   probe-set outputs, and plots for Quarto reporting.
# Role: Feature-space analysis / reporting-support utility
# Pipeline: Reporting / Feature-space concordance
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Generate Probe Feature-Space Concordance Outputs
#'
#' Compares a global latent-space input feature set, such as the hu35ksuba
#' top-3000 variance probes, against comparison-specific limma-filtered probe
#' sets stored by the analysis pipeline. The analysis reports overlap counts,
#' overlap fractions, fold enrichment relative to a chip-specific probe universe,
#' and hypergeometric enrichment p-values with FDR adjustment.
#'
#' The function also exports a per-comparison decomposition of probe sets:
#' overlap, limma-only, and latent-only. These outputs are designed to support
#' later biological annotation using GO, KEGG, or MSigDB resources without
#' forcing annotation into the concordance step.
#'
#' @param filtered_probes_path Path to filtered_probes_<chip>.rds.
#' @param latent_features_path Path to top-N latent input feature CSV.
#' @param chip Character chip identifier. Default: "hu35ksuba".
#' @param output_table_dir Directory for CSV/RDS outputs.
#' @param output_plot_dir Directory for PNG plot outputs.
#' @param plot_utils_path Optional path to plot utility functions. If present,
#'   save_png_light() will be used; otherwise ggplot2::ggsave() is used.
#' @param universe_probes Optional character vector defining the probe universe.
#'   If NULL, the universe is inferred as the union of all limma-filtered probes
#'   and latent feature probes. For formal inference, prefer passing the full
#'   eligible chip probe universe after preprocessing/QC.
#' @param top_feature_col Optional column name in latent feature CSV containing
#'   probe IDs. If NULL, the function guesses from common names or uses the first
#'   column.
#' @param annotation_path Optional path to full_chip_annotations.rds. If supplied
#'   and write_annotated_rds is TRUE, an annotation-joined RDS is written.
#' @param write_annotated_rds Logical. If TRUE and annotation_path is supplied,
#'   writes an annotation-joined RDS for downstream GO/KEGG/MSigDB inspection.
#'   Defaults to FALSE because annotation joins can be large.
#' @return Invisibly returns a list with concordance_table, group_summary,
#'   probe_sets_long, and probe_sets_wide.
#' @export

generate_probe_feature_concordance <- function(filtered_probes_path = "output/global_cancer/RData/filtered_probes/filtered_probes_hu35ksuba.rds",
                                               latent_features_path = "data/global_cancer/processed/ml_inputs/hu35ksuba_top3000_variance_features.csv",
                                               chip = "hu35ksuba",
                                               output_table_dir = "output/global_cancer/tables/probe_concordance",
                                               output_plot_dir = "output/global_cancer/plots/probe_concordance",
                                               plot_utils_path = "R/helpers/plot_utils.R",
                                               universe_probes = NULL,
                                               top_feature_col = NULL,
                                               annotation_path = NULL,
                                               write_annotated_rds = FALSE) {
  required_pkgs <- c("dplyr", "readr", "tibble", "tidyr", "ggplot2", "stringr")
  missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0) {
    stop("Missing required packages: ",
         paste(missing_pkgs, collapse = ", "),
         call. = FALSE)
  }
  
  if (!file.exists(filtered_probes_path)) {
    stop("Filtered probes file not found: ",
         filtered_probes_path,
         call. = FALSE)
  }
  if (!file.exists(latent_features_path)) {
    stop("Latent feature file not found: ",
         latent_features_path,
         call. = FALSE)
  }
  
  dir.create(output_table_dir,
             recursive = TRUE,
             showWarnings = FALSE)
  dir.create(output_plot_dir,
             recursive = TRUE,
             showWarnings = FALSE)
  
  if (!is.null(plot_utils_path) && file.exists(plot_utils_path)) {
    source(plot_utils_path)
  }
  
  save_plot <- function(plot,
                        path,
                        width = 9,
                        height = 6,
                        dpi = 300) {
    if (exists("save_png_light", mode = "function")) {
      save_png_light(plot,
                     path,
                     width = width,
                     height = height,
                     dpi = dpi)
    } else {
      ggplot2::ggsave(
        path,
        plot = plot,
        width = width,
        height = height,
        dpi = dpi,
        bg = "white"
      )
    }
  }
  
  group_palette <- c(
    blastomas  = "#CC79A7",
    carcinomas = "#D55E00",
    leukemias  = "#0072B2",
    lymphomas  = "#009E73"
  )
  
  # ---- Load inputs ----
  filtered_obj <- readRDS(filtered_probes_path)
  latent_df <- readr::read_csv(latent_features_path, show_col_types = FALSE)
  
  latent_probes <- extract_latent_feature_ids(latent_df, top_feature_col = top_feature_col)
  latent_probes <- unique(stats::na.omit(as.character(latent_probes)))
  
  filtered_tbl <- flatten_filtered_probe_object(filtered_obj)
  
  if (!all(c("group", "comparison", "probes") %in% names(filtered_tbl))) {
    stop(
      "Could not flatten filtered probes object into group/comparison/probes columns.",
      call. = FALSE
    )
  }
  
  filtered_tbl <- filtered_tbl |>
    dplyr::mutate(
      group = as.character(.data$group),
      comparison = as.character(.data$comparison),
      probes = lapply(.data$probes, function(x)
        unique(stats::na.omit(as.character(
          x
        ))))
    ) |>
    dplyr::filter(lengths(.data$probes) > 0)
  
  if (nrow(filtered_tbl) == 0) {
    stop("No non-empty filtered probe sets were found.", call. = FALSE)
  }
  
  # ---- Universe ----
  if (is.null(universe_probes)) {
    universe_probes <- unique(c(
      unlist(filtered_tbl$probes, use.names = FALSE),
      latent_probes
    ))
    universe_source <- "inferred_union_of_limma_filtered_and_latent_top_features"
  } else {
    universe_probes <- unique(stats::na.omit(as.character(universe_probes)))
    universe_source <- "user_supplied"
  }
  
  universe_n <- length(universe_probes)
  latent_probes_in_universe <- intersect(latent_probes, universe_probes)
  top_n <- length(latent_probes_in_universe)
  
  if (top_n == 0) {
    stop("No latent feature probes overlap the supplied/inferred universe.",
         call. = FALSE)
  }
  
  # ---- Concordance table ----
  concordance_tbl <- filtered_tbl |>
    dplyr::rowwise() |>
    dplyr::mutate(
      chip = chip,
      
      limma_probes_in_universe = list(intersect(.data$probes, universe_probes)),
      latent_probes_in_universe = list(latent_probes_in_universe),
      
      limma_n = length(.data$limma_probes_in_universe),
      top_feature_n = top_n,
      
      overlap_probes = list(
        intersect(
          .data$limma_probes_in_universe,
          .data$latent_probes_in_universe
        )
      ),
      limma_only_probes = list(
        setdiff(
          .data$limma_probes_in_universe,
          .data$latent_probes_in_universe
        )
      ),
      latent_only_probes = list(
        setdiff(
          .data$latent_probes_in_universe,
          .data$limma_probes_in_universe
        )
      ),
      
      overlap_n = length(.data$overlap_probes),
      non_overlap_limma = length(.data$limma_only_probes),
      non_overlap_top3000 = length(.data$latent_only_probes),
      
      overlap_frac_of_limma = dplyr::if_else(.data$limma_n > 0, .data$overlap_n / .data$limma_n, NA_real_),
      
      overlap_frac_of_top3000 = dplyr::if_else(
        .data$top_feature_n > 0,
        .data$overlap_n / .data$top_feature_n,
        NA_real_
      ),
      
      universe_n = universe_n,
      universe_source = universe_source,
      
      expected_overlap = (.data$limma_n * .data$top_feature_n) / .data$universe_n,
      
      fold_enrichment = dplyr::if_else(
        .data$expected_overlap > 0,
        .data$overlap_n / .data$expected_overlap,
        NA_real_
      ),
      
      hypergeom_p = dplyr::if_else(
        .data$universe_n >= .data$limma_n &&
          .data$universe_n >= .data$top_feature_n &&
          .data$universe_n > 0,
        stats::phyper(
          q = .data$overlap_n - 1,
          m = .data$limma_n,
          n = .data$universe_n - .data$limma_n,
          k = .data$top_feature_n,
          lower.tail = FALSE
        ),
        NA_real_
      )
    ) |>
    dplyr::ungroup() |>
    dplyr::mutate(hypergeom_fdr = stats::p.adjust(.data$hypergeom_p, method = "BH")) |>
    dplyr::arrange(.data$group,
                   dplyr::desc(.data$overlap_frac_of_limma),
                   .data$comparison)
  
  table_for_export <- concordance_tbl |>
    dplyr::select(
      chip,
      group,
      comparison,
      limma_n,
      top_feature_n,
      overlap_n,
      non_overlap_limma,
      non_overlap_top3000,
      overlap_frac_of_limma,
      overlap_frac_of_top3000,
      universe_n,
      universe_source,
      expected_overlap,
      fold_enrichment,
      hypergeom_p,
      hypergeom_fdr
    )
  
  # ---- Probe-set decomposition outputs ----
  probe_sets_wide <- concordance_tbl |>
    dplyr::select(
      chip,
      group,
      comparison,
      limma_probes = limma_probes_in_universe,
      latent_probes = latent_probes_in_universe,
      overlap_probes,
      limma_only_probes,
      latent_only_probes
    )
  
  probe_sets_long <- make_probe_set_long(probe_sets_wide)
  
  clean_chip <- gsub("[^a-zA-Z0-9]", "_", tolower(chip))
  
  csv_path <- file.path(output_table_dir,
                        paste0(clean_chip, "_top_features_vs_limma_concordance.csv"))
  rds_path <- file.path(output_table_dir,
                        paste0(clean_chip, "_top_features_vs_limma_concordance.rds"))
  group_csv_path <- file.path(
    output_table_dir,
    paste0(clean_chip, "_top_features_vs_limma_group_summary.csv")
  )
  
  probe_sets_wide_rds_path <- file.path(
    output_table_dir,
    paste0(clean_chip, "_top_features_vs_limma_probe_sets_wide.rds")
  )
  probe_sets_long_csv_path <- file.path(
    output_table_dir,
    paste0(clean_chip, "_top_features_vs_limma_probe_sets_long.csv")
  )
  probe_sets_long_rds_path <- file.path(
    output_table_dir,
    paste0(clean_chip, "_top_features_vs_limma_probe_sets_long.rds")
  )
  
  # Backward-compatible path from earlier versions.
  overlap_rds_path <- file.path(
    output_table_dir,
    paste0(
      clean_chip,
      "_top_features_vs_limma_overlap_probe_sets.rds"
    )
  )
  overlap_sets <- concordance_tbl |>
    dplyr::select(chip, group, comparison, overlap_probes)
  
  readr::write_csv(table_for_export, csv_path)
  saveRDS(table_for_export, rds_path)
  saveRDS(overlap_sets, overlap_rds_path)
  saveRDS(probe_sets_wide, probe_sets_wide_rds_path)
  readr::write_csv(probe_sets_long, probe_sets_long_csv_path)
  saveRDS(probe_sets_long, probe_sets_long_rds_path)
  
  annotated_probe_sets <- NULL
  annotated_rds_path <- NULL
  if (isTRUE(write_annotated_rds)) {
    if (is.null(annotation_path) || !file.exists(annotation_path)) {
      warning(
        "write_annotated_rds is TRUE, but annotation_path is missing or does not exist. Skipping annotation join."
      )
    } else {
      annotated_probe_sets <- annotate_probe_set_long(
        probe_sets_long = probe_sets_long,
        annotation_path = annotation_path,
        chip = chip
      )
      annotated_rds_path <- file.path(
        output_table_dir,
        paste0(
          clean_chip,
          "_top_features_vs_limma_probe_sets_annotated.rds"
        )
      )
      saveRDS(annotated_probe_sets, annotated_rds_path)
    }
  }
  
  # ---- Plots ----
  plot_tbl <- table_for_export |>
    dplyr::mutate(
      comparison_label = stringr::str_wrap(.data$comparison, width = 35),
      significant = .data$hypergeom_fdr <= 0.05
    )
  
  p_overlap <- ggplot2::ggplot(
    plot_tbl,
    ggplot2::aes(
      x = forcats::fct_reorder(
        .data$comparison_label,
        .data$overlap_frac_of_limma,
        .desc = TRUE
      ),
      y = .data$overlap_frac_of_limma,
      fill = .data$group
    )
  ) +
    ggplot2::geom_col() +
    ggplot2::coord_flip() +
    ggplot2::scale_fill_manual(values = group_palette, drop = FALSE) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = paste0(
        chip,
        ": limma-filtered probe coverage by latent top-3000 features"
      ),
      x = NULL,
      y = "Overlap fraction of limma-filtered probes",
      fill = "Cancer group"
    )
  
  save_plot(
    p_overlap,
    file.path(
      output_plot_dir,
      paste0(clean_chip, "_top_features_vs_limma_overlap_fraction.png")
    ),
    width = 10,
    height = max(5, 0.35 * nrow(plot_tbl))
  )
  
  p_fold <- ggplot2::ggplot(
    plot_tbl,
    ggplot2::aes(
      x = forcats::fct_reorder(.data$comparison_label, .data$fold_enrichment, .desc = TRUE),
      y = .data$fold_enrichment,
      fill = .data$group
    )
  ) +
    ggplot2::geom_hline(yintercept = 1, linetype = "dashed") +
    ggplot2::geom_col() +
    ggplot2::coord_flip() +
    ggplot2::scale_fill_manual(values = group_palette, drop = FALSE) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = paste0(
        chip,
        ": enrichment of limma-filtered probes in latent top features"
      ),
      x = NULL,
      y = "Fold enrichment over random expectation",
      fill = "Cancer group"
    )
  
  save_plot(
    p_fold,
    file.path(
      output_plot_dir,
      paste0(clean_chip, "_top_features_vs_limma_fold_enrichment.png")
    ),
    width = 10,
    height = max(5, 0.35 * nrow(plot_tbl))
  )
  
  group_summary <- table_for_export |>
    dplyr::group_by(.data$chip, .data$group) |>
    dplyr::summarise(
      comparisons_n = dplyr::n(),
      median_limma_n = stats::median(.data$limma_n, na.rm = TRUE),
      median_overlap_n = stats::median(.data$overlap_n, na.rm = TRUE),
      median_overlap_frac_of_limma = stats::median(.data$overlap_frac_of_limma, na.rm = TRUE),
      median_overlap_frac_of_top3000 = stats::median(.data$overlap_frac_of_top3000, na.rm = TRUE),
      median_fold_enrichment = stats::median(.data$fold_enrichment, na.rm = TRUE),
      significant_comparisons_n = sum(.data$hypergeom_fdr <= 0.05, na.rm = TRUE),
      .groups = "drop"
    )
  
  readr::write_csv(group_summary, group_csv_path)
  
  p_group <- ggplot2::ggplot(
    group_summary,
    ggplot2::aes(
      x = forcats::fct_reorder(.data$group, .data$median_overlap_frac_of_limma, .desc = TRUE),
      y = .data$median_overlap_frac_of_limma,
      fill = .data$group
    )
  ) +
    ggplot2::geom_col(show.legend = FALSE) +
    ggplot2::coord_flip() +
    ggplot2::scale_fill_manual(values = group_palette, drop = FALSE) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = paste0(chip, ": median concordance by cancer group"),
      x = NULL,
      y = "Median overlap fraction of limma-filtered probes"
    )
  
  save_plot(
    p_group,
    file.path(
      output_plot_dir,
      paste0(clean_chip, "_top_features_vs_limma_group_summary.png")
    ),
    width = 8,
    height = max(4, 0.45 * nrow(group_summary))
  )
  
  message("Wrote concordance table: ", csv_path)
  message("Wrote group summary: ", group_csv_path)
  message("Wrote overlap probe sets: ", overlap_rds_path)
  message("Wrote probe-set decomposition: ", probe_sets_long_csv_path)
  
  if (!is.null(annotated_rds_path)) {
    message("Wrote annotated probe-set RDS: ", annotated_rds_path)
  }
  
  invisible(list(
    concordance_table = table_for_export,
    group_summary = group_summary,
    overlap_sets = overlap_sets,
    probe_sets_wide = probe_sets_wide,
    probe_sets_long = probe_sets_long,
    annotated_probe_sets = annotated_probe_sets
  ))
}

#' Extract latent feature IDs from a CSV read as a data frame.
#' @keywords internal
extract_latent_feature_ids <- function(latent_df, top_feature_col = NULL) {
  if (!is.null(top_feature_col)) {
    if (!top_feature_col %in% names(latent_df)) {
      stop(
        "Requested top_feature_col not found in latent feature CSV: ",
        top_feature_col,
        call. = FALSE
      )
    }
    return(latent_df[[top_feature_col]])
  }
  
  candidate_cols <- c(
    "probe_id",
    "probe",
    "probeset",
    "probe_set",
    "probe_set_id",
    "feature",
    "feature_id",
    "id",
    "ID"
  )
  
  found <- candidate_cols[candidate_cols %in% names(latent_df)]
  if (length(found) > 0) {
    return(latent_df[[found[1]]])
  }
  
  latent_df[[1]]
}

#' Flatten filtered probe object into group/comparison/probes rows.
#' @keywords internal
flatten_filtered_probe_object <- function(obj) {
  if (is.data.frame(obj)) {
    return(flatten_filtered_probe_dataframe(obj))
  }
  
  if (is.list(obj) &&
      !is.null(obj$`__summary__`$probe_hits_by_group)) {
    return(flatten_named_probe_list(obj$`__summary__`$probe_hits_by_group))
  }
  
  if (is.list(obj) &&
      !is.null(names(obj)) && all(grepl("::", names(obj)))) {
    return(flatten_named_probe_list(obj))
  }
  
  if (is.list(obj) && !is.null(names(obj))) {
    rows <- list()
    for (group_nm in names(obj)) {
      group_obj <- obj[[group_nm]]
      if (is.null(group_obj))
        next
      
      if (is.list(group_obj) && !is.null(names(group_obj))) {
        for (cmp_nm in names(group_obj)) {
          probes <- group_obj[[cmp_nm]]
          if (is.atomic(probes)) {
            rows[[length(rows) + 1]] <- tibble::tibble(
              group = group_nm,
              comparison = cmp_nm,
              probes = list(as.character(probes))
            )
          }
        }
      } else if (is.atomic(group_obj)) {
        parsed <- parse_group_comparison_name(group_nm)
        rows[[length(rows) + 1]] <- tibble::tibble(
          group = parsed$group,
          comparison = parsed$comparison,
          probes = list(as.character(group_obj))
        )
      }
    }
    
    if (length(rows) > 0) {
      return(dplyr::bind_rows(rows))
    }
  }
  
  stop(
    "Unsupported filtered probe object structure. Inspect with str(readRDS(...), max.level = 3).",
    call. = FALSE
  )
}

#' Flatten a named list of probe vectors.
#' @keywords internal
flatten_named_probe_list <- function(x) {
  rows <- lapply(names(x), function(nm) {
    parsed <- parse_group_comparison_name(nm)
    tibble::tibble(
      group = parsed$group,
      comparison = parsed$comparison,
      probes = list(as.character(x[[nm]]))
    )
  })
  dplyr::bind_rows(rows)
}

#' Flatten a data frame filtered-probe representation.
#' @keywords internal
flatten_filtered_probe_dataframe <- function(x) {
  probe_col <- intersect(
    c(
      "probe",
      "probe_id",
      "probe_set",
      "probe_set_id",
      "feature",
      "feature_id"
    ),
    names(x)
  )[1]
  probes_col <- intersect(c("probes", "filtered_probes", "probe_ids"), names(x))[1]
  
  if (!is.na(probes_col) &&
      all(c("group", "comparison") %in% names(x))) {
    return(
      x |>
        dplyr::transmute(
          group = as.character(.data$group),
          comparison = as.character(.data$comparison),
          probes = .data[[probes_col]]
        )
    )
  }
  
  if (!is.na(probe_col) &&
      all(c("group", "comparison") %in% names(x))) {
    return(
      x |>
        dplyr::group_by(.data$group, .data$comparison) |>
        dplyr::summarise(probes = list(as.character(.data[[probe_col]])), .groups = "drop")
    )
  }
  
  stop(
    "Unsupported filtered probe data frame. Expected group/comparison plus probe or probes column.",
    call. = FALSE
  )
}

#' Parse a name into group and comparison.
#' @keywords internal
parse_group_comparison_name <- function(nm) {
  if (grepl("::", nm)) {
    list(group = sub("::.*", "", nm),
         comparison = sub(".*::", "", nm))
  } else {
    list(group = NA_character_, comparison = nm)
  }
}

#' Convert wide list-column probe sets to long annotation-ready membership table.
#' @keywords internal
make_probe_set_long <- function(probe_sets_wide) {
  rows <- vector("list", nrow(probe_sets_wide) * 3L)
  idx <- 1L
  
  set_cols <- c(overlap = "overlap_probes",
                limma_only = "limma_only_probes",
                latent_only = "latent_only_probes")
  
  for (i in seq_len(nrow(probe_sets_wide))) {
    base_row <- probe_sets_wide[i, c("chip", "group", "comparison")]
    
    for (set_type in names(set_cols)) {
      probes <- unique(stats::na.omit(as.character(probe_sets_wide[[set_cols[[set_type]]]][[i]])))
      rows[[idx]] <- tibble::tibble(
        chip = base_row$chip,
        group = base_row$group,
        comparison = base_row$comparison,
        set_type = set_type,
        probe_id = probes
      )
      idx <- idx + 1L
    }
  }
  
  dplyr::bind_rows(rows) |>
    dplyr::arrange(.data$group,
                   .data$comparison,
                   .data$set_type,
                   .data$probe_id)
}

#' Join probe-set membership to full-chip annotation table.
#' @keywords internal
annotate_probe_set_long <- function(probe_sets_long, annotation_path, chip = "hu35ksuba") {
  full_chip_annotations <- readRDS(annotation_path)
  
  if (!chip %in% names(full_chip_annotations)) {
    stop("Chip not found in full_chip_annotations: ", chip, call. = FALSE)
  }
  if (is.null(full_chip_annotations[[chip]]$annotation_table)) {
    stop("annotation_table not found for chip: ", chip, call. = FALSE)
  }
  
  annotation_table <- full_chip_annotations[[chip]]$annotation_table |>
    tibble::as_tibble() |>
    dplyr::mutate(PROBEID = as.character(.data$PROBEID))
  
  probe_sets_long |>
    dplyr::mutate(probe_id = as.character(.data$probe_id)) |>
    dplyr::left_join(annotation_table, by = c("probe_id" = "PROBEID"))
}

