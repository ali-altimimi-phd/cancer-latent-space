# ==============================================================================
# File: R/visualizations/plot_structural_synthesis_diagnostics.R
# Purpose: Generate structural synthesis diagnostic plots from descriptor-first
#          structural phenotype tables.
# Project: Global Cancer Complexity / Structural Inference
# ------------------------------------------------------------------------------
# Architectural role:
#   structural engines -> structural phenotype tables -> plots/reports
#
# This script reads only Stage 9 synthesis outputs. It does not read raw engine
# outputs such as complexity_results.rds, entropy_results.rds, or
# mp_spectral_results.rds.
#
# Main pipeline-callable function:
#   plot_structural_synthesis_diagnostics()
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(forcats)
  library(tibble)
  library(here)
})

source(here::here("R", "helpers", "plot_utils.R"))

# ---- Logging and file helpers -------------------------------------------------

.log_plot_msg <- function(logger = NULL, msg, section = "SYNTHESIS") {
  if (!is.null(logger) && !is.null(logger$log)) {
    logger$log(msg, section = section)
  } else {
    message(msg)
  }
  invisible(NULL)
}

.read_required_csv <- function(path, label = basename(path)) {
  if (!file.exists(path)) {
    stop("Required CSV not found for ", label, ": ", path, call. = FALSE)
  }
  readr::read_csv(path, show_col_types = FALSE)
}

.normalize_filter_regime_request <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }

  x
}

.filter_structural_plot_df <- function(df,
                                       chips = NULL,
                                       filter_regimes = NULL) {
  filter_regimes <- .normalize_filter_regime_request(filter_regimes)

  if (!is.null(chips) && "chip" %in% names(df)) {
    df <- df |> dplyr::filter(.data$chip %in% chips)
  }

  if (!is.null(filter_regimes) && "filter_regime" %in% names(df)) {
    df <- df |> dplyr::filter(.data$filter_regime %in% filter_regimes)
  }

  df
}

.first_col <- function(df, candidates, required = TRUE, label = "column") {
  hit <- intersect(candidates, names(df))

  if (length(hit) > 0) {
    return(hit[[1]])
  }

  if (isTRUE(required)) {
    stop(
      "Could not find required ", label, ". Tried: ",
      paste(candidates, collapse = ", "),
      call. = FALSE
    )
  }

  NA_character_
}

.build_plot_wide_table <- function(wide_tbl, average_across_chips = TRUE) {
  if (!isTRUE(average_across_chips)) {
    return(wide_tbl)
  }

  numeric_cols <- names(wide_tbl)[vapply(wide_tbl, is.numeric, logical(1))]

  wide_tbl |>
    dplyr::group_by(
      dplyr::across(dplyr::all_of(c("filter_regime", "group", "comparison")))
    ) |>
    dplyr::summarise(
      dplyr::across(
        dplyr::all_of(numeric_cols),
        ~ mean(.x, na.rm = TRUE)
      ),
      n_chips = dplyr::n_distinct(.data$chip),
      chip = "averaged_across_chips",
      .groups = "drop"
    ) |>
    dplyr::mutate(
      dplyr::across(
        dplyr::all_of(numeric_cols),
        ~ ifelse(is.nan(.x), NA_real_, .x)
      )
    ) |>
    dplyr::relocate(dplyr::all_of(c("chip", "filter_regime", "group", "comparison")))
}

taxonomy_plot_path <- here::here(
  "R",
  "visualizations",
  "plot_integrated_four_engine_taxonomy.R"
)

if (file.exists(taxonomy_plot_path)) {
  source(taxonomy_plot_path)
} else {
  stop("Missing taxonomy plot script: ", taxonomy_plot_path, call. = FALSE)
}

# ---- Plot constructors --------------------------------------------------------

plot_structural_phenotype_heatmap <- function(
    heatmap_tbl,
    title = "Structural phenotype integration: key delta metrics per cancer",
    subtitle = "Raw values shown · colour = within-metric z-score") {

  metric_labels <- c(
    "complexity__kappa_delta" = "kappa delta\n(complexity)",
    "complexity__effrank_delta" = "Effective-rank\ndelta",
    "complexity__pr_delta" = "Complexity PR\ndelta",
    "complexity__eig_entropy_delta" = "Complexity eig.\nentropy delta",
    "complexity__anisotropy_delta" = "Anisotropy\ndelta",
    "entropy__shannon_delta" = "Shannon\ndelta",
    "entropy__spectral_delta" = "Entropy spectral\ndelta",
    "entropy__entropy_spectral_delta" = "Entropy spectral\ndelta",
    "mp__spectral_entropy_delta" = "MP spectral\nentropy delta",
    "mp__participation_ratio_delta" = "MP PR\ndelta",
    "mp__largest_eigenvalue_fraction_delta" = "MP largest eigen.\nfraction delta",
    "mp__excess_spectral_mass_delta" = "MP excess spectral\nmass delta",
    "latent__latent_pr_delta" = "VAE PR\ndelta",
    "latent__latent_eig_entropy_delta" = "VAE eig.\nentropy delta",
    "latent__centroid_distance" = "Centroid\ndistance"
  )

  plot_df <- heatmap_tbl |>
    dplyr::mutate(
      metric_key = paste(.data$engine, .data$metric, sep = "__"),
      metric_label = dplyr::recode(
        .data$metric_key,
        !!!metric_labels,
        .default = .data$metric_key
      ),
      comparison = forcats::fct_rev(factor(.data$comparison)),
      metric_label = factor(
        .data$metric_label,
        levels = unique(metric_labels[metric_labels %in% unique(.data$metric_label)])
      )
    )

  ggplot2::ggplot(
    plot_df,
    ggplot2::aes(
      x = .data$metric_label,
      y = .data$comparison,
      fill = .data$z_score
    )
  ) +
    ggplot2::geom_tile(color = "white", linewidth = 0.4) +
    ggplot2::geom_text(
      ggplot2::aes(label = ifelse(is.na(.data$value), "", round(.data$value, 2))),
      size = 3
    ) +
    ggplot2::facet_grid(
      rows = ggplot2::vars(.data$group),
      scales = "free_y",
      space = "free_y"
    ) +
    ggplot2::scale_fill_gradient2(
      low = "#2166AC",
      mid = "white",
      high = "#B2182B",
      midpoint = 0,
      na.value = "grey90"
    ) +
    ggplot2::labs(
      title = title,
      subtitle = subtitle,
      x = NULL,
      y = NULL,
      fill = "Z-score"
    ) +
    theme_structural_minimal(base_size = 11) +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 0, hjust = 0.5),
      panel.grid = ggplot2::element_blank(),
      strip.text.y = ggplot2::element_text(angle = 0)
    )
}

plot_mp_spectral_entropy_rank <- function(plot_wide) {
  mp_entropy_col <- .first_col(
    plot_wide,
    candidates = "mp__spectral_entropy_delta",
    label = "MP spectral entropy delta column"
  )

  comparison_levels <- plot_wide |>
    dplyr::filter(!is.na(.data[[mp_entropy_col]])) |>
    dplyr::arrange(.data[[mp_entropy_col]]) |>
    dplyr::pull("comparison") |>
    unique()

  plot_df <- plot_wide |>
    dplyr::filter(!is.na(.data[[mp_entropy_col]])) |>
    dplyr::mutate(
      comparison = factor(.data$comparison, levels = comparison_levels)
    )

  ggplot2::ggplot(
    plot_df,
    ggplot2::aes(
      x = .data[[mp_entropy_col]],
      y = .data$comparison,
      color = .data$group
    )
  ) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.7) +
    ggplot2::geom_point(size = 3) +
    ggplot2::labs(
      title = "Marchenko-Pastur spectral entropy shift: tumor vs normal",
      subtitle = "Positive values indicate higher tumor spectral entropy relative to matched normal tissue",
      x = "Spectral entropy delta (tumor - normal)",
      y = NULL,
      color = "Group"
    ) +
    theme_structural_minimal(base_size = 11)
}

plot_mp_entropy_by_participation_ratio_space <- function(plot_wide) {
  mp_entropy_col <- .first_col(
    plot_wide,
    candidates = "mp__spectral_entropy_delta",
    label = "MP spectral entropy delta column"
  )

  mp_pr_col <- .first_col(
    plot_wide,
    candidates = "mp__participation_ratio_delta",
    label = "MP participation-ratio delta column"
  )

  plot_df <- plot_wide |>
    dplyr::filter(!is.na(.data[[mp_entropy_col]]), !is.na(.data[[mp_pr_col]]))

  ggplot2::ggplot(
    plot_df,
    ggplot2::aes(
      x = .data[[mp_entropy_col]],
      y = .data[[mp_pr_col]],
      color = .data$group,
      label = .data$comparison
    )
  ) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.7) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.7) +
    ggplot2::geom_point(size = 3) +
    ggplot2::geom_text(nudge_y = 0.12, size = 3, check_overlap = TRUE) +
    ggplot2::labs(
      title = "2D spectral organization space: entropy x dimensionality",
      subtitle = "MP spectral entropy shift plotted against MP participation-ratio shift",
      x = "MP spectral entropy delta (tumor - normal)",
      y = "MP participation-ratio delta (tumor - normal)",
      color = "Group"
    ) +
    theme_structural_minimal(base_size = 11)
}

plot_complexity_entropy_taxonomy <- function(plot_wide, add_quadrant_labels = TRUE) {
  kappa_col <- .first_col(
    plot_wide,
    candidates = "complexity__kappa_delta",
    label = "complexity kappa delta column"
  )

  mp_entropy_col <- .first_col(
    plot_wide,
    candidates = "mp__spectral_entropy_delta",
    label = "MP spectral entropy delta column"
  )

  plot_df <- plot_wide |>
    dplyr::filter(!is.na(.data[[kappa_col]]), !is.na(.data[[mp_entropy_col]]))

  p <- ggplot2::ggplot(
    plot_df,
    ggplot2::aes(
      x = .data[[kappa_col]],
      y = .data[[mp_entropy_col]],
      color = .data$group,
      label = .data$comparison
    )
  ) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.7) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.7) +
    ggplot2::geom_point(size = 3) +
    ggplot2::geom_text(nudge_y = 0.04, size = 3, check_overlap = TRUE) +
    ggplot2::labs(
      title = "2D complexity x entropy taxonomy",
      subtitle = "Condition-number complexity shift x Marchenko-Pastur spectral entropy shift",
      x = "Delta condition number (tumor - normal) [complexity engine]",
      y = "Delta MP spectral entropy (tumor - normal)",
      color = "Group",
      caption = paste(
        "Quadrants:",
        "I   = gained complexity / high entropy; distributed reorganization",
        "II  = lost complexity / high entropy; structural proxy",
        "III = lost complexity / low entropy; structural compression",
        "IV  = gained complexity / low entropy; concentrated reorganization",
        "Note: labels are structural-state descriptors, not direct measurements of dynamical chaos.",
        sep = "\n"
      )      
    ) +
    theme_structural_minimal(base_size = 11) +
    ggplot2::theme(
      plot.caption = ggplot2::element_text(
        hjust = 0,
        size = 8.5,
        lineheight = 1.05,
        margin = ggplot2::margin(t = 10)
      ),
      plot.caption.position = "plot"
    )

  if (isTRUE(add_quadrant_labels) && nrow(plot_df) > 0) {
    x_min <- min(plot_df[[kappa_col]], na.rm = TRUE)
    x_max <- max(plot_df[[kappa_col]], na.rm = TRUE)
    y_min <- min(plot_df[[mp_entropy_col]], na.rm = TRUE)
    y_max <- max(plot_df[[mp_entropy_col]], na.rm = TRUE)

    quad_labels <- tibble::tibble(
      x = c(x_max, x_min, x_min, x_max),
      y = c(y_max, y_max, y_min, y_min),
      label = c("I", "II", "III", "IV"),
      hjust = c(1.15, -0.15, -0.15, 1.15),
      vjust = c(1.15, 1.15, -0.15, -0.15)
    )

    p <- p +
      ggplot2::geom_text(
        data = quad_labels,
        ggplot2::aes(
          x = .data$x,
          y = .data$y,
          label = .data$label,
          hjust = .data$hjust,
          vjust = .data$vjust
        ),
        inherit.aes = FALSE,
        color = "grey35",
        fontface = "bold",
        size = 5
      )
  }

  p
}

plot_integrated_structural_taxonomy <- function(plot_wide) {
  kappa_col <- .first_col(
    plot_wide,
    candidates = "complexity__kappa_delta",
    label = "complexity kappa delta column"
  )

  mp_entropy_col <- .first_col(
    plot_wide,
    candidates = "mp__spectral_entropy_delta",
    label = "MP spectral entropy delta column"
  )

  centroid_col <- .first_col(
    plot_wide,
    candidates = "latent__centroid_distance",
    required = FALSE,
    label = "latent centroid distance column"
  )

  if (is.na(centroid_col)) {
    return(NULL)
  }

  plot_df <- plot_wide |>
    dplyr::filter(
      !is.na(.data[[kappa_col]]),
      !is.na(.data[[mp_entropy_col]]),
      !is.na(.data[[centroid_col]])
    )

  if (nrow(plot_df) == 0) {
    return(NULL)
  }

  ggplot2::ggplot(
    plot_df,
    ggplot2::aes(
      x = .data[[kappa_col]],
      y = .data[[mp_entropy_col]],
      color = .data$group,
      shape = .data$group,
      size = .data[[centroid_col]],
      label = .data$comparison
    )
  ) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.7) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.7) +
    ggplot2::geom_point(alpha = 0.9) +
    ggplot2::geom_text(nudge_y = 0.04, size = 3, check_overlap = TRUE) +
    ggplot2::labs(
      title = "Integrated structural cancer taxonomy",
      subtitle = "Complexity x Marchenko-Pastur; size = VAE latent centroid distance",
      x = "Delta condition number (tumor - normal) [complexity engine]",
      y = "Delta MP spectral entropy (tumor - normal)",
      color = "Group",
      shape = "Group",
      size = "Centroid distance\n(VAE latent)"
    ) +
    theme_structural_minimal(base_size = 11)
}

# ---- Main pipeline-callable plotting function --------------------------------

#' Build structural synthesis diagnostic plots
#'
#' @param study_name Study/output namespace.
#' @param synthesis_table_dir Directory containing Stage 9 CSV outputs.
#' @param synthesis_plot_dir Directory for output PNG/PDF plots.
#' @param chips Optional chip filter; NULL keeps all chips.
#' @param filter_regimes Optional filter-regime filter; NULL keeps all regimes.
#' @param average_across_chips Whether taxonomy/rank plots average over chips.
#' @param save_pdf Whether to save PDF companions for each PNG.
#' @param logger Optional project logger.
#' @param print_plots Whether to print ggplot objects in interactive sessions.
#'
#' @return Invisible list containing plot objects and written file paths.
#' @export
plot_structural_synthesis_diagnostics <- function(
    study_name = "global_cancer",
    synthesis_table_dir = here::here(
      "output", study_name, "structural_inference", "tables", "synthesis"
    ),
    synthesis_plot_dir = here::here(
      "output", study_name, "structural_inference", "plots", "synthesis"
    ),
    chips = NULL,
    filter_regimes = NULL,
    average_across_chips = TRUE,
    save_pdf = TRUE,
    logger = NULL,
    print_plots = interactive()) {

  .log_plot_msg(logger, "📊 Loading structural synthesis tables for plotting...")

  long_path <- file.path(synthesis_table_dir, "structural_phenotype_long.csv")
  wide_path <- file.path(synthesis_table_dir, "structural_phenotype_wide.csv")
  latent_aligned_wide_path <- file.path(
    synthesis_table_dir,
    "structural_phenotype_latent_aligned_wide.csv"
  )
  heatmap_path <- file.path(synthesis_table_dir, "structural_phenotype_heatmap_long.csv")

  phenotype_long <- .read_required_csv(long_path, "structural phenotype long table")
  phenotype_wide <- .read_required_csv(wide_path, "structural phenotype wide table")
  phenotype_latent_aligned_wide <- .read_required_csv(
    latent_aligned_wide_path,
    "latent-aligned structural phenotype wide table"
  )
  heatmap_long <- .read_required_csv(heatmap_path, "structural phenotype heatmap table")

  phenotype_long <- .filter_structural_plot_df(phenotype_long, chips, filter_regimes)
  phenotype_wide <- .filter_structural_plot_df(phenotype_wide, chips, filter_regimes)
  phenotype_latent_aligned_wide <- .filter_structural_plot_df(
    phenotype_latent_aligned_wide,
    chips = chips,
    filter_regimes = NULL
  )
  heatmap_long <- .filter_structural_plot_df(heatmap_long, chips, filter_regimes)

  dir.create(synthesis_plot_dir, recursive = TRUE, showWarnings = FALSE)

  .log_plot_msg(logger, "📊 Building structural synthesis diagnostic plots...")

  plot_wide <- .build_plot_wide_table(
    wide_tbl = phenotype_wide,
    average_across_chips = average_across_chips
  )

  latent_plot_wide <- .build_plot_wide_table(
    wide_tbl = phenotype_latent_aligned_wide,
    average_across_chips = FALSE
  )

  filter_label <- if (is.null(filter_regimes)) {
    "all filter regimes"
  } else {
    paste(.normalize_filter_regime_request(filter_regimes), collapse = ", ")
  }

  chip_label <- if (isTRUE(average_across_chips)) {
    "averaged across chips where available"
  } else if (is.null(chips)) {
    "all chips"
  } else {
    paste(chips, collapse = ", ")
  }

  heatmap_plot <- plot_structural_phenotype_heatmap(
    heatmap_tbl = heatmap_long,
    subtitle = paste(
      "Raw values shown · colour = within-metric z-score ·",
      chip_label,
      "·",
      filter_label
    )
  )

  mp_rank_plot <- plot_mp_spectral_entropy_rank(plot_wide)
  mp_space_plot <- plot_mp_entropy_by_participation_ratio_space(plot_wide)
  complexity_entropy_plot <- plot_complexity_entropy_taxonomy(plot_wide)
  
  .log_plot_msg(
    logger,
    sprintf(
      "Latent plot wide: rows=%d; centroid_col=%s; nonmissing_centroid=%d",
      nrow(latent_plot_wide),
      "latent__centroid_distance" %in% names(latent_plot_wide),
      if ("latent__centroid_distance" %in% names(latent_plot_wide)) {
        sum(!is.na(latent_plot_wide$latent__centroid_distance))
      } else {
        0L
      }
    )
  )
  
  integrated_plot <- plot_integrated_four_engine_taxonomy(latent_plot_wide)
  .log_plot_msg(logger, "💾 Writing structural synthesis diagnostic plots...")

  written_files <- list(
    heatmap = save_plot_pair(
      plot = heatmap_plot,
      output_dir = synthesis_plot_dir,
      basename_no_ext = "structural_phenotype_heatmap",
      width = 11,
      height = 6,
      save_pdf = save_pdf
    ),
    mp_rank = save_plot_pair(
      plot = mp_rank_plot,
      output_dir = synthesis_plot_dir,
      basename_no_ext = "mp_spectral_entropy_shift_ranked",
      width = 9,
      height = 5,
      save_pdf = save_pdf
    ),
    mp_space = save_plot_pair(
      plot = mp_space_plot,
      output_dir = synthesis_plot_dir,
      basename_no_ext = "mp_entropy_by_participation_ratio_space",
      width = 9,
      height = 5,
      save_pdf = save_pdf
    ),
    complexity_entropy_taxonomy = save_plot_pair(
      plot = complexity_entropy_plot,
      output_dir = synthesis_plot_dir,
      basename_no_ext = "complexity_entropy_taxonomy_space",
      width = 10,
      height = 6,
      save_pdf = save_pdf
    )
  )

  if (!is.null(integrated_plot)) {
    written_files$integrated_taxonomy <- save_plot_pair(
      plot = integrated_plot,
      output_dir = synthesis_plot_dir,
      basename_no_ext = "integrated_four_engine_taxonomy_space",
      width = 9,
      height = 5,
      save_pdf = save_pdf
    )
  } else {
    .log_plot_msg(
      logger,
      "ℹ️ Integrated latent-size taxonomy plot skipped because latent centroid distance was unavailable."
    )
  }

  if (isTRUE(print_plots)) {
    print(heatmap_plot)
    print(mp_rank_plot)
    print(mp_space_plot)
    print(complexity_entropy_plot)
    if (!is.null(integrated_plot)) {
      print(integrated_plot)
    }
  }

  invisible(list(
    plots = list(
      heatmap = heatmap_plot,
      mp_rank = mp_rank_plot,
      mp_space = mp_space_plot,
      complexity_entropy_taxonomy = complexity_entropy_plot,
      integrated_taxonomy = integrated_plot
    ),
    files = written_files,
    input_files = list(
      long = long_path,
      wide = wide_path,
      latent_aligned_wide = latent_aligned_wide_path,
      heatmap_long = heatmap_path
    ),
    plot_dir = synthesis_plot_dir
  ))
}
