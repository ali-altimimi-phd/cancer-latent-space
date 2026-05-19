# ==============================================================================
# File: R/helpers/plot_utils.R
# Purpose: General-purpose plotting utilities used across Global Cancer analyses
# Project: Global Cancer Complexity / Structural Inference
# Author: Ali M. Al-Timimi
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tibble)
})

# ---- Data-validity helpers ----------------------------------------------------

#' Filter rows with valid positive p-values for plotting
#'
#' Removes rows with missing, non-finite, or non-positive permutation p-values
#' before computing transformations such as -log10(p_perm).
#'
#' @param df Data frame containing a p-value column.
#' @param p_col Name of the p-value column.
#'
#' @return Filtered data frame containing only rows with valid positive p-values.
#' @export
valid_plot_rows <- function(df, p_col = "p_perm") {
  if (!p_col %in% names(df)) {
    stop("p-value column not found: ", p_col, call. = FALSE)
  }

  df |>
    dplyr::filter(
      !is.na(.data[[p_col]]),
      is.finite(.data[[p_col]]),
      .data[[p_col]] > 0
    )
}

# ---- Theme helpers ------------------------------------------------------------

#' Minimal project theme for structural-inference plots
#'
#' @param base_size Base font size.
#'
#' @return A ggplot2 theme object.
#' @export
theme_structural_minimal <- function(base_size = 11) {
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold"),
      panel.grid.minor = ggplot2::element_blank()
    )
}

# ---- Generic plot constructors retained for legacy/reusable analyses ----------

#' Create a bar plot with confidence intervals
#'
#' @param values Numeric vector of means.
#' @param ci_lows Numeric vector of lower CI bounds.
#' @param ci_highs Numeric vector of upper CI bounds.
#' @param group_labels Character vector of group labels.
#' @param title Plot title.
#' @param ylab Y-axis label.
#'
#' @return A ggplot object.
#' @export
make_ci_barplot <- function(values, ci_lows, ci_highs, group_labels, title, ylab) {
  tibble::tibble(
    Group = group_labels,
    Mean = values,
    CI_Low = ci_lows,
    CI_High = ci_highs
  ) |>
    ggplot2::ggplot(ggplot2::aes(x = .data$Group, y = .data$Mean)) +
    ggplot2::geom_bar(stat = "identity", fill = "steelblue") +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = .data$CI_Low, ymax = .data$CI_High),
      width = 0.2
    ) +
    ggplot2::labs(title = title, y = ylab, x = NULL) +
    theme_structural_minimal()
}

#' Create a histogram showing permutation-test results
#'
#' @param null_dist Numeric vector of permuted/null values.
#' @param observed_value Numeric observed value.
#' @param title Plot title.
#' @param xlab X-axis label.
#'
#' @return A ggplot object.
#' @export
make_permutation_plot <- function(null_dist, observed_value, title, xlab) {
  tibble::tibble(Permuted = null_dist) |>
    ggplot2::ggplot(ggplot2::aes(x = .data$Permuted)) +
    ggplot2::geom_histogram(bins = 40, fill = "gray", color = "black") +
    ggplot2::geom_vline(
      xintercept = observed_value,
      color = "red",
      linetype = "dashed",
      linewidth = 1
    ) +
    ggplot2::labs(title = title, x = xlab, y = "Count") +
    theme_structural_minimal()
}

#' Create a violin plot with jittered points
#'
#' @param values Numeric vector of values.
#' @param groups Factor or character vector of group labels.
#' @param title Plot title.
#' @param ylab Y-axis label.
#'
#' @return A ggplot object.
#' @export
make_violin_jitter_plot <- function(values, groups, title, ylab) {
  tibble::tibble(
    Value = values,
    Group = groups
  ) |>
    ggplot2::ggplot(ggplot2::aes(x = .data$Group, y = .data$Value, fill = .data$Group)) +
    ggplot2::geom_violin(trim = FALSE, color = "black") +
    ggplot2::geom_jitter(width = 0.1, alpha = 0.3, size = 1.5) +
    ggplot2::labs(title = title, y = ylab, x = NULL) +
    theme_structural_minimal()
}

# ---- Plot-saving helpers ------------------------------------------------------

#' Save a ggplot with a light background
#'
#' @param plot ggplot object to save.
#' @param filename Output file name, including extension.
#' @param width Plot width in inches.
#' @param height Plot height in inches.
#' @param dpi Resolution in dots per inch for raster output.
#' @param bg Background color.
#'
#' @return Invisibly returns the written filename.
#' @export
save_plot_light <- function(plot,
                            filename,
                            width = 7,
                            height = 5,
                            dpi = 300,
                            bg = "white") {
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)

  ggplot2::ggsave(
    filename = filename,
    plot = plot,
    width = width,
    height = height,
    dpi = dpi,
    bg = bg
  )

  invisible(filename)
}

#' Save a ggplot to PNG with light background
#'
#' Backward-compatible wrapper retained for older visualization scripts.
#'
#' @param plot ggplot object to save.
#' @param filename Output PNG file name.
#' @param width Plot width in inches.
#' @param height Plot height in inches.
#' @param dpi Resolution in dots per inch.
#'
#' @return Invisibly returns the written filename.
#' @export
save_png_light <- function(plot, filename, width = 7, height = 5, dpi = 300) {
  save_plot_light(
    plot = plot,
    filename = filename,
    width = width,
    height = height,
    dpi = dpi,
    bg = "white"
  )
}

#' Save a ggplot as PNG and optionally PDF
#'
#' @param plot ggplot object.
#' @param output_dir Output directory.
#' @param basename_no_ext File basename without extension.
#' @param width Plot width in inches.
#' @param height Plot height in inches.
#' @param dpi PNG resolution.
#' @param save_pdf Whether to also save a PDF companion.
#' @param bg Background color.
#'
#' @return Named character vector of written file paths.
#' @export
save_plot_pair <- function(plot,
                           output_dir,
                           basename_no_ext,
                           width = 7,
                           height = 5,
                           dpi = 300,
                           save_pdf = TRUE,
                           bg = "white") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  png_path <- file.path(output_dir, paste0(basename_no_ext, ".png"))

  save_plot_light(
    plot = plot,
    filename = png_path,
    width = width,
    height = height,
    dpi = dpi,
    bg = bg
  )

  written <- c(png = png_path)

  if (isTRUE(save_pdf)) {
    pdf_path <- file.path(output_dir, paste0(basename_no_ext, ".pdf"))

    save_plot_light(
      plot = plot,
      filename = pdf_path,
      width = width,
      height = height,
      dpi = dpi,
      bg = bg
    )

    written <- c(written, pdf = pdf_path)
  }

  written
}
