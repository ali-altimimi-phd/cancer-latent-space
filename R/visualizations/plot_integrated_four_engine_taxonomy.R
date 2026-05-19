# ==============================================================================
# File: R/visualizations/plot_integrated_four_engine_taxonomy.R
# Purpose: Build the integrated four-engine cancer taxonomy plot from structural
#          phenotype synthesis tables.
# Project: Global Cancer Complexity / Structural Inference
# ------------------------------------------------------------------------------
# Architectural role:
#   structural phenotype synthesis -> taxonomic result figure
#
# This file intentionally contains only the integrated taxonomy plot constructor
# and its local helpers. It is sourced by plot_structural_synthesis_diagnostics.R,
# but can also be sourced independently after the usual visualization libraries.
#
# Default statistical design:
#   x-axis = log2 effective-rank ratio, tumor / normal, from the complexity engine
#   y-axis = MP spectral entropy delta, tumor - normal
#   size   = VAE latent centroid distance
#
# Rationale:
#   Effective-rank ratio is the default complexity coordinate because it is less
#   brittle than condition-number change and more directly represents distributed
#   dimensional organization. Condition-number-based coordinates remain available
#   as explicit alternatives.
# ==============================================================================

# ---- Local helper utilities ---------------------------------------------------

.first_col_local <- function(df, candidates, required = TRUE, label = "column") {
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

.compute_integrated_taxonomy_x <- function(plot_wide,
                                           x_metric = "effective_rank_log2_ratio") {
  x_metric <- match.arg(
    x_metric,
    choices = c(
      "effective_rank_log2_ratio",
      "effrank_delta",
      "kappa_delta",
      "composite_kappa_delta",
      "mp_participation_ratio_delta"
    )
  )

  if (identical(x_metric, "effective_rank_log2_ratio")) {
    normal_col <- .first_col_local(
      plot_wide,
      candidates = c(
        "complexity__effrank_normal",
        "complexity__effective_rank_normal",
        "complexity__eff_rank_normal",
        "complexity__Eff Rank_normal"
      ),
      required = FALSE,
      label = "normal effective-rank column"
    )

    tumor_col <- .first_col_local(
      plot_wide,
      candidates = c(
        "complexity__effrank_tumor",
        "complexity__effective_rank_tumor",
        "complexity__eff_rank_tumor",
        "complexity__Eff Rank_tumor"
      ),
      required = FALSE,
      label = "tumor effective-rank column"
    )

    if (!is.na(normal_col) && !is.na(tumor_col)) {
      return(list(
        values = log2(plot_wide[[tumor_col]] / plot_wide[[normal_col]]),
        label = "log2 effective-rank ratio (tumor / normal) [complexity engine]",
        zero_label = "No effective-rank change",
        metric = x_metric,
        source_columns = c(normal = normal_col, tumor = tumor_col)
      ))
    }

    fallback_col <- .first_col_local(
      plot_wide,
      candidates = c(
        "complexity__effrank_delta",
        "complexity__effective_rank_delta",
        "complexity__eff_rank_delta",
        "complexity__Eff Rank_delta"
      ),
      required = TRUE,
      label = "effective-rank delta fallback column"
    )

    warning(
      "Effective-rank normal/tumor columns were unavailable; ",
      "using effective-rank delta as the integrated taxonomy x-axis.",
      call. = FALSE
    )

    return(list(
      values = plot_wide[[fallback_col]],
      label = "Delta effective rank (tumor - normal) [complexity engine]",
      zero_label = "No effective-rank change",
      metric = "effrank_delta_fallback",
      source_columns = c(delta = fallback_col)
    ))
  }

  column_map <- list(
    effrank_delta = list(
      candidates = c(
        "complexity__effrank_delta",
        "complexity__effective_rank_delta",
        "complexity__eff_rank_delta",
        "complexity__Eff Rank_delta"
      ),
      label = "Delta effective rank (tumor - normal) [complexity engine]"
    ),
    kappa_delta = list(
      candidates = c("complexity__kappa_delta"),
      label = "Delta condition number (tumor - normal) [complexity engine]"
    ),
    composite_kappa_delta = list(
      candidates = c("complexity__composite_kappa_delta"),
      label = "Delta composite kappa (tumor - normal) [complexity engine]"
    ),
    mp_participation_ratio_delta = list(
      candidates = c("mp__participation_ratio_delta"),
      label = "Delta MP participation ratio (tumor - normal)"
    )
  )

  selected <- column_map[[x_metric]]
  x_col <- .first_col_local(
    plot_wide,
    candidates = selected$candidates,
    required = TRUE,
    label = paste(x_metric, "column")
  )

  list(
    values = plot_wide[[x_col]],
    label = selected$label,
    zero_label = "No x-axis structural change",
    metric = x_metric,
    source_columns = c(x = x_col)
  )
}

# ---- Plot constructor ---------------------------------------------------------

#' Plot integrated four-engine cancer taxonomy
#'
#' @param plot_wide Wide structural phenotype table, usually the latent-aligned
#'   synthesis table or a filtered version of structural_phenotype_wide.csv.
#' @param x_metric Complexity coordinate for the x-axis. Default is
#'   effective_rank_log2_ratio.
#' @param add_quadrant_labels Whether to annotate the four structural quadrants.
#' @param title Plot title.
#' @param subtitle Plot subtitle.
#'
#' @return A ggplot object, or NULL if latent centroid distance is unavailable.
#' @export
plot_integrated_four_engine_taxonomy <- function(
    plot_wide,
    x_metric = "effective_rank_log2_ratio",
    add_quadrant_labels = TRUE,
    title = "Integrated four-engine cancer taxonomy",
    subtitle = "Complexity reorganization x Marchenko-Pastur spectral entropy; size = VAE centroid distance") {

  mp_entropy_col <- .first_col_local(
    plot_wide,
    candidates = c("mp__spectral_entropy_delta", "mp__entropy_spectral_delta", "mp__spectral_delta"),
    label = "MP spectral entropy delta column"
  )

  centroid_col <- .first_col_local(
    plot_wide,
    candidates = c(
      "latent__centroid_distance",
      "latent__vae_centroid_distance",
      "latent__centroid_distance_delta"
    ),
    required = FALSE,
    label = "latent centroid distance column"
  )

  if (is.na(centroid_col)) {
    return(NULL)
  }

  x_info <- .compute_integrated_taxonomy_x(
    plot_wide = plot_wide,
    x_metric = x_metric
  )

  plot_df <- plot_wide |>
    dplyr::mutate(.integrated_taxonomy_x = x_info$values) |>
    dplyr::filter(
      !is.na(.data$.integrated_taxonomy_x),
      !is.na(.data[[mp_entropy_col]]),
      !is.na(.data[[centroid_col]])
    )

  if (nrow(plot_df) == 0) {
    return(NULL)
  }

  p <- ggplot2::ggplot(
    plot_df,
    ggplot2::aes(
      x = .data$.integrated_taxonomy_x,
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
      title = title,
      subtitle = subtitle,
      x = x_info$label,
      y = "Delta MP spectral entropy (tumor - normal)",
      color = "Group",
      shape = "Group",
      size = "Centroid distance\n(VAE latent)",
      caption = paste(
        "Quadrants use x-axis structural complexity change and y-axis MP spectral entropy change.",
        "Left/right: loss/gain of distributed complexity; down/up: lower/higher MP spectral entropy.",
        "Latent size is valid only for the VAE-aligned feature regime.",
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
    x_min <- min(plot_df$.integrated_taxonomy_x, na.rm = TRUE)
    x_max <- max(plot_df$.integrated_taxonomy_x, na.rm = TRUE)
    y_min <- min(plot_df[[mp_entropy_col]], na.rm = TRUE)
    y_max <- max(plot_df[[mp_entropy_col]], na.rm = TRUE)

    quad_labels <- tibble::tibble(
      x = c(x_max, x_min, x_min, x_max),
      y = c(y_max, y_max, y_min, y_min),
      label = c(
        "Gained complexity\nhigh entropy",
        "Lost complexity\nhigh entropy",
        "Lost complexity\nlow entropy",
        "Gained complexity\nlow entropy"
      ),
      hjust = c(1.05, -0.05, -0.05, 1.05),
      vjust = c(1.10, 1.10, -0.10, -0.10)
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
        color = "grey45",
        size = 3,
        lineheight = 0.95
      )
  }

  p
}
