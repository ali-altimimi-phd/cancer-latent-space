# ------------------------------------------------------------------------------
# File: build_latent_postprocessing_outputs.R
# Purpose: Build combined latent/dissertation metrics, construct latent taxonomy,
#   and summarize latent geometric regimes after notebook execution
# Role: Latent-space postprocessing utility
# Pipeline: Analysis
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Run Latent-Space Postprocessing
#'
#' @description
#' Runs the complete post-notebook latent-space postprocessing workflow in the
#' parent analysis R process. This replaces the older three-script Rscript
#' pattern and allows all messages to be written through the parent pipeline
#' logger.
#'
#' @param logger Logger object returned by `start_log()`.
#' @param latent_project_dir Directory containing the latent-space project.
#' @param reports_dir Directory containing cleaned dissertation-era reports.
#' @param table_dir Output directory for postprocessing tables.
#' @param plot_dir Output directory for postprocessing plots.
#' @param chip_id Chip/platform identifier to retain from legacy metrics.
#' @param mode Legacy analysis mode to retain, usually `"ALL"`.
#' @param gene_set_name Legacy gene-set name to retain, usually `"ALL"`.
#'
#' @return Invisibly returns a named list containing the combined metrics,
#'   taxonomy table, regime counts, and cross-tabulation.
#' @export
run_latent_postprocessing <- function(logger = NULL,
                                      latent_project_dir = here::here("output", "global_cancer"),
                                      reports_dir = here::here("output", "global_cancer", "reports"),
                                      table_dir = file.path(latent_project_dir, "tables", "latent"),
                                      plot_dir = file.path(latent_project_dir, "plots", "latent"),
                                      chip_id = "hu35ksuba",
                                      mode = "ALL",
                                      gene_set_name = "ALL") {
  suppressPackageStartupMessages({
    requireNamespace("readr", quietly = TRUE)
    requireNamespace("dplyr", quietly = TRUE)
    requireNamespace("ggplot2", quietly = TRUE)
  })

  log_it <- function(message, section = "LATENT_POST") {
    if (!is.null(logger)) {
      logger$log(message, section = section)
    } else {
      message(sprintf("[%s] %s", section, message))
    }
  }

  require_file <- function(path, label = "input file") {
    if (!file.exists(path)) {
      log_it(sprintf("Missing %s: %s", label, path), section = "ERROR")
      stop(sprintf("Missing %s: %s", label, path), call. = FALSE)
    }
    invisible(path)
  }

  clean_key <- function(x) {
    x |>
      as.character() |>
      trimws() |>
      toupper() |>
      gsub(" ", "", x = _, fixed = TRUE)
  }

  dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

  build_combined_metrics <- function() {
    log_it("Building combined dissertation + latent metrics table.")

    latent_n5_path <- file.path(
      latent_project_dir, "tables", "latent", "latent_comparison_metrics.csv"
    )
    latent_n6_path <- file.path(
      latent_project_dir, "tables", "latent", "notebook_06",
      "final_notebook_06_group_summary.csv"
    )
    entropy_path <- file.path(reports_dir, "entropy_cleaned.csv")
    complexity_path <- file.path(reports_dir, "complexity_cleaned.csv")

    require_file(latent_n5_path, "Notebook 5 latent comparison metrics")
    require_file(latent_n6_path, "Notebook 6 group summary")
    require_file(entropy_path, "cleaned entropy report")
    require_file(complexity_path, "cleaned complexity report")

    latent_n5 <- readr::read_csv(latent_n5_path, show_col_types = FALSE)
    latent_n6 <- readr::read_csv(latent_n6_path, show_col_types = FALSE)
    entropy <- readr::read_csv(entropy_path, show_col_types = FALSE)
    complexity <- readr::read_csv(complexity_path, show_col_types = FALSE)

    for (nm in c("latent_n5", "latent_n6", "entropy", "complexity")) {
      obj <- get(nm)
      if (!"comparison" %in% names(obj)) {
        stop(sprintf("Object '%s' is missing required column: comparison", nm), call. = FALSE)
      }
    }

    latent_n5$comparison <- clean_key(latent_n5$comparison)
    latent_n6$comparison <- clean_key(latent_n6$comparison)
    entropy$comparison <- clean_key(entropy$comparison)
    complexity$comparison <- clean_key(complexity$comparison)

    entropy_summary <- entropy |>
      dplyr::filter(
        toupper(.data$mode) == toupper(mode),
        toupper(.data$gene_set_normalized) == toupper(gene_set_name),
        toupper(.data$gene_set_name) == toupper(gene_set_name)
      ) |>
      dplyr::select(dplyr::any_of(c(
        "comparison",
        "shannon_1", "shannon_2", "shannon_delta",
        "spectral_1", "spectral_2", "spectral_delta",
        "direction", "p_perm"
      ))) |>
      dplyr::distinct(.data$comparison, .keep_all = TRUE)

    if ("direction" %in% names(entropy_summary)) {
      entropy_summary <- dplyr::rename(entropy_summary, entropy_direction = .data$direction)
    }
    if ("p_perm" %in% names(entropy_summary)) {
      entropy_summary <- dplyr::rename(entropy_summary, entropy_p_perm = .data$p_perm)
    }

    complexity_summary <- complexity |>
      dplyr::filter(
        toupper(.data$mode) == toupper(mode),
        toupper(.data$gene_set_normalized) == toupper(gene_set_name),
        toupper(.data$gene_set_name) == toupper(gene_set_name)
      ) |>
      dplyr::select(dplyr::any_of(c(
        "comparison",
        "svd κ 1", "svd κ 2", "δ svd κ",
        "effrank_1", "effrank_2", "effrank_delta",
        "pr_1", "pr_2", "pr_delta",
        "κ_composite_1", "κ_composite_2",
        "direction", "p_perm"
      ))) |>
      dplyr::distinct(.data$comparison, .keep_all = TRUE)

    if ("direction" %in% names(complexity_summary)) {
      complexity_summary <- dplyr::rename(complexity_summary, complexity_direction_legacy = .data$direction)
    }
    if ("p_perm" %in% names(complexity_summary)) {
      complexity_summary <- dplyr::rename(complexity_summary, complexity_p_perm = .data$p_perm)
    }

    log_it(sprintf("Entropy summary rows: %d", nrow(entropy_summary)))
    log_it(sprintf("Complexity summary rows: %d", nrow(complexity_summary)))

    combined <- latent_n5 |>
      dplyr::left_join(latent_n6, by = "comparison", suffix = c("_n5", "_n6")) |>
      dplyr::left_join(entropy_summary, by = "comparison") |>
      dplyr::left_join(complexity_summary, by = "comparison")

    combined_csv <- file.path(table_dir, "master_combined_metrics.csv")
    combined_rds <- file.path(table_dir, "master_combined_metrics.rds")

    readr::write_csv(combined, combined_csv)
    saveRDS(combined, combined_rds)

    log_it(sprintf("Combined metrics rows: %d", nrow(combined)))
    log_it(sprintf("Unique comparisons: %d", dplyr::n_distinct(combined$comparison)))
    log_it(sprintf("Wrote combined metrics CSV: %s", combined_csv))
    log_it(sprintf("Wrote combined metrics RDS: %s", combined_rds))

    combined
  }

  build_latent_taxonomy_table <- function(combined) {
    log_it("Building latent taxonomy table.")

    required <- c("comparison", "mean_silhouette", "mean_same_group_neighbor_fraction")
    missing <- setdiff(required, names(combined))
    if (length(missing) > 0) {
      stop(sprintf("Combined metrics missing required columns: %s", paste(missing, collapse = ", ")), call. = FALSE)
    }

    sil_q <- stats::quantile(
      combined$mean_silhouette,
      probs = c(0.25, 0.5, 0.75),
      na.rm = TRUE
    )
    nn_q <- stats::quantile(
      combined$mean_same_group_neighbor_fraction,
      probs = c(0.25, 0.5, 0.75),
      na.rm = TRUE
    )

    classify_sep <- function(sil, nn) {
      if (is.na(sil) || is.na(nn)) return("unknown")
      if (sil >= sil_q[[3]] && nn >= nn_q[[3]]) return("relatively_separated")
      if (sil >= sil_q[[2]] || nn >= nn_q[[2]]) return("moderately_structured")
      if (sil >= sil_q[[1]] || nn >= nn_q[[1]]) return("weak_structure")
      "overlap"
    }

    delta_col <- NULL
    candidate_delta_cols <- c("effrank_delta", "δ svd κ", "pr_delta_n5", "pr_delta")
    for (candidate in candidate_delta_cols) {
      if (candidate %in% names(combined)) {
        delta_col <- candidate
        break
      }
    }
    if (is.null(delta_col)) {
      stop(
        "No usable complexity delta column found. Expected one of: effrank_delta, δ svd κ, pr_delta_n5, pr_delta",
        call. = FALSE
      )
    }

    log_it(sprintf("Silhouette quartiles: %s", paste(round(sil_q, 4), collapse = ", ")))
    log_it(sprintf("Neighbor-fraction quartiles: %s", paste(round(nn_q, 4), collapse = ", ")))
    log_it(sprintf("Using complexity delta column: %s", delta_col))

    taxonomy <- combined |>
      dplyr::mutate(
        separation_class = mapply(
          classify_sep,
          .data$mean_silhouette,
          .data$mean_same_group_neighbor_fraction
        ),
        complexity_delta_used = .data[[delta_col]],
        complexity_direction = dplyr::case_when(
          is.na(.data$complexity_delta_used) ~ "unknown",
          .data$complexity_delta_used > 0 ~ "increase",
          .data$complexity_delta_used < 0 ~ "decrease",
          TRUE ~ "no_change"
        ),
        regime = dplyr::case_when(
          .data$complexity_direction == "increase" & .data$separation_class == "relatively_separated" ~ "structured_divergence",
          .data$complexity_direction == "increase" & .data$separation_class %in% c("moderately_structured", "weak_structure") ~ "partially_structured_expansion",
          .data$complexity_direction == "increase" & .data$separation_class == "overlap" ~ "noisy_expansion",
          .data$complexity_direction == "decrease" & .data$separation_class == "relatively_separated" ~ "constrained_specialization",
          .data$complexity_direction == "decrease" & .data$separation_class %in% c("moderately_structured", "weak_structure") ~ "partially_constrained_reorganization",
          .data$complexity_direction == "decrease" & .data$separation_class == "overlap" ~ "degenerate_overlap",
          TRUE ~ "mixed"
        )
      )

    taxonomy_csv <- file.path(table_dir, "latent_taxonomy_table.csv")
    taxonomy_rds <- file.path(table_dir, "latent_taxonomy_table.rds")

    readr::write_csv(taxonomy, taxonomy_csv)
    saveRDS(taxonomy, taxonomy_rds)

    log_it(sprintf("Wrote latent taxonomy CSV: %s", taxonomy_csv))
    log_it(sprintf("Wrote latent taxonomy RDS: %s", taxonomy_rds))

    taxonomy
  }

  summarize_latent_regimes <- function(taxonomy) {
    log_it("Summarizing latent regimes.")

    summary_counts <- taxonomy |>
      dplyr::count(.data$separation_class, .data$regime, name = "n") |>
      dplyr::arrange(.data$separation_class, .data$regime)

    cross_tab <- taxonomy |>
      dplyr::count(.data$complexity_direction, .data$separation_class, name = "n") |>
      dplyr::arrange(.data$complexity_direction, .data$separation_class)

    regime_summary <- taxonomy |>
      dplyr::group_by(.data$regime) |>
      dplyr::summarise(
        n = dplyr::n(),
        mean_silhouette = mean(.data$mean_silhouette, na.rm = TRUE),
        mean_same_group_neighbor_fraction = mean(.data$mean_same_group_neighbor_fraction, na.rm = TRUE),
        mean_complexity_delta_used = mean(.data$complexity_delta_used, na.rm = TRUE),
        .groups = "drop"
      ) |>
      dplyr::arrange(dplyr::desc(.data$n), .data$regime)

    readr::write_csv(summary_counts, file.path(table_dir, "regime_counts.csv"))
    readr::write_csv(cross_tab, file.path(table_dir, "complexity_vs_separation.csv"))
    readr::write_csv(regime_summary, file.path(table_dir, "latent_regime_summary_table.csv"))

    p <- ggplot2::ggplot(
      taxonomy,
      ggplot2::aes(x = .data$separation_class, fill = .data$complexity_direction)
    ) +
      ggplot2::geom_bar(position = "dodge") +
      ggplot2::theme_minimal() +
      ggplot2::labs(
        title = "Separation vs Complexity Direction",
        x = "Latent separation class",
        y = "Number of comparisons",
        fill = "Complexity direction"
      )

    plot_path <- file.path(plot_dir, "separation_vs_complexity.png")
    ggplot2::ggsave(
      filename = plot_path,
      plot = p,
      width = 8,
      height = 5,
      dpi = 300
    )

    log_it(sprintf("Wrote regime counts: %s", file.path(table_dir, "regime_counts.csv")))
    log_it(sprintf("Wrote complexity/separation cross-tab: %s", file.path(table_dir, "complexity_vs_separation.csv")))
    log_it(sprintf("Wrote regime summary table: %s", file.path(table_dir, "latent_regime_summary_table.csv")))
    log_it(sprintf("Wrote postprocessing plot: %s", plot_path))

    list(
      summary_counts = summary_counts,
      cross_tab = cross_tab,
      regime_summary = regime_summary,
      plot_path = plot_path
    )
  }

  log_it("Starting latent-space postprocessing.")
  combined <- build_combined_metrics()
  taxonomy <- build_latent_taxonomy_table(combined)
  regimes <- summarize_latent_regimes(taxonomy)
  log_it("Latent-space postprocessing completed successfully.")

  invisible(list(
    combined = combined,
    taxonomy = taxonomy,
    regimes = regimes
  ))
}
