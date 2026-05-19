# ==============================================================================
# File: R/synthesis/build_structural_phenotype_table.R
# Purpose: Build unified structural phenotype tables from MP, complexity, entropy,
#          and optional latent-geometry outputs.
# Project: Global Cancer Complexity / Structural Inference
# ------------------------------------------------------------------------------
# Notes:
#   - Pipeline-safe function: builds tables only; no plotting.
#   - Intended to be called from Stage 9: Synthesis.
#   - Produces a long descriptor-first structural phenotype table plus optional
#     wide and heatmap-ready derivatives.
#   - Complexity and entropy may include inferential columns when resampling was
#     enabled. MP is currently descriptive only.
#   - Latent geometry is treated as a Python-backed structural engine layer.
#   - When include_latent = TRUE, the function searches known latent comparison
#     metric locations and joins available latent geometry rows into the same
#     descriptor-long structural phenotype table.
#   - If no latent comparison metrics are found and latent_required = FALSE,
#     synthesis continues with MP, complexity, and entropy only.# ===============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(tibble)
})

# ---- Small helpers ------------------------------------------------------------

`%||%` <- function(x, y) {
  if (is.null(x))
    y
  else
    x
}

.safe_read_rds <- function(path,
                           required = TRUE,
                           label = basename(path)) {
  if (!file.exists(path)) {
    if (isTRUE(required)) {
      stop("Required file not found for ", label, ": ", path, call. = FALSE)
    }
    return(NULL)
  }
  readRDS(path)
}

.safe_read_csv <- function(path,
                           required = FALSE,
                           label = basename(path)) {
  if (!file.exists(path)) {
    if (isTRUE(required)) {
      stop("Required CSV not found for ", label, ": ", path, call. = FALSE)
    }
    return(NULL)
  }
  readr::read_csv(path, show_col_types = FALSE)
}

.first_existing_col <- function(df, candidates) {
  hit <- intersect(candidates, names(df))
  if (length(hit) == 0)
    return(NA_character_)
  hit[[1]]
}

.add_missing_context_cols <- function(df) {
  for (nm in c("chip", "filter_regime", "group", "comparison")) {
    if (!nm %in% names(df))
      df[[nm]] <- NA_character_
  }
  df
}

.metric_direction <- function(value) {
  dplyr::case_when(
    is.na(value) ~ NA_character_,
    value > 0 ~ "tumor_greater",
    value < 0 ~ "tumor_lower",
    TRUE ~ "no_change"
  )
}

.select_available <- function(df, cols) {
  dplyr::select(df, dplyr::any_of(cols))
}

.to_metric_long <- function(df,
                            engine,
                            metric_cols,
                            p_value_map = NULL,
                            p_fdr_map = NULL,
                            inference_status = "descriptive",
                            resampling_method = NA_character_,
                            n_resamples = NA_integer_,
                            runtime_sec = NA_real_,
                            source_file = NA_character_,
                            source_layer = "structural_engine",
                            notes = NA_character_) {
  df <- .add_missing_context_cols(df)
  metric_cols <- intersect(metric_cols, names(df))
  
  if (length(metric_cols) == 0) {
    return(tibble())
  }
  
  base_cols <- c("chip", "filter_regime", "group", "comparison")
  
  long <- df |>
    dplyr::select(dplyr::all_of(base_cols), dplyr::all_of(metric_cols)) |>
    tidyr::pivot_longer(
      cols = dplyr::all_of(metric_cols),
      names_to = "metric",
      values_to = "value"
    ) |>
    dplyr::mutate(
      engine = engine,
      direction = .metric_direction(value),
      inference_status = inference_status,
      resampling_method = resampling_method,
      n_resamples = n_resamples,
      runtime_sec = runtime_sec,
      source_file = source_file,
      source_layer = source_layer,
      notes = notes,
      .before = "metric"
    )
  
  long$p_value <- NA_real_
  long$p_fdr <- NA_real_
  
  if (!is.null(p_value_map)) {
    for (metric_name in names(p_value_map)) {
      p_col <- p_value_map[[metric_name]]
      if (p_col %in% names(df)) {
        p_df <- df |>
          dplyr::select(dplyr::all_of(base_cols), p_value = dplyr::all_of(p_col)) |>
          dplyr::mutate(metric = metric_name)
        
        long <- long |>
          dplyr::left_join(p_df,
                           by = c(base_cols, "metric"),
                           suffix = c("", ".joined")) |>
          dplyr::mutate(p_value = dplyr::coalesce(.data$p_value, .data$p_value.joined)) |>
          dplyr::select(-dplyr::any_of("p_value.joined"))
      }
    }
  }
  
  if (!is.null(p_fdr_map)) {
    for (metric_name in names(p_fdr_map)) {
      fdr_col <- p_fdr_map[[metric_name]]
      if (fdr_col %in% names(df)) {
        fdr_df <- df |>
          dplyr::select(dplyr::all_of(base_cols), p_fdr = dplyr::all_of(fdr_col)) |>
          dplyr::mutate(metric = metric_name)
        
        long <- long |>
          dplyr::left_join(
            fdr_df,
            by = c(base_cols, "metric"),
            suffix = c("", ".joined")
          ) |>
          dplyr::mutate(p_fdr = dplyr::coalesce(.data$p_fdr, .data$p_fdr.joined)) |>
          dplyr::select(-dplyr::any_of("p_fdr.joined"))
      }
    }
  }
  
  long |>
    dplyr::relocate(
      chip,
      filter_regime,
      group,
      comparison,
      engine,
      metric,
      value,
      direction,
      inference_status,
      p_value,
      p_fdr,
      resampling_method,
      n_resamples,
      runtime_sec,
      source_file,
      source_layer,
      notes
    ) |>
    dplyr::arrange(chip, filter_regime, group, comparison, engine, metric)
}

.zscore <- function(x) {
  if (all(is.na(x)))
    return(rep(NA_real_, length(x)))
  sx <- stats::sd(x, na.rm = TRUE)
  if (is.na(sx) || sx == 0)
    return(rep(NA_real_, length(x)))
  as.numeric(scale(x))
}

.make_latent_aligned_tables <- function(phenotype_long,
                                        latent_aligned_chip = "hu35ksuba",
                                        latent_aligned_filter_regime = "variance_global") {
  structural_aligned <- phenotype_long |>
    dplyr::filter(.data$engine != "latent") |>
    dplyr::filter(.data$chip == latent_aligned_chip) |>
    dplyr::filter(.data$filter_regime == latent_aligned_filter_regime)
  
  latent_group_map <- phenotype_long |>
    dplyr::filter(.data$engine != "latent") |>
    dplyr::filter(.data$chip == latent_aligned_chip) |>
    dplyr::filter(.data$filter_regime == latent_aligned_filter_regime) |>
    dplyr::distinct(.data$comparison, structural_group = .data$group)
  
  latent_aligned <- phenotype_long |>
    dplyr::filter(.data$engine == "latent") |>
    dplyr::mutate(
      chip = latent_aligned_chip,
      filter_regime = latent_aligned_filter_regime
    ) |>
    dplyr::left_join(latent_group_map, by = "comparison") |>
    dplyr::mutate(
      group = dplyr::coalesce(.data$structural_group, .data$group)
    ) |>
    dplyr::select(-dplyr::any_of("structural_group"))
  
  aligned <- dplyr::bind_rows(structural_aligned, latent_aligned) |>
    dplyr::filter(!is.na(.data$comparison))
  
  aligned_wide <- aligned |>
    dplyr::mutate(metric_key = paste(engine, metric, sep = "__")) |>
    dplyr::select(chip, filter_regime, group, comparison, metric_key, value) |>
    dplyr::distinct() |>
    tidyr::pivot_wider(names_from = metric_key, values_from = value) |>
    dplyr::arrange(chip, filter_regime, group, comparison)
  
  aligned_heatmap_long <- aligned |>
    dplyr::group_by(engine, metric) |>
    dplyr::mutate(z_score = .zscore(value)) |>
    dplyr::ungroup() |>
    dplyr::arrange(chip, filter_regime, group, comparison, engine, metric)
  
  aligned_summary <- aligned |>
    dplyr::group_by(chip, filter_regime, engine, metric, inference_status) |>
    dplyr::summarise(
      n_comparisons = dplyr::n_distinct(comparison),
      n_values = sum(!is.na(value)),
      mean_value = mean(value, na.rm = TRUE),
      median_value = stats::median(value, na.rm = TRUE),
      sd_value = stats::sd(value, na.rm = TRUE),
      n_tumor_greater = sum(direction == "tumor_greater", na.rm = TRUE),
      n_tumor_lower = sum(direction == "tumor_lower", na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(dplyr::across(
      c(mean_value, median_value, sd_value),
      ~ ifelse(is.nan(.x), NA_real_, .x)
    ))
  
  list(
    long = aligned,
    wide = aligned_wide,
    heatmap_long = aligned_heatmap_long,
    summary = aligned_summary
  )
}


# ---- Main function ------------------------------------------------------------

#' Build unified structural phenotype tables
#'
#' @param study_name Study/output namespace, default "global_cancer".
#' @param structural_rdata_dir Directory containing structural engine RDS outputs.
#' @param latent_table_dir Directory containing optional latent comparison metrics.
#' @param synthesis_table_dir Output directory for synthesis CSV tables.
#' @param synthesis_rdata_dir Output directory for synthesis RDS objects.
#' @param chips Character vector of chips to include; NULL keeps all chips.
#' @param filter_regimes Character vector of filter regimes to include; NULL keeps all.
#' @param include_latent Whether to include latent geometry metrics when available.
#' @param latent_required Whether missing latent metrics should stop execution.
#' @param latent_aligned_chip Chip used by the current latent/VAE feature space.
#' @param latent_aligned_filter_regime Structural filter regime used by the current latent/VAE feature space.
#' @param write_outputs Whether to write CSV/RDS outputs.
#' @param logger Optional project logger with $log(message, section = ...).
#'
#' @return A list containing long, wide, heatmap_long, and summary tables.
#' @export
build_structural_phenotype_table <- function(study_name = "global_cancer",
                                             structural_rdata_dir = here::here("output", study_name, "structural_inference", "RData"),
                                             latent_table_dir = here::here("output",
                                                                           study_name,
                                                                           "structural_inference",
                                                                           "tables",
                                                                           "latent"),
                                             synthesis_table_dir = here::here("output",
                                                                              study_name,
                                                                              "structural_inference",
                                                                              "tables",
                                                                              "synthesis"),
                                             synthesis_rdata_dir = here::here("output",
                                                                              study_name,
                                                                              "structural_inference",
                                                                              "RData",
                                                                              "synthesis"),
                                             chips = NULL,
                                             filter_regimes = NULL,
                                             include_latent = TRUE,
                                             latent_required = FALSE,
                                             latent_aligned_chip = "hu35ksuba",
                                             latent_aligned_filter_regime = "variance_global",
                                             write_outputs = TRUE,
                                             logger = NULL) {
  log_msg <- function(msg) {
    if (!is.null(logger) && !is.null(logger$log)) {
      logger$log(msg, section = "SYNTHESIS")
    } else {
      message(msg)
    }
  }
  
  if (isTRUE(write_outputs)) {
    dir.create(synthesis_table_dir,
               recursive = TRUE,
               showWarnings = FALSE)
    
    dir.create(synthesis_rdata_dir,
               recursive = TRUE,
               showWarnings = FALSE)
  }
  
  log_msg("📦 Loading structural engine outputs...")
  
  complexity_path <- file.path(structural_rdata_dir, "complexity_results.rds")
  entropy_path <- file.path(structural_rdata_dir, "entropy_results.rds")
  mp_path <- file.path(structural_rdata_dir, "mp_spectral_results.rds")
  
  latent_candidate_paths <- c(
    file.path(latent_table_dir, "latent_comparison_metrics.csv")
  )
  
  latent_path <- latent_candidate_paths[file.exists(latent_candidate_paths)][1]
  
  if (is.na(latent_path)) {
    latent_path <- latent_candidate_paths[[1]]
  }
  
  complexity_results <- .safe_read_rds(complexity_path, required = TRUE, label = "complexity_results")
  entropy_results <- .safe_read_rds(entropy_path, required = TRUE, label = "entropy_results")
  mp_spectral_results <- .safe_read_rds(mp_path, required = TRUE, label = "mp_spectral_results")
  
  complexity_summary <- complexity_results$summary
  entropy_summary <- entropy_results$summary
  mp_deltas <- mp_spectral_results$deltas
  
  if (is.null(complexity_summary)) {
    stop("complexity_results.rds does not contain $summary", call. = FALSE)
  }
  if (is.null(entropy_summary)) {
    stop("entropy_results.rds does not contain $summary", call. = FALSE)
  }
  if (is.null(mp_deltas)) {
    stop("mp_spectral_results.rds does not contain $deltas", call. = FALSE)
  }
  
  if (!is.null(chips)) {
    complexity_summary <- complexity_summary |> dplyr::filter(.data$chip %in% chips)
    entropy_summary <- entropy_summary |> dplyr::filter(.data$chip %in% chips)
    mp_deltas <- mp_deltas |> dplyr::filter(.data$chip %in% chips)
  }
  
  if (!is.null(filter_regimes)) {
    complexity_summary <- complexity_summary |> dplyr::filter(.data$filter_regime %in% filter_regimes)
    entropy_summary <- entropy_summary |> dplyr::filter(.data$filter_regime %in% filter_regimes)
    mp_deltas <- mp_deltas |> dplyr::filter(.data$filter_regime %in% filter_regimes)
  }
  
  log_msg(
    sprintf(
      "ℹ️ Regimes found after loading: complexity={%s}; entropy={%s}; mp={%s}",
      paste(sort(
        unique(complexity_summary$filter_regime)
      ), collapse = ", "),
      paste(sort(
        unique(entropy_summary$filter_regime)
      ), collapse = ", "),
      paste(sort(unique(
        mp_deltas$filter_regime
      )), collapse = ", ")
    )
  )
  
  log_msg("🧬 Converting engine outputs to descriptor-long format...")
  
  complexity_long <- .to_metric_long(
    df = complexity_summary,
    engine = "complexity",
    metric_cols = c(
      "kappa_delta",
      "effrank_delta",
      "pr_delta",
      "eig_entropy_delta",
      "anisotropy_delta"
    ),
    p_value_map = c(kappa_delta = "p_perm", effrank_delta = "p_perm"),
    p_fdr_map = c(kappa_delta = "p_perm_fdr", effrank_delta = "p_perm_fdr"),
    inference_status = if ("p_perm" %in% names(complexity_summary))
      "permutation_available"
    else
      "descriptive",
    resampling_method = if ("p_perm" %in% names(complexity_summary))
      "permutation"
    else
      NA_character_,
    source_file = basename(complexity_path),
    source_layer = "complexity_engine"
  )
  
  entropy_long <- .to_metric_long(
    df = entropy_summary,
    engine = "entropy",
    metric_cols = c(
      "shannon_delta",
      "spectral_delta",
      "entropy_spectral_delta"
    ),
    p_value_map = c(
      shannon_delta = "p_perm_shannon",
      spectral_delta = "p_perm_spectral",
      entropy_spectral_delta = "p_perm_spectral"
    ),
    p_fdr_map = c(
      shannon_delta = "p_perm_shannon_fdr",
      spectral_delta = "p_perm_spectral_fdr",
      entropy_spectral_delta = "p_perm_spectral_fdr"
    ),
    inference_status = if (any(
      c("p_perm_shannon", "p_perm_spectral") %in% names(entropy_summary)
    )) {
      "permutation_available"
    } else {
      "descriptive"
    },
    resampling_method = if (any(
      c("p_perm_shannon", "p_perm_spectral") %in% names(entropy_summary)
    )) {
      "permutation"
    } else {
      NA_character_
    },
    source_file = basename(entropy_path),
    source_layer = "entropy_engine"
  )
  
  mp_long <- .to_metric_long(
    df = mp_deltas,
    engine = "mp",
    metric_cols = c(
      "spectral_entropy_delta",
      "participation_ratio_delta",
      "largest_eigenvalue_fraction_delta",
      "excess_spectral_mass_delta"
    ),
    inference_status = "descriptive_only",
    resampling_method = NA_character_,
    source_file = basename(mp_path),
    source_layer = "mp_engine",
    notes = "MP engine currently descriptive only; resampling placeholders are reserved for future architecture."
  )
  
  latent_long <- tibble()
  
  if (isTRUE(include_latent)) {
    latent_metrics <- .safe_read_csv(latent_path, required = latent_required, label = "latent_comparison_metrics")
    
    if (!is.null(latent_metrics)) {
      latent_metrics <- .add_missing_context_cols(latent_metrics)
      
      if (!is.null(chips) && "chip" %in% names(latent_metrics)) {
        latent_metrics <- latent_metrics |>
          dplyr::filter(is.na(.data$chip) | .data$chip %in% chips)
      }
      
      if (!is.null(filter_regimes) &&
          "filter_regime" %in% names(latent_metrics)) {
        latent_metrics <- latent_metrics |>
          dplyr::filter(is.na(.data$filter_regime) |
                          .data$filter_regime %in% filter_regimes)
      }
      
      latent_long <- .to_metric_long(
        df = latent_metrics,
        engine = "latent",
        metric_cols = c(
          "pr_delta",
          "eig_entropy_delta",
          "latent_pr_delta",
          "latent_eig_entropy_delta",
          "centroid_distance"
        ),
        inference_status = "structural_engine",
        resampling_method = NA_character_,
        source_file = basename(latent_path),
        source_layer = "latent_engine",
        notes = "Latent geometry is produced by the Python latent structural engine."
        ) |>
        dplyr::mutate(
          metric = dplyr::recode(
            .data$metric,
            pr_delta = "latent_pr_delta",
            eig_entropy_delta = "latent_eig_entropy_delta"
          )
        )
    } else {
      log_msg(
        paste0(
          "ℹ️ Latent comparison metrics not found at expected candidate paths; ",
          "continuing with MP, complexity, and entropy only. Checked: ",
          paste(latent_candidate_paths, collapse = " | ")
        )
      )
    }
  }
  
  phenotype_long <- dplyr::bind_rows(complexity_long, entropy_long, mp_long, latent_long) |>
    dplyr::arrange(chip, filter_regime, group, comparison, engine, metric)
  
  phenotype_wide <- phenotype_long |>
    dplyr::mutate(metric_key = paste(engine, metric, sep = "__")) |>
    dplyr::select(chip, filter_regime, group, comparison, metric_key, value) |>
    dplyr::distinct() |>
    tidyr::pivot_wider(names_from = metric_key, values_from = value) |>
    dplyr::arrange(chip, filter_regime, group, comparison)
  
  heatmap_long <- phenotype_long |>
    dplyr::group_by(engine, metric) |>
    dplyr::mutate(z_score = .zscore(value)) |>
    dplyr::ungroup() |>
    dplyr::arrange(chip, filter_regime, group, comparison, engine, metric)
  
  phenotype_summary <- phenotype_long |>
    dplyr::group_by(chip, filter_regime, engine, metric, inference_status) |>
    dplyr::summarise(
      n_comparisons = dplyr::n_distinct(comparison),
      n_values = sum(!is.na(value)),
      mean_value = mean(value, na.rm = TRUE),
      median_value = stats::median(value, na.rm = TRUE),
      sd_value = stats::sd(value, na.rm = TRUE),
      n_tumor_greater = sum(direction == "tumor_greater", na.rm = TRUE),
      n_tumor_lower = sum(direction == "tumor_lower", na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(dplyr::across(
      c(mean_value, median_value, sd_value),
      ~ ifelse(is.nan(.x), NA_real_, .x)
    ))
  
  latent_aligned_tables <- .make_latent_aligned_tables(
    phenotype_long = phenotype_long,
    latent_aligned_chip = latent_aligned_chip,
    latent_aligned_filter_regime = latent_aligned_filter_regime
  )
  
  if (nrow(latent_aligned_tables$wide) == 0) {
    log_msg(
      sprintf(
        "ℹ️ Latent-aligned synthesis is empty. Available structural filter regimes are: {%s}. Requested latent alignment: chip=%s, filter_regime=%s.",
        paste(sort(
          unique(phenotype_long$filter_regime[phenotype_long$engine != "latent"])
        ), collapse = ", "),
        latent_aligned_chip,
        latent_aligned_filter_regime
      )
    )
  }
  
  if (isTRUE(write_outputs)) {
    log_msg("💾 Writing structural phenotype synthesis tables...")
    
    readr::write_csv(
      phenotype_long,
      file.path(synthesis_table_dir, "structural_phenotype_long.csv")
    )
    
    readr::write_csv(
      phenotype_wide,
      file.path(synthesis_table_dir, "structural_phenotype_wide.csv")
    )
    
    readr::write_csv(
      heatmap_long,
      file.path(
        synthesis_table_dir,
        "structural_phenotype_heatmap_long.csv"
      )
    )
    
    readr::write_csv(
      phenotype_summary,
      file.path(synthesis_table_dir, "structural_phenotype_summary.csv")
    )
    
    readr::write_csv(
      latent_aligned_tables$long,
      file.path(
        synthesis_table_dir,
        "structural_phenotype_latent_aligned_long.csv"
      )
    )
    
    readr::write_csv(
      latent_aligned_tables$wide,
      file.path(
        synthesis_table_dir,
        "structural_phenotype_latent_aligned_wide.csv"
      )
    )
    
    readr::write_csv(
      latent_aligned_tables$heatmap_long,
      file.path(
        synthesis_table_dir,
        "structural_phenotype_latent_aligned_heatmap_long.csv"
      )
    )
    
    readr::write_csv(
      latent_aligned_tables$summary,
      file.path(
        synthesis_table_dir,
        "structural_phenotype_latent_aligned_summary.csv"
      )
    )
    
    saveRDS(
      list(
        long = phenotype_long,
        wide = phenotype_wide,
        heatmap_long = heatmap_long,
        summary = phenotype_summary,
        latent_aligned = latent_aligned_tables
      ),
      file.path(synthesis_rdata_dir, "structural_phenotype_tables.rds")
    )
  }
  
  log_msg("✅ Structural phenotype table build complete.")
  
  invisible(
    list(
      long = phenotype_long,
      wide = phenotype_wide,
      heatmap_long = heatmap_long,
      summary = phenotype_summary,
      
      table_dir = synthesis_table_dir,
      rdata_dir = synthesis_rdata_dir,
      
      files = list(
        long = file.path(synthesis_table_dir, "structural_phenotype_long.csv"),
        wide = file.path(synthesis_table_dir, "structural_phenotype_wide.csv"),
        heatmap_long = file.path(
          synthesis_table_dir,
          "structural_phenotype_heatmap_long.csv"
        ),
        summary = file.path(synthesis_table_dir, "structural_phenotype_summary.csv"),
        latent_aligned_long = file.path(
          synthesis_table_dir,
          "structural_phenotype_latent_aligned_long.csv"
        ),
        latent_aligned_wide = file.path(
          synthesis_table_dir,
          "structural_phenotype_latent_aligned_wide.csv"
        ),
        latent_aligned_heatmap_long = file.path(
          synthesis_table_dir,
          "structural_phenotype_latent_aligned_heatmap_long.csv"
        ),
        latent_aligned_summary = file.path(
          synthesis_table_dir,
          "structural_phenotype_latent_aligned_summary.csv"
        ),
        rds = file.path(synthesis_rdata_dir, "structural_phenotype_tables.rds")
      )
    )
  )
}
