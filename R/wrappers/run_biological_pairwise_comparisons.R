# ------------------------------------------------------------------------------
# File: run_biological_pairwise_comparisons.R
# Purpose: Run GO/KEGG/Hallmark biological pairwise comparisons across
#          chip x filtered-probe-regime outputs
# Role: Biological enrichment wrapper
# Pipeline: Biological enrichment
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Run Biological Pairwise Comparisons Across Filter Regimes
#'
#' This wrapper runs GO BP, GO MF, KEGG, and MSigDB/Hallmark biological pairwise
#' comparisons for every chip x filter-regime combination produced by the
#' structural inference pipeline.
#'
#' The biological enrichment layer does not define feature-selection logic.
#' It consumes already-created filtered-probe objects/files from the structural
#' inference layer.
#'
#' This wrapper is responsible for:
#'
#' - consuming concrete filtered-probe regime labels, such as `limma` and
#'   `variance_top3000`;
#' - validating required filtered-probe objects/files;
#' - building comparison input lists from the correct chip/regime-specific
#'   filtered-probe object;
#' - building comparison-specific admissible gene-set inputs once per
#'   chip x filter-regime x gene-set-mode context;
#' - reusing those admissible inputs across selected engines;
#' - writing chip-, regime-, mode-, and engine-aware RDS outputs.
#'
#' Gene-set-level minimum probe thresholds are applied after intersecting:
#'
#' - chip-specific expression probes;
#' - comparison-specific filtered probes;
#' - annotation-derived gene-set probe memberships; and
#' - probes present in both comparison matrices.
#'
#' Expected filtered-probe object names include:
#'
#' - filtered_probes_hu35ksuba_limma
#' - filtered_probes_hu35ksuba_variance_top3000
#' - filtered_probes_hu6800_limma
#' - filtered_probes_hu6800_variance_top3000
#'
#' @param chips Character vector of chip IDs.
#' @param annotations Named list of chip annotation tables.
#' @param engines Character vector. Usually c("complexity", "entropy").
#' @param gene_set_modes Character vector of gene-set modes: GO_BP, GO_MF, KEGG,
#'   MSIGDB, or FULL.
#' @param filter_regimes Character vector of concrete filtered-probe regime
#'   labels, such as c("limma", "variance_top3000").
#' @param min_gene_set_probes Named list or named numeric vector giving the
#'   minimum number of usable probes required per gene-set mode.
#' @param filtered_probes_dir Directory containing filtered-probe RDS files.
#' @param output_dir Directory for RDS result outputs.
#' @param pairwise_logs_dir Optional directory for detailed pairwise gene-set
#'   logging, including skipped gene sets and probe-threshold filtering events.
#' @param biological_resampling_seed Integer seed used by biological
#'   permutation/bootstrap routines.
#' @param run_complexity_permutation Logical; whether to run complexity
#'   permutation testing for each admissible gene set.
#' @param complexity_n_perm Integer number of complexity permutations.
#' @param complexity_permutation_metric Complexity metric used for permutation
#'   testing, usually `"kappa"`.
#' @param complexity_permutation_unit Resampling unit for complexity
#'   permutation testing, usually `"sample_label"`.
#' @param run_complexity_bootstrap Logical; whether to run complexity bootstrap
#'   confidence intervals for each admissible gene set.
#' @param complexity_n_boot Integer number of complexity bootstrap replicates.
#' @param complexity_bootstrap_metric Complexity metric used for bootstrap
#'   intervals, usually `"kappa"`.
#' @param complexity_bootstrap_unit Resampling unit for complexity bootstrap,
#'   usually `"sample"`.
#' @param complexity_covariance_space Covariance orientation for complexity
#'   descriptors, usually `"sample"`.
#' @param run_entropy_permutation Logical; whether to run entropy permutation
#'   testing for each admissible gene set.
#' @param entropy_n_perm Integer number of entropy permutations.
#' @param entropy_permutation_metric Entropy metric used for permutation
#'   testing, usually `"spectral"`.
#' @param entropy_permutation_unit Resampling unit for entropy permutation
#'   testing, usually `"sample_label"`.
#' @param run_entropy_bootstrap Logical; whether to run entropy bootstrap
#'   intervals for each admissible gene set.
#' @param entropy_n_boot Integer number of entropy bootstrap replicates.
#' @param entropy_bootstrap_metric Entropy metric used for bootstrap intervals,
#'   usually `"spectral"`.
#' @param entropy_bootstrap_unit Resampling unit for entropy bootstrap,
#'   usually `"sample"`.
#' @param entropy_covariance_space Covariance orientation for entropy spectral
#'   descriptors, usually `"sample"`.
#' @param logger Optional existing pipeline logger.
#' @param overwrite Logical. If TRUE, overwrite output files and global objects.
#'
#' @return Invisibly returns a data frame manifest of written outputs.
run_biological_pairwise_comparisons <- function(
    chips,
    annotations,
    engines = c("complexity", "entropy"),
    gene_set_modes,
    filter_regimes,
    min_gene_set_probes,
    filtered_probes_dir,
    output_dir,
    pairwise_logs_dir = NULL,
    biological_resampling_seed = 20260510L,
    
    run_complexity_permutation = FALSE,
    complexity_n_perm = 0,
    complexity_permutation_metric = "kappa",
    complexity_permutation_unit = "sample_label",
    
    run_complexity_bootstrap = FALSE,
    complexity_n_boot = 0,
    complexity_bootstrap_metric = "kappa",
    complexity_bootstrap_unit = "sample",
    complexity_covariance_space = "sample",
    
    run_entropy_permutation = FALSE,
    entropy_n_perm = 0,
    entropy_permutation_metric = "spectral",
    entropy_permutation_unit = "sample_label",
    
    run_entropy_bootstrap = FALSE,
    entropy_n_boot = 0,
    entropy_bootstrap_metric = "spectral",
    entropy_bootstrap_unit = "sample",
    entropy_covariance_space = "sample",
    logger = NULL,
    overwrite = TRUE
) {

  # ---- Load dependencies -------------------------------------------------------

  source(here::here("R/helpers/biological_enrichment/run_pairwise_analysis.R"))
  source(here::here("R/helpers/biological_enrichment/prepare_gene_set_engine_inputs.R"))
  source(here::here("R/helpers/biological_enrichment/gene_set_tools.R"))

  source(here::here("R/engines/complexity/core_complexity_metrics.R"))
  source(here::here("R/engines/complexity/statistical_complexity_helpers.R"))
  source(here::here("R/engines/complexity/compute_single_complexity.R"))
  source(here::here("R/engines/complexity/compare_pair_complexity.R"))

  source(here::here("R/engines/entropy/core_entropy_metrics.R"))
  source(here::here("R/engines/entropy/statistical_entropy_helpers.R"))
  source(here::here("R/engines/entropy/compute_single_entropy.R"))
  source(here::here("R/engines/entropy/compare_pair_entropy.R"))

  source(here::here("R/helpers/pipeline_logger.R"))
  source(here::here("R/helpers/build_comparison_input_list.R"))
  source(here::here("R/helpers/structural/resolve_comparison_matrices.R"))

  # ---- Validate inputs ---------------------------------------------------------

  valid_modes <- c("GO_BP", "GO_MF", "KEGG", "MSIGDB", "FULL")
  normalized_modes <- toupper(gene_set_modes)

  invalid_modes <- setdiff(normalized_modes, valid_modes)
  if (length(invalid_modes) > 0) {
    stop(
      sprintf(
        "gene_set_modes must be among %s. Invalid: %s",
        paste(valid_modes, collapse = ", "),
        paste(invalid_modes, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  invalid_engines <- setdiff(engines, c("complexity", "entropy"))
  if (length(invalid_engines) > 0) {
    stop(
      sprintf(
        "engines must contain only 'complexity' and/or 'entropy'. Invalid: %s",
        paste(invalid_engines, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if (!dir.exists(filtered_probes_dir)) {
    stop(
      sprintf("Filtered-probe directory not found: %s", filtered_probes_dir),
      call. = FALSE
    )
  }

  if (missing(output_dir) || is.null(output_dir)) {
    stop("output_dir is required.", call. = FALSE)
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # ---- Logging ----------------------------------------------------------------

  if (is.null(logger)) {
    main_log_dir <- file.path(output_dir, "logs")
    dir.create(main_log_dir, recursive = TRUE, showWarnings = FALSE)

    logger <- start_log(
      logfile = file.path(
        main_log_dir,
        sprintf(
          "biological_pairwise_main_%s.txt",
          format(Sys.time(), "%Y%m%d_%H%M%S")
        )
      )
    )
  }

  if (is.null(pairwise_logs_dir)) {
    pairwise_logs_dir <- file.path(output_dir, "logs", "pairwise_detail")
  }

  dir.create(pairwise_logs_dir, recursive = TRUE, showWarnings = FALSE)

  pairwise_detail_logger <- start_log(
    logfile = file.path(
      pairwise_logs_dir,
      sprintf(
        "biological_pairwise_detail_%s.txt",
        format(Sys.time(), "%Y%m%d_%H%M%S")
      )
    )
  )

  logger$log(
    sprintf(
      "🔀 Running biological pairwise comparisons: modes=%s; engines=%s; filter_regimes=%s",
      paste(normalized_modes, collapse = ", "),
      paste(engines, collapse = ", "),
      paste(filter_regimes, collapse = ", ")
    ),
    section = "BIO_PAIRWISE"
  )

  pairwise_detail_logger$log(
    "🔎 Starting detailed biological pairwise gene-set log.",
    section = "BIO_PAIRWISE_DETAIL"
  )

  # ---- Build gene-set probe indices once per chip -----------------------------

  logger$log(
    "⚙️ Building gene-set probe indices...",
    section = "BIO_PAIRWISE"
  )

  gene_set_indices <- lapply(annotations, build_gene_set_probe_index)

  logger$log(
    "✅ Gene-set probe indices built.",
    section = "BIO_PAIRWISE"
  )

  # ---- Validate chip resources ------------------------------------------------

  missing_annotation_chips <- setdiff(chips, names(annotations))
  if (length(missing_annotation_chips) > 0) {
    stop(
      sprintf(
        "Missing annotations for chip(s): %s",
        paste(missing_annotation_chips, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  required_probe_files <- as.vector(
    outer(
      chips,
      filter_regimes,
      FUN = function(chip, regime) {
        sprintf("filtered_probes_%s_%s.rds", chip, regime)
      }
    )
  )

  missing_probe_files <- required_probe_files[
    !file.exists(file.path(filtered_probes_dir, required_probe_files))
  ]

  if (length(missing_probe_files) > 0) {
    stop(
      sprintf(
        "Missing filtered-probe files required for biological comparisons: %s",
        paste(missing_probe_files, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # ---- Main loop --------------------------------------------------------------

  output_manifest <- data.frame()

  for (gene_set_mode in normalized_modes) {

    mode_min_probes <- min_gene_set_probes[[gene_set_mode]]

    if (is.null(mode_min_probes) || is.na(mode_min_probes)) {
      stop(
        sprintf(
          "No min_gene_set_probes value defined for gene_set_mode: %s",
          gene_set_mode
        ),
        call. = FALSE
      )
    }

    if (gene_set_mode %in% c("GO_BP", "GO_MF")) {
      ontology <- sub("GO_", "", gene_set_mode)
      source_key <- "GO"
    } else {
      ontology <- NULL
      source_key <- gene_set_mode
    }

    mode_suffix <- tolower(gsub("[^a-zA-Z0-9]+", "_", gene_set_mode))

    for (chip in chips) {
      for (filter_regime in filter_regimes) {

        logger$log(
          sprintf(
            "🔄 Preparing admissible inputs: chip=%s; filter_regime=%s; gene_set_mode=%s",
            chip,
            filter_regime,
            gene_set_mode
          ),
          section = "BIO_PAIRWISE"
        )

        pairwise_detail_logger$log(
          sprintf(
            "🔄 Detailed admissibility start: chip=%s; filter_regime=%s; gene_set_mode=%s; min_gene_set_probes=%s",
            chip,
            filter_regime,
            gene_set_mode,
            mode_min_probes
          ),
          section = "BIO_PAIRWISE_DETAIL"
        )

        filtered_name <- sprintf(
          "filtered_probes_%s_%s",
          chip,
          filter_regime
        )

        comparison_name <- sprintf(
          "comparison_map_%s",
          chip
        )

        if (!exists(filtered_name, envir = .GlobalEnv)) {
          filtered_path <- file.path(
            filtered_probes_dir,
            sprintf("%s.rds", filtered_name)
          )

          if (!file.exists(filtered_path)) {
            stop(
              sprintf(
                "Missing filtered-probe object/file for %s: %s",
                filtered_name,
                filtered_path
              ),
              call. = FALSE
            )
          }

          assign(
            filtered_name,
            readRDS(filtered_path),
            envir = .GlobalEnv
          )

          pairwise_detail_logger$log(
            sprintf("📦 Loaded filtered probes: %s", filtered_name),
            section = "BIO_PAIRWISE_DETAIL"
          )
        }

        if (!exists(comparison_name, envir = .GlobalEnv)) {
          stop(
            sprintf("Missing comparison map object: %s", comparison_name),
            call. = FALSE
          )
        }

        filtered_results <- get(filtered_name, envir = .GlobalEnv)
        comparison_map <- get(comparison_name, envir = .GlobalEnv)
        annotation_set <- annotations[[chip]]

        comparison_list <- build_comparison_input_list(
          comparison_map = comparison_map,
          filtered_results = filtered_results,
          chip_id = chip
        )

        gene_sets <- if (identical(source_key, "FULL")) {
          "FULL"
        } else {
          adaptive_gene_set_filter(
            annotation = annotation_set,
            source = source_key,
            min_probes = mode_min_probes,
            ontology = ontology
          )
        }

        admissibility <- logger$timed(
          sprintf(
            "admissibility | %s | %s | %s",
            chip,
            filter_regime,
            gene_set_mode
          ),
          {
            build_admissible_gene_set_inputs(
              comparison_list = comparison_list,
              gene_sets = gene_sets,
              gene_set_indices = gene_set_indices,
              gene_set_mode = gene_set_mode,
              filter_regime = filter_regime,
              min_probes = mode_min_probes,
              logger = pairwise_detail_logger,
              verbose = FALSE
            )
          }
        )

        for (engine in engines) {

          logger$log(
            sprintf(
              "🔬 Starting engine from cached admissible inputs: engine=%s; chip=%s; filter_regime=%s; gene_set_mode=%s",
              engine,
              chip,
              filter_regime,
              gene_set_mode
            ),
            section = "BIO_PAIRWISE"
          )

          result <- logger$timed(
            sprintf(
              "%s | %s | %s | %s",
              engine,
              chip,
              filter_regime,
              gene_set_mode
            ),
            {
              run_pairwise_analysis(
                admissible_inputs = admissibility$inputs,
                skipped = admissibility$skipped,
                engine = engine,
                
                biological_resampling_seed = biological_resampling_seed,
                
                run_complexity_permutation = run_complexity_permutation,
                complexity_n_perm = complexity_n_perm,
                complexity_permutation_metric = complexity_permutation_metric,
                complexity_permutation_unit = complexity_permutation_unit,
                
                run_complexity_bootstrap = run_complexity_bootstrap,
                complexity_n_boot = complexity_n_boot,
                complexity_bootstrap_metric = complexity_bootstrap_metric,
                complexity_bootstrap_unit = complexity_bootstrap_unit,
                complexity_covariance_space = complexity_covariance_space,
                
                run_entropy_permutation = run_entropy_permutation,
                entropy_n_perm = entropy_n_perm,
                entropy_permutation_metric = entropy_permutation_metric,
                entropy_permutation_unit = entropy_permutation_unit,
                
                run_entropy_bootstrap = run_entropy_bootstrap,
                entropy_n_boot = entropy_n_boot,
                entropy_bootstrap_metric = entropy_bootstrap_metric,
                entropy_bootstrap_unit = entropy_bootstrap_unit,
                entropy_covariance_space = entropy_covariance_space,
                
                logger = pairwise_detail_logger,
                verbose = FALSE
              )
            }
          )

          out_name <- sprintf(
            "%s_results_%s_%s_%s",
            engine,
            chip,
            filter_regime,
            mode_suffix
          )

          out_path <- file.path(
            output_dir,
            sprintf("%s.rds", out_name)
          )

          if (file.exists(out_path) && !isTRUE(overwrite)) {
            logger$log(
              sprintf("⏭️ Output exists; not overwriting: %s", basename(out_path)),
              section = "BIO_PAIRWISE"
            )
          } else {
            saveRDS(result, out_path)
          }

          assign(out_name, result, envir = .GlobalEnv)

          manifest_row <- data.frame(
            engine = engine,
            chip = chip,
            filter_regime = filter_regime,
            gene_set_mode = gene_set_mode,
            min_gene_set_probes = mode_min_probes,
            output_object = out_name,
            output_path = out_path,
            stringsAsFactors = FALSE
          )

          output_manifest <- rbind(output_manifest, manifest_row)

          logger$log(
            sprintf("💾 Saved biological pairwise results: %s", basename(out_path)),
            section = "BIO_PAIRWISE"
          )

          pairwise_detail_logger$log(
            sprintf(
              "✅ Completed detailed run: engine=%s; chip=%s; filter_regime=%s; gene_set_mode=%s; output=%s",
              engine,
              chip,
              filter_regime,
              gene_set_mode,
              basename(out_path)
            ),
            section = "BIO_PAIRWISE_DETAIL"
          )
        }
      }
    }
  }

  manifest_path <- file.path(
    output_dir,
    "biological_pairwise_results_manifest.csv"
  )

  utils::write.csv(
    output_manifest,
    manifest_path,
    row.names = FALSE
  )

  logger$log(
    sprintf(
      "✅ Biological pairwise comparisons completed. Manifest: %s",
      manifest_path
    ),
    section = "BIO_PAIRWISE"
  )

  pairwise_detail_logger$log(
    sprintf(
      "✅ Detailed biological pairwise logging completed. Manifest: %s",
      manifest_path
    ),
    section = "BIO_PAIRWISE_DETAIL"
  )

  invisible(output_manifest)
}
