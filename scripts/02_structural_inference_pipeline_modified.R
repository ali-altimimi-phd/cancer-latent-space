# ------------------------------------------------------------------------------
# File: 02_structural_inference_pipeline.R
# Purpose: Execute the Global Cancer Structural Inference pipeline
# Role: Top-level structural inference runner
# Pipeline: Structural Inference
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------
NULL

#' Run Structural Inference Pipeline for Global Cancer Microarray Data
#'
#' This script is intended to be run after the Global Cancer preprocessing
#' pipeline has created canonical ExpressionSet and metadata artifacts under:
#'
#'   data/global_cancer/processed/RData/
#'
#' The pipeline treats limma and variance filtering as first-class
#' probe-selection regimes by default. Either regime can be selected by
#' changing `selection_regimes` in `structural_inference_config.R`.
#'
#' Current implemented stages:
#'   1. Configuration and logging
#'   2. Build or load expression matrices and comparison maps
#'   3. Run comparison-aware filtering
#'   4. Marchenko–Pastur spectral engine
#'   5. Complexity engine
#'   6. Entropy engine
#'
#' Later stages should consume the emitted chip/filter-specific filtered-probe
#' products and generate the final structural phenotype master table.
#'
#' @return Invisibly returns a list containing structural filtered probes and
#'         the filter manifest.
run_structural_inference_pipeline <- function() {
  # ---- Stage 1: Configuration + Logging ----------------------------------------
  
  source(here::here("R/config/global_cancer/structural_inference_config.R"),
         local = FALSE)
  
  source(here::here("R/helpers/pipeline_logger.R"))
  
  logger <- start_log(structural_pipeline_logfile)
  
  logger$log("🚀 Starting Global Cancer structural inference pipeline...",
             section = "PIPELINE")
  
  # ---- Local validation and input-preparation helpers --------------------------
  
  validate_structural_config <- function() {
    if (!all(chips %in% valid_chips)) {
      invalid <- setdiff(chips, valid_chips)
      stop(sprintf(
        "❌ Invalid chips value(s): %s",
        paste(invalid, collapse = ", ")
      ))
    }
    
    if (!all(selection_regimes %in% valid_selection_regimes)) {
      invalid <- setdiff(selection_regimes, valid_selection_regimes)
      stop(sprintf(
        "❌ Invalid selection_regimes value(s): %s",
        paste(invalid, collapse = ", ")
      ))
    }
    
    if (!variance_mode %in% valid_variance_modes) {
      stop(sprintf(
        "❌ Invalid variance_mode: %s. Valid values: %s",
        variance_mode,
        paste(valid_variance_modes, collapse = ", ")
      ))
    }
    
    if (isTRUE(build_matrix_maps) && !file.exists(eset_path)) {
      stop(
        glue::glue(
          "❌ Missing ExpressionSets required for matrix-map construction: {eset_path}"
        )
      )
    }
    
    if (!file.exists(load_pipeline_inputs_helper)) {
      stop(
        glue::glue(
          "❌ Missing shared input-loading helper: {load_pipeline_inputs_helper}"
        )
      )
    }
    
    invisible(TRUE)
  }
  
  
  build_local_filter_regime_labels <- function() {
    regime_labels <- ifelse(
      selection_regimes == "variance" & variance_mode == "top_n",
      sprintf("variance_top%d", variance_top_n),
      selection_regimes
    )
    
    regime_labels <- ifelse(
      selection_regimes == "variance" & variance_mode == "threshold",
      sprintf("variance_q%s", gsub(
        "\\.", "", as.character(variance_threshold)
      )),
      regime_labels
    )
    
    regime_labels
  }
  
  sanitize_rds_token <- function(x) {
    x <- as.character(x)
    x <- gsub("[^A-Za-z0-9._-]+", "_", x)
    x <- gsub("_+", "_", x)
    x <- gsub("^_|_$", "", x)
    x
  }
  
  collect_engine_values <- function(x, column_name) {
    values <- character()
    
    if (is.data.frame(x) && column_name %in% names(x)) {
      values <- c(values, as.character(unique(x[[column_name]])))
    }
    
    if (is.list(x)) {
      values <- c(values, unlist(
        lapply(x, collect_engine_values, column_name = column_name),
        use.names = FALSE
      ))
    }
    
    unique(values[!is.na(values) & nzchar(values)])
  }
  
  subset_engine_output <- function(x,
                                   chip_id = NULL,
                                   filter_regime = NULL,
                                   comparison_id = NULL) {
    if (is.data.frame(x)) {
      y <- x
      
      if (!is.null(chip_id) && "chip" %in% names(y)) {
        y <- y[y$chip == chip_id, , drop = FALSE]
      }
      
      regime_col <- intersect(c("filter_regime", "selection_regime", "regime"),
                              names(y))
      
      if (!is.null(filter_regime) && length(regime_col) > 0) {
        y <- y[y[[regime_col[1]]] == filter_regime, , drop = FALSE]
      }
      
      comparison_col <- intersect(c("comparison", "comparison_id"), names(y))
      
      if (!is.null(comparison_id) && length(comparison_col) > 0) {
        y <- y[y[[comparison_col[1]]] == comparison_id, , drop = FALSE]
      }
      
      if (nrow(y) == 0) {
        return(NULL)
      }
      
      return(y)
    }
    
    if (is.list(x)) {
      y <- lapply(
        x,
        subset_engine_output,
        chip_id = chip_id,
        filter_regime = filter_regime,
        comparison_id = comparison_id
      )
      
      keep <- vapply(y, function(z) {
        if (is.null(z)) {
          return(FALSE)
        }
        if (is.data.frame(z)) {
          return(nrow(z) > 0)
        }
        if (is.list(z)) {
          return(length(z) > 0)
        }
        TRUE
      }, logical(1))
      
      y <- y[keep]
      
      if (length(y) == 0) {
        return(NULL)
      }
      
      return(y)
    }
    
    NULL
  }
  
  save_partitioned_engine_rds <- function(engine_results,
                                          engine_name,
                                          structural_rdata_dir,
                                          chips,
                                          filter_regimes,
                                          logger = NULL) {
    engine_dir <- file.path(structural_rdata_dir, engine_name)
    by_regime_dir <- file.path(engine_dir, "by_filter_regime")
    by_comparison_dir <- file.path(engine_dir, "by_comparison")
    
    dir.create(by_regime_dir,
               recursive = TRUE,
               showWarnings = FALSE)
    dir.create(by_comparison_dir,
               recursive = TRUE,
               showWarnings = FALSE)
    
    comparison_ids <- unique(c(
      collect_engine_values(engine_results, "comparison"),
      collect_engine_values(engine_results, "comparison_id")
    ))
    
    written_files <- character()
    
    for (chip_id in chips) {
      for (filter_regime in filter_regimes) {
        regime_subset <- subset_engine_output(engine_results,
                                              chip_id = chip_id,
                                              filter_regime = filter_regime)
        
        if (!is.null(regime_subset)) {
          regime_file <- file.path(
            by_regime_dir,
            sprintf(
              "%s_results_%s_%s.rds",
              engine_name,
              sanitize_rds_token(chip_id),
              sanitize_rds_token(filter_regime)
            )
          )
          
          saveRDS(regime_subset, regime_file)
          written_files <- c(written_files, regime_file)
        }
        
        for (comparison_id in comparison_ids) {
          comparison_subset <- subset_engine_output(
            engine_results,
            chip_id = chip_id,
            filter_regime = filter_regime,
            comparison_id = comparison_id
          )
          
          if (is.null(comparison_subset)) {
            next
          }
          
          comparison_file <- file.path(
            by_comparison_dir,
            sprintf(
              "%s_results_%s_%s_%s.rds",
              engine_name,
              sanitize_rds_token(chip_id),
              sanitize_rds_token(filter_regime),
              sanitize_rds_token(comparison_id)
            )
          )
          
          saveRDS(comparison_subset, comparison_file)
          written_files <- c(written_files, comparison_file)
        }
      }
    }
    
    if (!is.null(logger)) {
      logger$log(
        sprintf(
          "💾 Wrote %d partitioned %s RDS file(s).",
          length(written_files),
          engine_name
        ),
        section = toupper(engine_name)
      )
    }
    
    invisible(written_files)
  }
  
  
  rebuild_local_matrix_objects <- function() {
    matrix_lists <- list()
    comparison_maps <- list()
    
    if (exists("matrices_hu35ksuba", inherits = TRUE)) {
      matrix_lists[["hu35ksuba"]] <- get("matrices_hu35ksuba", inherits = TRUE)
    }
    
    if (exists("matrices_hu6800", inherits = TRUE)) {
      matrix_lists[["hu6800"]] <- get("matrices_hu6800", inherits = TRUE)
    }
    
    if (exists("comparison_map_hu35ksuba", inherits = TRUE)) {
      comparison_maps[["hu35ksuba"]] <- get("comparison_map_hu35ksuba", inherits = TRUE)
    }
    
    if (exists("comparison_map_hu6800", inherits = TRUE)) {
      comparison_maps[["hu6800"]] <- get("comparison_map_hu6800", inherits = TRUE)
    }
    
    missing_matrix_chips <- setdiff(chips, names(matrix_lists))
    missing_map_chips <- setdiff(chips, names(comparison_maps))
    
    if (length(missing_matrix_chips) > 0) {
      stop(sprintf(
        "❌ Missing matrix lists for chip(s): %s",
        paste(missing_matrix_chips, collapse = ", ")
      ))
    }
    
    if (length(missing_map_chips) > 0) {
      stop(sprintf(
        "❌ Missing comparison maps for chip(s): %s",
        paste(missing_map_chips, collapse = ", ")
      ))
    }
    
    list(matrix_lists = matrix_lists, comparison_maps = comparison_maps)
  }
  
  prepare_structural_inputs <- function(require_filtered_probes = FALSE) {
    probe_file_regimes <- ifelse(
      selection_regimes == "variance" & variance_mode == "top_n",
      sprintf("variance_top%d", variance_top_n),
      selection_regimes
    )
    
    probe_file_regimes <- ifelse(
      selection_regimes == "variance" & variance_mode == "threshold",
      sprintf("variance_q%s", gsub(
        "\\.", "", as.character(variance_threshold)
      )),
      probe_file_regimes
    )
    
    load_pipeline_inputs(
      matrices_path = matrices_path,
      filtered_probes_dir = structural_filtered_probes_dir,
      chips = chips,
      filter_regimes = probe_file_regimes,
      require_matrix_maps = TRUE,
      require_filtered_probes = require_filtered_probes,
      logger = logger,
      overwrite = TRUE
    )
    
    rebuilt <- rebuild_local_matrix_objects()
    
    list(
      matrix_lists = rebuilt$matrix_lists,
      comparison_maps = rebuilt$comparison_maps
    )
  }
  
  validate_structural_config()
  
  source(load_pipeline_inputs_helper)
  
  logger$log(sprintf("🧭 Chips: %s", paste(chips, collapse = ", ")), section = "CONFIG")
  
  logger$log(sprintf(
    "🧪 Probe selection regimes: %s",
    paste(selection_regimes, collapse = ", ")
  ), section = "CONFIG")
  
  logger$log(sprintf("📂 Matrix maps path: %s", matrices_path), section = "CONFIG")
  
  logger$log(
    sprintf(
      "📂 Filtered probes directory: %s",
      structural_filtered_probes_dir
    ),
    section = "CONFIG"
  )
  
  logger$log(sprintf(
    "🧬 Filtered-probe file regimes: %s",
    paste(build_local_filter_regime_labels(), collapse = ", ")
  ),
  section = "CONFIG")
  
  # ---- Stage 2: Build or load matrix + comparison maps ----
  if (isTRUE(build_matrix_maps)) {
    logger$log("⚙️ Building expression matrices and comparison maps...")
    
    if (exists("eset_list", inherits = TRUE)) {
      logger$log("⏭️ Using existing 'eset_list' from memory.")
    } else {
      loaded_objs <- load(eset_path)
      if (!"eset_list" %in% loaded_objs) {
        stop("❌ 'eset_list' not found in loaded RData file.")
      }
      eset_list <- get("eset_list")
      logger$log("✅ ExpressionSets loaded from disk.")
    }
    
    source(here::here("R/helpers/matrix_builders.R"), local = TRUE)
    
    matrix_lists <- list()
    comparison_maps <- list()
    
    for (chip_id in chips) {
      if (!chip_id %in% names(eset_list)) {
        stop(sprintf("❌ Chip '%s' not found in eset_list.", chip_id))
      }
      
      logger$log(sprintf("⚙️ Building matrix/comparison map for chip: %s", chip_id))
      matrix_lists[[chip_id]] <- build_matrix_lists_by_tissue(eset_list[[chip_id]])
      comparison_maps[[chip_id]] <- define_predefined_comparisons(names(matrix_lists[[chip_id]]))
    }
    
    # Preserve legacy object names for compatibility with existing helpers.
    if ("hu35ksuba" %in% names(matrix_lists)) {
      matrices_hu35ksuba <- matrix_lists[["hu35ksuba"]]
      comparison_map_hu35ksuba <- comparison_maps[["hu35ksuba"]]
    }
    if ("hu6800" %in% names(matrix_lists)) {
      matrices_hu6800 <- matrix_lists[["hu6800"]]
      comparison_map_hu6800 <- comparison_maps[["hu6800"]]
    }
    
    dir.create(dirname(matrices_path),
               recursive = TRUE,
               showWarnings = FALSE)
    
    save(
      matrices_hu35ksuba,
      matrices_hu6800,
      comparison_map_hu35ksuba,
      comparison_map_hu6800,
      file = matrices_path
    )
    
    logger$log("💾 Matrix and comparison maps saved.")
  } else {
    logger$log("🔀 Loading expression matrices and comparison maps...",
               section = "INPUTS")
  }
  
  prepared <- prepare_structural_inputs(require_filtered_probes = FALSE)
  
  matrix_lists <- prepared$matrix_lists
  comparison_maps <- prepared$comparison_maps
  
  
  # ---- Stage 3: Run probe-selection regimes ----
  
  structural_filtered_results <- list()
  structural_filter_manifest <- data.frame()
  
  if (isTRUE(run_probe_selection)) {
    logger$log(
      "🔀 Checking expression matrices and comparison maps before probe selection...",
      section = "INPUTS"
    )
    
    prepared <- prepare_structural_inputs(require_filtered_probes = FALSE)
    
    matrix_lists <- prepared$matrix_lists
    comparison_maps <- prepared$comparison_maps
    
    logger$log("✅ Expression matrices and comparison maps available for probe selection.",
               section = "INPUTS")
    
    logger$log("🧬 Running structural probe selection by canonical filter regime...",
               section = "FILTERING")
    
    source(here::here("R/wrappers/run_grouped_probe_filtering.R"))
    source(here::here("R/filters/select_high_variance_probes.R"))
    
    required_filter_cols <- c("filter_regime",
                              "filter_method",
                              "filter_scope",
                              "filter_n",
                              "variance_mode")
    
    missing_filter_cols <- setdiff(required_filter_cols, names(filter_regime_tbl))
    
    if (length(missing_filter_cols) > 0) {
      stop(
        "filter_regime_tbl is missing required columns: ",
        paste(missing_filter_cols, collapse = ", "),
        call. = FALSE
      )
    }
    
    for (chip_id in chips) {
      structural_filtered_results[[chip_id]] <- list()
      
      for (regime_i in seq_len(nrow(filter_regime_tbl))) {
        regime_row <- filter_regime_tbl[regime_i, , drop = FALSE]
        
        filter_regime_i <- regime_row$filter_regime
        filter_method_i <- regime_row$filter_method
        filter_scope_i  <- regime_row$filter_scope
        filter_n_i      <- regime_row$filter_n
        variance_mode_i <- regime_row$variance_mode
        
        logfc_cutoff_i <- if (identical(filter_method_i, "limma")) {
          limma_logfc_cutoff
        } else {
          NULL
        }
        
        pval_cutoff_i <- if (identical(filter_method_i, "limma")) {
          limma_pval_cutoff
        } else {
          NULL
        }
        
        top_n_i <- if (identical(filter_method_i, "variance") &&
                       identical(variance_mode_i, "top_n")) {
          as.integer(filter_n_i)
        } else {
          NULL
        }
        
        var_threshold_i <- if (identical(filter_method_i, "variance") &&
                               identical(variance_mode_i, "threshold")) {
          variance_threshold
        } else {
          NULL
        }
        
        save_path_i <- here::here(
          structural_filtered_probes_dir,
          sprintf(
            "filtered_probes_%s_%s.rds",
            chip_id,
            filter_regime_i
          )
        )
        
        logger$log(
          sprintf(
            paste(
              "🧬 Running probe-selection regime:",
              "chip=%s; regime=%s; method=%s; scope=%s"
            ),
            chip_id,
            filter_regime_i,
            filter_method_i,
            filter_scope_i
          ),
          section = "FILTERING"
        )
        
        result_i <- run_grouped_probe_filtering(
          matrix_list              = matrix_lists[[chip_id]],
          comparison_map           = comparison_maps[[chip_id]],
          chip_id                  = chip_id,
          method                   = filter_method_i,
          logfc_cutoff             = logfc_cutoff_i,
          pval_cutoff              = pval_cutoff_i,
          var_threshold            = var_threshold_i,
          top_n                    = top_n_i,
          variance_selection_mode  = variance_mode_i,
          save_path                = save_path_i
        )
        
        result_i$`__metadata__`$filter_regime <- filter_regime_i
        result_i$`__metadata__`$filter_method <- filter_method_i
        result_i$`__metadata__`$filter_scope  <- filter_scope_i
        result_i$`__metadata__`$filter_n      <- filter_n_i
        
        structural_filtered_results[[chip_id]][[filter_regime_i]] <- result_i
        
        manifest_row <- data.frame(
          study_name = study_name,
          chip = chip_id,
          filter_regime = filter_regime_i,
          filter_method = filter_method_i,
          filter_scope = filter_scope_i,
          variance_mode = ifelse(
            identical(filter_method_i, "variance"),
            variance_mode_i,
            NA_character_
          ),
          variance_top_n = ifelse(
            identical(filter_method_i, "variance") &&
              identical(variance_mode_i, "top_n"),
            filter_n_i,
            NA_real_
          ),
          variance_threshold = ifelse(
            identical(filter_method_i, "variance") &&
              identical(variance_mode_i, "threshold"),
            variance_threshold,
            NA_real_
          ),
          limma_logfc_cutoff = ifelse(
            identical(filter_method_i, "limma"),
            limma_logfc_cutoff,
            NA_real_
          ),
          limma_pval_cutoff = ifelse(
            identical(filter_method_i, "limma"),
            limma_pval_cutoff,
            NA_real_
          ),
          filtered_probe_path = save_path_i,
          stringsAsFactors = FALSE
        )
        
        structural_filter_manifest <- rbind(structural_filter_manifest, manifest_row)
        
        logger$log(
          sprintf(
            "✅ Completed probe-selection regime chip=%s; regime=%s",
            chip_id,
            filter_regime_i
          ),
          section = "FILTERING"
        )
      }
    }
    
    saveRDS(
      structural_filtered_results,
      here::here(
        structural_filtered_probes_dir,
        "structural_filtered_results_all.rds"
      )
    )
    
    utils::write.csv(structural_filter_manifest,
                     structural_filter_manifest_path,
                     row.names = FALSE)
    
    logger$log(
      sprintf(
        "💾 Probe-selection manifest saved: %s",
        structural_filter_manifest_path
      ),
      section = "FILTERING"
    )
    
  } else {
    logger$log("⏭️ Skipping probe selection because run_probe_selection is FALSE.",
               section = "FILTERING")
  }
  
  
  # ==============================================================================
  # Stage 4: Structural descriptor engines
  # ==============================================================================
  
  if (isTRUE(run_mp_engine) ||
      isTRUE(run_complexity_engine) || isTRUE(run_entropy_engine)) {
    logger$timed("Structural descriptor engines", {
      logger$log("🧬 Running structural descriptor engines...", section = "STRUCTURAL_ENGINES")
      
      logger$log("📦 Ensuring required structural inputs are available...",
                 section = "INPUTS")
      
      prepared <- prepare_structural_inputs(require_filtered_probes = TRUE)
      
      matrix_lists <- prepared$matrix_lists
      comparison_maps <- prepared$comparison_maps
      
      logger$log("✅ Required structural inputs are available.", section = "INPUTS")
      
      # ---- Validate canonical filter-regime table -------------------------------
      
      if (!exists("filter_regime_tbl")) {
        stop(
          "Missing required object: filter_regime_tbl. ",
          "Define it in structural_inference_config.R.",
          call. = FALSE
        )
      }
      
      required_filter_cols <- c(
        "filter_regime",
        "filter_method",
        "filter_scope",
        "filter_n",
        "variance_mode"
      )
      
      missing_filter_cols <- setdiff(required_filter_cols, names(filter_regime_tbl))
      
      if (length(missing_filter_cols) > 0) {
        stop(
          "filter_regime_tbl is missing required columns: ",
          paste(missing_filter_cols, collapse = ", "),
          call. = FALSE
        )
      }
      
      invalid_filter_regimes <- setdiff(unique(filter_regime_tbl$filter_regime),
                                        valid_filter_regimes)
      
      if (length(invalid_filter_regimes) > 0) {
        stop(
          "Invalid filter_regime values in filter_regime_tbl: ",
          paste(invalid_filter_regimes, collapse = ", "),
          call. = FALSE
        )
      }
      
      logger$log(sprintf(
        "Using filter regimes: %s",
        paste(filter_regime_tbl$filter_regime, collapse = ", ")
      ), section = "STRUCTURAL_ENGINES")
      
      # ---- Shared helpers -------------------------------------------------------
      
      source(here::here("R/helpers/structural/resolve_comparison_matrices.R"))
      source(here::here(
        "R/helpers/structural/build_selection_regime_labels.R"
      ))
      
      # ---- MP spectral engine ---------------------------------------------------
      
      if (isTRUE(run_mp_engine)) {
        logger$timed("Marchenko-Pastur spectral engine", {
          logger$log("🌊 Running Marchenko–Pastur spectral organization analysis...",
                     section = "MP")
          
          source(here::here(
            "R/engines/spectral/statistical_spectral_helpers.R"
          ))
          source(here::here("R/engines/spectral/core_spectral_metrics.R"))
          source(here::here("R/engines/spectral/marchenko_pastur_helpers.R"))
          source(here::here("R/engines/spectral/compute_single_spectral.R"))
          source(here::here("R/engines/spectral/compare_pair_spectral.R"))
          source(here::here("R/engines/spectral/compute_spectral_deltas.R"))
          source(here::here("R/engines/spectral/run_pairwise_spectral.R"))
          source(here::here("R/wrappers/run_mp_structural_engine.R"))
          
          mp_results <- run_mp_structural_engine(
            matrix_lookup = matrix_lists,
            comparison_lookup = comparison_maps,
            filtered_probes_dir = structural_filtered_probes_dir,
            chips = chips,
            filter_regime_tbl = filter_regime_tbl,
            min_samples_per_condition = min_samples_per_condition,
            min_selected_probes = min_selected_probes
          )
          
          saveRDS(mp_results,
                  file.path(structural_rdata_dir, "mp_spectral_results.rds"))
          
          save_partitioned_engine_rds(
            engine_results = mp_results,
            engine_name = "mp_spectral",
            structural_rdata_dir = structural_rdata_dir,
            chips = chips,
            filter_regimes = filter_regime_tbl$filter_regime,
            logger = logger
          )
          
          logger$log("✅ Marchenko–Pastur spectral engine completed successfully.",
                     section = "MP")
        })
        
      } else {
        logger$log("⏭️ Marchenko–Pastur spectral engine skipped.",
                   section = "MP")
      }
      
      # ---- Complexity engine ----------------------------------------------------
      
      if (isTRUE(run_complexity_engine)) {
        logger$timed("Complexity structural engine", {
          logger$log("🧮 Running complexity structural analysis...",
                     section = "COMPLEXITY")
          
          source(here::here("R/engines/complexity/core_complexity_metrics.R"))
          source(here::here(
            "R/engines/complexity/statistical_complexity_helpers.R"
          ))
          source(here::here(
            "R/engines/complexity/compute_single_complexity.R"
          ))
          source(here::here("R/engines/complexity/compare_pair_complexity.R"))
          source(here::here("R/engines/complexity/run_pairwise_complexity.R"))
          source(here::here("R/wrappers/run_complexity_structural_engine.R"))
          
          complexity_results <- run_complexity_structural_engine(
            matrix_lookup = matrix_lists,
            comparison_lookup = comparison_maps,
            filtered_probes_dir = structural_filtered_probes_dir,
            chips = chips,
            filter_regime_tbl = filter_regime_tbl,
            
            run_permutation = run_complexity_permutation,
            n_perm = complexity_n_perm,
            permutation_metric = complexity_permutation_metric,
            permutation_unit = complexity_permutation_unit,
            
            run_bootstrap = run_complexity_bootstrap,
            n_boot = complexity_n_boot,
            bootstrap_metric = complexity_bootstrap_metric,
            bootstrap_unit = complexity_bootstrap_unit,
            
            covariance_space = complexity_covariance_space,
            seed = structural_resampling_seed
          )
          
          saveRDS(
            complexity_results,
            file.path(structural_rdata_dir, "complexity_results.rds")
          )
          
          save_partitioned_engine_rds(
            engine_results = complexity_results,
            engine_name = "complexity",
            structural_rdata_dir = structural_rdata_dir,
            chips = chips,
            filter_regimes = filter_regime_tbl$filter_regime,
            logger = logger
          )
          
          logger$log("✅ Complexity structural engine completed successfully.",
                     section = "COMPLEXITY")
        })
        
      } else {
        logger$log("⏭️ Complexity structural engine skipped.", section = "COMPLEXITY")
      }
      
      # ---- Entropy engine -------------------------------------------------------
      
      if (isTRUE(run_entropy_engine)) {
        logger$timed("Entropy structural engine", {
          logger$log("🧮 Running entropy structural analysis...", section = "ENTROPY")
          
          source(here::here("R/engines/entropy/core_entropy_metrics.R"))
          source(here::here(
            "R/engines/entropy/statistical_entropy_helpers.R"
          ))
          source(here::here("R/engines/entropy/compute_single_entropy.R"))
          source(here::here("R/engines/entropy/compare_pair_entropy.R"))
          source(here::here("R/engines/entropy/run_pairwise_entropy.R"))
          source(here::here("R/wrappers/run_entropy_structural_engine.R"))
          
          logger$log(
            sprintf(
              paste(
                "Entropy inference settings:",
                "permutation=%s n_perm=%d metric=%s;",
                "bootstrap=%s n_boot=%d metric=%s;",
                "covariance_space=%s"
              ),
              run_entropy_permutation,
              entropy_n_perm,
              entropy_permutation_metric,
              run_entropy_bootstrap,
              entropy_n_boot,
              entropy_bootstrap_metric,
              entropy_covariance_space
            ),
            section = "ENTROPY"
          )
          
          entropy_results <- run_entropy_structural_engine(
            matrix_lookup = matrix_lists,
            comparison_lookup = comparison_maps,
            filtered_probes_dir = structural_filtered_probes_dir,
            chips = chips,
            filter_regime_tbl = filter_regime_tbl,
            
            run_permutation = run_entropy_permutation,
            n_perm = entropy_n_perm,
            run_bootstrap = run_entropy_bootstrap,
            n_boot = entropy_n_boot,
            
            permutation_metric = entropy_permutation_metric,
            bootstrap_metric = entropy_bootstrap_metric,
            covariance_space = entropy_covariance_space,
            
            permutation_unit = entropy_permutation_unit,
            bootstrap_unit = entropy_bootstrap_unit,
            
            seed = structural_resampling_seed
          )
          
          saveRDS(
            entropy_results,
            file.path(structural_rdata_dir, "entropy_results.rds")
          )
          
          save_partitioned_engine_rds(
            engine_results = entropy_results,
            engine_name = "entropy",
            structural_rdata_dir = structural_rdata_dir,
            chips = chips,
            filter_regimes = filter_regime_tbl$filter_regime,
            logger = logger
          )
          
          logger$log("✅ Entropy structural engine completed successfully.",
                     section = "ENTROPY")
        })
        
      } else {
        logger$log("⏭️ Entropy structural engine skipped.", section = "ENTROPY")
      }
      
      logger$log("✅ Structural descriptor engine stage completed.",
                 section = "STRUCTURAL_ENGINES")
    })
    
  } else {
    logger$log("⏭️ All structural descriptor engines skipped.", section = "STRUCTURAL_ENGINES")
  }
  
  
  # ---- Stage 5: Latent-space engine --------------------------------------------
  
  # The latent engine is orchestrated by R and executed by a Python backend.
  #
  # R owns:
  #   - configuration
  #   - provenance
  #   - logging
  #   - pipeline ordering
  #
  # Python owns:
  #   - preprocessing
  #   - VAE training
  #   - latent table generation
  #   - latent metric computation
  
  if (isTRUE(run_latent_engine)) {
    logger$log("🐍 Running latent-space structural engine...", section = "LATENT")
    
    source(here::here("R", "engines", "latent", "run_latent_engine.R"))
    
    run_latent_python_scripts_stage(logger = logger)
    
    logger$log("✅ Latent-space structural engine completed successfully.",
               section = "LATENT")
    
  } else {
    logger$log("⏭️ Latent-space structural engine skipped.", section = "LATENT")
  }
  
  
  # ---- Stage 6: Synthesis ------------------------------------------------------
  
  structural_synthesis_results <- NULL
  structural_plot_results <- NULL
  
  if (isTRUE(build_structural_phenotype_table)) {
    logger$log("🧬 Building unified structural phenotype tables...",
               section = "SYNTHESIS")
    
    source(
      here::here(
        "R",
        "structural_synthesis",
        "build_structural_phenotype_table.R"
      )
    )
    
    structural_synthesis_results <- build_structural_phenotype_table(
      study_name = study_name,
      structural_rdata_dir = structural_rdata_dir,
      latent_table_dir = latent_tables_dir,
      synthesis_table_dir = structural_synthesis_table_dir,
      synthesis_rdata_dir = structural_synthesis_rdata_dir,
      chips = chips,
      filter_regimes = selection_regimes,
      include_latent = TRUE,
      logger = logger
    )
    
    logger$log(
      paste0(
        "✅ Structural phenotype CSV tables written to: ",
        structural_synthesis_results$table_dir
      ),
      section = "SYNTHESIS"
    )
    
    logger$log(
      paste0(
        "✅ Structural phenotype RDS written to: ",
        structural_synthesis_results$files$rds
      ),
      section = "SYNTHESIS"
    )
  }
  
  if (isTRUE(build_structural_synthesis_plots)) {
    logger$log("📊 Building structural synthesis diagnostic plots...",
               section = "SYNTHESIS")
    
    source(here::here(
      "R",
      "visualizations",
      "plot_structural_synthesis_diagnostics.R"
    ))
    
    structural_plot_results <- plot_structural_synthesis_diagnostics(
      study_name = study_name,
      synthesis_table_dir = structural_synthesis_table_dir,
      synthesis_plot_dir = structural_synthesis_plot_dir,
      chips = chips,
      filter_regimes = selection_regimes,
      average_across_chips = TRUE,
      save_pdf = TRUE,
      logger = logger
    )
    
    logger$log(
      paste0(
        "✅ Structural synthesis diagnostic plots written to: ",
        structural_plot_results$plot_dir
      ),
      section = "SYNTHESIS"
    )
  }
  
  # ---- Stage 7: DuckDB warehouse ingestion ----------------------------------
  
  structural_duckdb_ingestion_results <- NULL
  
  if (isTRUE(ingest_structural_results_to_duckdb)) {
    logger$timed("DuckDB structural results ingestion", {
      logger$log("🦆 Ingesting structural inference results into DuckDB...",
                 section = "DUCKDB")
      
      source(
        here::here(
          "R",
          "warehouse",
          "ingest",
          "ingest_structural_results_to_duckdb.R"
        )
      )
      
      ingest_structural_results_to_duckdb(
        study_name = study_name,
        structural_output_dir = structural_output_dir,
        warehouse_dir = warehouse_dir,
        db_path = warehouse_db_path,
        write_csv_exports = TRUE,
        logger = logger
      )
      
      logger$log("✅ Structural inference DuckDB ingestion completed.",
                 section = "DUCKDB")
    })
    
  } else {
    logger$log("⏭️ Structural DuckDB ingestion skipped.", section = "DUCKDB")
  }
  
  # ---- Stage 8: Post-run validation -----------------------------------------
  
  # ------------------------------------------------------------------------------
  # Validation stage notes
  # ------------------------------------------------------------------------------
  #
  # The structural pipeline validator operates in a stage-aware manner.
  #
  # Validation checks are only performed for stages that are enabled in the
  # current pipeline run.
  #
  # Examples:
  #
  #   run_latent_engine = FALSE
  #
  # disables validation of:
  #
  #   - latent_comparison_metrics.csv
  #   - latent_sample_coordinates.csv
  #   - vae_training_summary.csv
  #   - latent DuckDB fact tables
  #
  # Likewise:
  #
  #   ingest_structural_results_to_duckdb = FALSE
  #
  # skips DuckDB warehouse validation entirely.
  #
  # For a complete post-run validation of the integrated structural framework,
  # the recommended configuration is:
  #
  #   run_latent_engine <- TRUE
  #   ingest_structural_results_to_duckdb <- TRUE
  #   validate_structural_pipeline_outputs <- TRUE
  #
  # while optionally leaving expensive computational stages disabled:
  #
  #   run_mp_engine         <- FALSE
  #   run_complexity_engine <- FALSE
  #   run_entropy_engine    <- FALSE
  #
  # This permits fast integrity validation of existing outputs without
  # recomputing structural descriptors.
  # ------------------------------------------------------------------------------
  
  structural_validation_results <- NULL
  
  if (isTRUE(validate_structural_pipeline_outputs)) {
    logger$timed("Structural pipeline output validation", {
      logger$log("🔎 Validating structural pipeline outputs...", section = "VALIDATION")
      
      source(here::here(
        "R",
        "validation",
        "validate_structural_pipeline_run.R"
      ))
      
      validate_structural_pipeline_run(
        structural_rdata_dir = structural_rdata_dir,
        structural_tables_dir = structural_tables_dir,
        structural_synthesis_table_dir = structural_synthesis_table_dir,
        warehouse_db_path = warehouse_db_path,
        
        run_mp_engine = run_mp_engine,
        run_complexity_engine = run_complexity_engine,
        run_entropy_engine = run_entropy_engine,
        run_latent_engine = run_latent_engine,
        
        build_structural_phenotype_table = build_structural_phenotype_table,
        ingest_structural_results_to_duckdb = ingest_structural_results_to_duckdb,
        
        latent_tables_dir = latent_tables_dir,
        latent_chip_id = latent_chip_id,
        latent_filter_regime = latent_filter_regime,
        
        logger = logger
      )
      
      logger$log(
        paste0(
          "✅ Structural validation status: ",
          structural_validation_results$status
        ),
        section = "VALIDATION"
      )
    })
    
  } else {
    logger$log("⏭️ Structural pipeline output validation skipped.",
               section = "VALIDATION")
  }
  
  
  # ---- Stage 9: End ----------------------------------------------------------
  
  logger$log("🏁 Structural inference pipeline completed.")
  
  invisible(
    list(
      structural_filtered_results          = structural_filtered_results,
      structural_filter_manifest           = structural_filter_manifest,
      structural_synthesis_results         = structural_synthesis_results,
      structural_plot_results              = structural_plot_results,
      structural_duckdb_ingestion_results  = structural_duckdb_ingestion_results,
      structural_validation_results        = structural_validation_results
    )
  )
}


if (sys.nframe() == 0) {
  run_structural_inference_pipeline()
}
