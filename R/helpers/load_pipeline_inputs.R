# ------------------------------------------------------------------------------
# File: load_pipeline_inputs.R
# Purpose: Load and validate reusable pipeline input artifacts
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Load matrix maps from an RData file
#'
#' Loads expression matrix lists and comparison maps into the global environment.
#'
#' Expected object names:
#' - matrices_hu35ksuba
#' - matrices_hu6800
#' - comparison_map_hu35ksuba
#' - comparison_map_hu6800
#'
#' @param matrices_path Path to global_cancer_matrix_maps.RData.
#' @param chips Character vector of chip IDs.
#' @param logger Pipeline logger.
#' @param overwrite Logical. If TRUE, overwrite existing global objects.
#'
#' @return Invisibly returns TRUE.
load_matrix_maps_checked <- function(
    matrices_path,
    chips,
    logger,
    overwrite = TRUE
) {
  if (missing(matrices_path) || is.null(matrices_path)) {
    stop("❌ matrices_path is missing or NULL.")
  }
  
  if (!file.exists(matrices_path)) {
    stop(sprintf("❌ Matrix map file not found: %s", matrices_path))
  }
  
  logger$log("🔀 Loading expression matrices and comparison maps...")
  
  tmp_env <- new.env(parent = emptyenv())
  loaded_objects <- load(matrices_path, envir = tmp_env)
  
  required_objects <- c(
    paste0("matrices_", chips),
    paste0("comparison_map_", chips)
  )
  
  missing_objects <- setdiff(required_objects, loaded_objects)
  
  if (length(missing_objects) > 0) {
    stop(sprintf(
      "❌ Matrix map file is missing required objects: %s",
      paste(missing_objects, collapse = ", ")
    ))
  }
  
  for (obj_name in required_objects) {
    if (exists(obj_name, envir = .GlobalEnv) && !isTRUE(overwrite)) {
      logger$log(sprintf("⏭️ Skipping existing object: %s", obj_name))
      next
    }
    
    assign(
      obj_name,
      get(obj_name, envir = tmp_env),
      envir = .GlobalEnv
    )
    
    logger$log(sprintf("✅ Loaded: %s", obj_name))
  }
  
  logger$log("✅ Matrix maps loaded and validated.")
  invisible(TRUE)
}


#' Load filtered probe results from a directory
#'
#' Loads all filtered-probe RDS files into the global environment using
#' each file name without extension as the object name.
#'
#' Example:
#' filtered_probes_hu35ksuba_limma.rds
#' becomes object:
#' filtered_probes_hu35ksuba_limma
#'
#' @param filtered_probes_dir Directory containing filtered-probe .rds files.
#' @param logger Pipeline logger.
#' @param overwrite Logical. If TRUE, overwrite existing global objects.
#' @param pattern File pattern for RDS files.
#'
#' @return Invisibly returns loaded file paths.
load_filtered_probes_checked <- function(
    filtered_probes_dir,
    logger,
    overwrite = TRUE,
    pattern = "\\.rds$"
) {
  if (missing(filtered_probes_dir) || is.null(filtered_probes_dir)) {
    stop("❌ filtered_probes_dir is missing or NULL.")
  }
  
  if (!dir.exists(filtered_probes_dir)) {
    stop(sprintf(
      "❌ Filtered probes directory not found: %s",
      filtered_probes_dir
    ))
  }
  
  logger$log("🧬 Loading filtered probe results...")
  
  rds_files <- list.files(
    filtered_probes_dir,
    pattern = pattern,
    full.names = TRUE
  )
  
  if (length(rds_files) == 0) {
    stop(sprintf(
      "❌ No filtered-probe RDS files found in: %s",
      filtered_probes_dir
    ))
  }
  
  for (rds_file in rds_files) {
    obj_name <- tools::file_path_sans_ext(basename(rds_file))
    
    if (exists(obj_name, envir = .GlobalEnv) && !isTRUE(overwrite)) {
      logger$log(sprintf("⏭️ Skipping existing object: %s", obj_name))
      next
    }
    
    assign(
      obj_name,
      readRDS(rds_file),
      envir = .GlobalEnv
    )
    
    logger$log(sprintf("✅ Loaded: %s", obj_name))
  }
  
  logger$log(sprintf(
    "✅ Loaded %d filtered-probe-related object(s)",
    length(rds_files)
  ))
  
  invisible(rds_files)
}


#' Validate expected filtered-probe files
#'
#' This checks that expected chip × filter-regime files exist.
#'
#' @param filtered_probes_dir Directory containing filtered-probe .rds files.
#' @param chips Character vector of chip IDs.
#' @param filter_regimes Character vector of filter regime labels.
#' @param logger Pipeline logger.
#'
#' @return Invisibly returns expected file paths.
validate_filtered_probe_files <- function(
    filtered_probes_dir,
    chips,
    filter_regimes,
    logger
) {
  if (missing(filter_regimes) || is.null(filter_regimes)) {
    logger$log("⚠️ filter_regimes not supplied; skipping exact filtered-probe file validation.")
    return(invisible(TRUE))
  }
  
  expected_files <- as.vector(outer(
    chips,
    filter_regimes,
    FUN = function(chip, regime) {
      sprintf("filtered_probes_%s_%s.rds", chip, regime)
    }
  ))
  
  expected_paths <- file.path(filtered_probes_dir, expected_files)
  
  missing_paths <- expected_paths[!file.exists(expected_paths)]
  
  if (length(missing_paths) > 0) {
    stop(sprintf(
      "❌ Missing expected filtered-probe file(s): %s",
      paste(basename(missing_paths), collapse = ", ")
    ))
  }
  
  logger$log("✅ Expected filtered-probe files are present.")
  invisible(expected_paths)
}


#' Load required pipeline inputs
#'
#' Canonical loader for upstream artifacts used by structural inference and
#' biological interpretation pipelines.
#'
#' Config files define paths. Pipeline runners decide which artifacts are required.
#'
#' @param matrices_path Path to matrix/comparison-map RData file.
#' @param filtered_probes_dir Directory containing filtered-probe RDS files.
#' @param chips Character vector of chip IDs.
#' @param logger Pipeline logger.
#' @param filter_regimes Optional character vector of expected filter regimes.
#' @param require_matrix_maps Logical. Load and validate matrix maps.
#' @param require_filtered_probes Logical. Load and validate filtered probes.
#' @param validate_expected_filtered_files Logical. Check expected chip × regime files.
#' @param overwrite Logical. Overwrite global objects if already present.
#'
#' @return Invisibly returns TRUE.
load_pipeline_inputs <- function(
    matrices_path = NULL,
    filtered_probes_dir = NULL,
    chips,
    logger,
    filter_regimes = NULL,
    require_matrix_maps = TRUE,
    require_filtered_probes = FALSE,
    validate_expected_filtered_files = TRUE,
    overwrite = TRUE
) {
  logger$log("📦 Loading pipeline input artifacts...")
  
  if (isTRUE(require_matrix_maps)) {
    load_matrix_maps_checked(
      matrices_path = matrices_path,
      chips = chips,
      logger = logger,
      overwrite = overwrite
    )
  }
  
  if (isTRUE(require_filtered_probes)) {
    if (isTRUE(validate_expected_filtered_files)) {
      validate_filtered_probe_files(
        filtered_probes_dir = filtered_probes_dir,
        chips = chips,
        filter_regimes = filter_regimes,
        logger = logger
      )
    }
    
    load_filtered_probes_checked(
      filtered_probes_dir = filtered_probes_dir,
      logger = logger,
      overwrite = overwrite
    )
  }
  
  logger$log("✅ Pipeline input artifacts loaded and validated.")
  invisible(TRUE)
}