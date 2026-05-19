# ------------------------------------------------------------------------------
# File: load_biological_annotations.R
# Purpose: Load and validate biological annotation resources
# Pipeline: Biological Interpretation
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Load biological annotation resources
#'
#' Loads the full chip annotation object required by the biological
#' interpretation pipeline. The annotation path is supplied explicitly from the
#' biological interpretation config file.
#'
#' Expected default source:
#' data/global_cancer/processed/RData/annotations/full_chip_annotations.rds
#'
#' @param annotations_path Path to full_chip_annotations.rds.
#' @param logger Pipeline logger object with `$log()` method.
#' @param overwrite Logical. If TRUE, overwrite an existing global `annotations`
#'   object.
#'
#' @return Invisibly returns the loaded annotations object.
#' @export
load_biological_annotations <- function(
    annotations_path,
    logger,
    overwrite = TRUE
) {
  logger$log(
    "📦 Loading biological annotation resources...",
    section = "ANNOTATIONS"
  )
  
  if (missing(annotations_path) || is.null(annotations_path)) {
    stop("❌ annotations_path is missing or NULL.")
  }
  
  if (!file.exists(annotations_path)) {
    stop(sprintf(
      "❌ Annotation file not found: %s",
      annotations_path
    ))
  }
  
  if (exists("annotations", envir = .GlobalEnv) && !isTRUE(overwrite)) {
    logger$log(
      "⏭️ Using existing global annotations object.",
      section = "ANNOTATIONS"
    )
    
    return(invisible(get("annotations", envir = .GlobalEnv)))
  }
  
  annotations <- readRDS(annotations_path)
  
  if (!is.list(annotations)) {
    stop("❌ Loaded annotation object is not a list.")
  }
  
  assign("annotations", annotations, envir = .GlobalEnv)
  
  logger$log(
    sprintf("✅ Loaded biological annotations from: %s", annotations_path),
    section = "ANNOTATIONS"
  )
  
  logger$log(
    sprintf(
      "✅ Annotation chips available: %s",
      paste(names(annotations), collapse = ", ")
    ),
    section = "ANNOTATIONS"
  )
  
  invisible(annotations)
}