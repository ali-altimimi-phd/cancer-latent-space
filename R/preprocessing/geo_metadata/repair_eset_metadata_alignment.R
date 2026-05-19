# ------------------------------------------------------------------------------
# File: repair_eset_metadata_alignment.R
# Purpose: Synchronize ExpressionSet metadata row names with expression columns
# Role: Metadata alignment helper
# Pipeline: Preprocessing
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Repair ExpressionSet Metadata Alignment
#'
#' @description
#' Ensures that each ExpressionSet has metadata row names identical to expression
#' matrix column names. GEO accessions are retained as GSM identifiers.
#'
#' @param eset_list Named list of ExpressionSet objects.
#' @param logger Optional pipeline logger.
#'
#' @return A named list of ExpressionSet objects with synchronized pData rownames.
#' @export
repair_eset_metadata_alignment <- function(eset_list, logger = NULL) {
  log_it <- function(msg, section = "METADATA") {
    if (!is.null(logger)) {
      logger$log(msg, section = section)
    } else {
      message(msg)
    }
  }
  
  if (!requireNamespace("Biobase", quietly = TRUE)) {
    stop("Package 'Biobase' is required for ExpressionSet alignment.", call. = FALSE)
  }
  
  for (chip in names(eset_list)) {
    es <- eset_list[[chip]]
    
    expr_names <- colnames(Biobase::exprs(es))
    expr_gsm   <- sub("_.*$", "", expr_names)
    pdat       <- Biobase::pData(es)
    
    if (!"geo_accession" %in% colnames(pdat)) {
      stop(sprintf("Missing 'geo_accession' column for chip: %s", chip), call. = FALSE)
    }
    
    match_idx <- match(expr_gsm, pdat$geo_accession)
    
    if (any(is.na(match_idx))) {
      missing_ids <- expr_names[is.na(match_idx)]
      stop(sprintf(
        "Metadata mismatch for %s: %d samples missing. First: %s",
        chip, length(missing_ids), missing_ids[[1]]
      ), call. = FALSE)
    }
    
    pdat <- pdat[match_idx, , drop = FALSE]
    
    if (!identical(expr_gsm, pdat$geo_accession)) {
      stop(sprintf("GSM alignment check failed for chip: %s", chip), call. = FALSE)
    }
    
    rownames(pdat) <- expr_names
    pdat$sample_id <- expr_names
    pdat$gsm_id    <- pdat$geo_accession
    
    Biobase::pData(es) <- pdat
    
    if (!identical(colnames(Biobase::exprs(es)), rownames(Biobase::pData(es)))) {
      stop(sprintf("Final ExpressionSet alignment failed for chip: %s", chip), call. = FALSE)
    }
    
    eset_list[[chip]] <- es
    
    log_it(sprintf(
      "🔗 Metadata aligned for %s: %d samples.",
      chip, length(expr_names)
    ))
  }
  
  eset_list
}