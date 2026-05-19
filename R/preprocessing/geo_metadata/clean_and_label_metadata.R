#' Clean and Label Phenotype Metadata for ExpressionSet List
#'
#' This helper applies known corrections to GEO-style phenotype metadata and derives
#' two additional columns:
#' - `condition`: binary factor indicating "normal" or "cancer"
#' - `tissue_label`: descriptive identifier based on tissue and disease
#'
#' It acts on each `ExpressionSet` in a named list, updating the phenotype data.
#'
#' @param eset_list A named list of \code{ExpressionSet} objects
#' @param tissue_fixes Named vector of known corrections for tissue labels
#' @param disease_fixes Named vector of known corrections for disease labels
#' @param logger Optional pipeline logger (default = NULL). If provided, log messages will be written.
#'
#' @return An updated list of \code{ExpressionSet} objects with cleaned and enriched metadata
clean_and_label_metadata <- function(
    eset_list,
    tissue_fixes = NULL,
    disease_fixes = NULL,
    logger = NULL
) {
  # ---- Dependency Check ----
  if (!requireNamespace("Biobase", quietly = TRUE)) {
    stop("❌ Package 'Biobase' is required but not installed.")
  }
  
  # ---- Input Checks ----
  stopifnot(is.list(eset_list))
  
  if (is.null(tissue_fixes)) {
    tissue_fixes <- character(0)
  }
  
  if (is.null(disease_fixes)) {
    disease_fixes <- character(0)
  }
  
  clean_metadata <- function(eset) {
    stopifnot(inherits(eset, "ExpressionSet"))
    
    pdat <- Biobase::pData(eset)
    
    required_cols <- c("characteristics_ch1", "characteristics_ch1.1")
    missing_cols <- setdiff(required_cols, colnames(pdat))
    
    if (length(missing_cols) > 0) {
      stop(
        glue::glue(
          "❌ Missing required phenotype column(s): {paste(missing_cols, collapse = ', ')}"
        )
      )
    }
    
    # ---- Tissue Label Fixes ----
    for (bad in names(tissue_fixes)) {
      fixed <- tissue_fixes[[bad]]
      mask <- pdat$characteristics_ch1.1 == bad
      
      if (any(mask, na.rm = TRUE)) {
        msg <- glue::glue("🩺 Fixing tissue: '{bad}' → '{fixed}'")
        if (!is.null(logger)) logger$log(msg) else message(msg)
        pdat$characteristics_ch1.1[mask] <- fixed
      }
    }
    
    # ---- Disease Label Fixes ----
    for (bad in names(disease_fixes)) {
      fixed <- disease_fixes[[bad]]
      mask <- pdat$characteristics_ch1 == bad
      
      if (any(mask, na.rm = TRUE)) {
        msg <- glue::glue("🩺 Fixing disease: '{bad}' → '{fixed}'")
        if (!is.null(logger)) logger$log(msg) else message(msg)
        pdat$characteristics_ch1[mask] <- fixed
      }
    }
    
    # ---- Derive Condition and Tissue Label ----
    disease <- tolower(gsub("^disease state: ", "", pdat$characteristics_ch1))
    tissue  <- tolower(gsub("^organism part: ", "", pdat$characteristics_ch1.1))
    
    condition <- ifelse(disease == "normal", "normal", "cancer")
    tissue_label <- ifelse(condition == "normal", tissue, paste(tissue, disease, sep = "_"))
    
    pdat$condition <- factor(condition, levels = c("normal", "cancer"))
    pdat$tissue_label <- make.names(tissue_label)
    
    Biobase::pData(eset) <- pdat
    return(eset)
  }
  
  cleaned_esets <- lapply(eset_list, clean_metadata)
  
  if (!is.null(logger)) {
    logger$log(glue::glue("✅ Cleaned and labeled metadata for {length(cleaned_esets)} ExpressionSet object(s)."))
  }
  
  return(cleaned_esets)
}