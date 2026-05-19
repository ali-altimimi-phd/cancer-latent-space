#' Attach GEO Metadata to ExpressionSets
#'
#' Downloads and attaches GEO metadata from GSE68928 to matching chip-specific
#' ExpressionSet objects. Modifies `pData` of each ExpressionSet if metadata matches.
#'
#' @param eset_list A named list of `ExpressionSet` objects.
#' @param geo_dir Directory to cache GEO files.
#' @param logger Logging object (optional).
#'
#' @return Modified `eset_list` with updated `pData` (if matches found).
attach_geo_metadata <- function(eset_list, geo_dir, logger = NULL) {
  # ---- Dependency Check ----
  if (!requireNamespace("GEOquery", quietly = TRUE)) {
    stop("❌ Package 'GEOquery' is required but not installed.")
  }
  
  if (!requireNamespace("Biobase", quietly = TRUE)) {
    stop("❌ Package 'Biobase' is required but not installed.")
  }
  
  # ---- Ensure GEO Cache Directory Exists ----
  if (!is.null(logger)) {
    logger$log("📥 Downloading/loading GEO metadata for GSE68928...")
  }
  
  geo <- tryCatch({
    GEOquery::getGEO("GSE68928", GSEMatrix = TRUE, destdir = geo_dir)
  }, error = function(e) {
    stop("❌ Failed to download GEO metadata — ", e$message)
  })
  
  if (!is.null(logger)) {
    logger$log(glue::glue("✅ GEO metadata loaded: {length(geo)} platform object(s) retrieved."))
  }
  
  # ---- Map GEO Platforms to Internal Chip Names ----
  geo_map <- list(
    hu35ksuba = "GPL98",
    hu6800    = "GPL80"
  )
  
  geo_esets_mapped <- lapply(geo_map, function(gpl_id) {
    idx <- which(vapply(geo, function(x) Biobase::annotation(x), character(1)) == gpl_id)
    if (length(idx) == 1) {
      geo[[idx]]
    } else {
      NULL
    }
  })
  
  names(geo_esets_mapped) <- names(geo_map)
  
  # ---- Attach Metadata to Matching ExpressionSets ----
  for (chip in names(eset_list)) {
    eset <- eset_list[[chip]]
    geo_chip <- geo_esets_mapped[[chip]]
    
    if (is.null(geo_chip)) {
      msg <- glue::glue("⚠️ No GEO metadata found for chip type {chip}.")
      message(msg)
      if (!is.null(logger)) logger$log(msg)
      next
    }
    
    pheno_geo <- Biobase::pData(geo_chip)
    sample_ids <- sub("(_.*)$", "", Biobase::sampleNames(eset))
    
    rownames(pheno_geo) <- sub("\\.CEL(\\.gz)?$", "", rownames(pheno_geo), ignore.case = TRUE)
    matched_rows <- match(sample_ids, rownames(pheno_geo))
    
    if (any(!is.na(matched_rows))) {
      matched_pheno <- pheno_geo[matched_rows, , drop = FALSE]
      rownames(matched_pheno) <- sample_ids
      
      Biobase::pData(eset) <- matched_pheno
      
      if (!is.null(logger)) {
        n_matched <- sum(!is.na(matched_rows))
        logger$log(glue::glue("✅ Metadata attached for {chip}: matched {n_matched}/{length(sample_ids)} samples."))
      }
    } else {
      msg <- glue::glue("⚠️ No matching GEO rows for {chip}.")
      message(msg)
      if (!is.null(logger)) logger$log(msg)
    }
    
    eset_list[[chip]] <- eset
  }
  
  if (!is.null(logger)) {
    logger$log("🏁 GEO metadata attachment step completed.")
  }
  
  return(eset_list)
}