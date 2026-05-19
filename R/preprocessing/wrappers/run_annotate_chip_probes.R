#' Annotate all probes from full-chip ExpressionSets
#'
#' Annotates all probes for each chip in the provided `eset_list`, using the appropriate
#' Bioconductor annotation database. The results include gene symbols, GO terms (molecular function),
#' KEGG pathways, and MSigDB H hits. Output is saved as a `.rds` file if `annotations_path` is provided.
#'
#' @param eset_list A named list of `ExpressionSet` objects, keyed by chip ID.
#' @param chip_ids Character vector of chip IDs to annotate (e.g., `c("hu35ksuba", "hu6800")`).
#' @param annotations_path Optional string. Path to save the combined annotation list as `.rds`.
#' @param logger Optional pipeline logger.
#'
#' @return A named list of annotations for each chip, each containing:
#'   - `annotation_table`: Data frame of merged probe annotations
#'   - `go_counts`: Summary of GO terms (MF + BP)
#'   - `kegg_counts`: Summary of KEGG pathways
#'   - `msig_counts`: MSigDB H collection gene set matches
#'
#' @seealso \code{annotate_chip_probes}, \code{run_preprocessing_pipeline}
run_annotate_chip_probes <- function(
    eset_list,
    chip_ids,
    annotations_path = NULL,
    logger = NULL
) {
  # ---- Dependency Check ----
  if (!requireNamespace("Biobase", quietly = TRUE)) {
    stop("❌ Package 'Biobase' is required but not installed.")
  }
  
  # ---- Load Annotation Helper ----
  source(here::here("R/annotate/annotate_chip_probes.R"), local = TRUE)
  
  # ---- Validate Inputs ----
  if (!is.list(eset_list) || !all(chip_ids %in% names(eset_list))) {
    stop("❌ `eset_list` must be a named list and contain all requested `chip_ids`.")
  }
  
  annotations <- list()
  
  # ---- Annotation Loop ----
  for (chip in chip_ids) {
    eset <- eset_list[[chip]]
    probe_ids <- rownames(Biobase::exprs(eset))
    
    msg <- glue::glue("🔍 Annotating {length(probe_ids)} probes for chip: {chip}...")
    if (!is.null(logger)) logger$log(msg) else message(msg)
    
    annotations[[chip]] <- annotate_chip_probes(probe_ids, chip)
    
    msg <- glue::glue("✅ Completed annotation for chip: {chip}")
    if (!is.null(logger)) logger$log(msg) else message(msg)
  }
  
  # ---- Save If Path Provided ----
  if (!is.null(annotations_path)) {
    dir.create(dirname(annotations_path), recursive = TRUE, showWarnings = FALSE)
    saveRDS(annotations, file = annotations_path)
    
    msg <- glue::glue("💾 Full-chip annotations saved to: {basename(annotations_path)}")
    if (!is.null(logger)) logger$log(msg) else message(msg)
  }
  
  return(annotations)
}