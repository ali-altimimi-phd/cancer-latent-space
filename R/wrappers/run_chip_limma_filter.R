# wrappers/run_chip_limma_filter.R
# =============================================================
# Wrapper for running limma-based probe filtering on a chip-specific ExpressionSet.
# Handles caching, logging, and result persistence automatically.
# =============================================================
#' Run limma-based probe filtering for a specific chip
#'
#' This wrapper manages rerun logic, disk I/O, and logging for limma-based filtering
#' using preprocessed ExpressionSet data. If results already exist and `rerun = FALSE`,
#' the saved results are loaded and returned.
#'
#' @param eset An ExpressionSet object for the chip.
#' @param chip_name String. The chip ID (e.g., "hu35ksuba", "hu6800").
#' @param logfc_cutoff Numeric. Minimum log2 fold-change threshold. Default = 0.33.
#' @param pval_cutoff Numeric. Maximum adjusted p-value threshold. Default = 0.05.
#' @param rerun Logical. Force re-run even if results are saved. Default = FALSE.
#'
#' @return A named list `result` with filtered probes and matrices.
#' @export
run_chip_limma_filter <- function(eset,
                                  chip_name,
                                  logfc_cutoff = 1,
                                  pval_cutoff = 0.05,
                                  rerun = FALSE) {
  stopifnot(inherits(eset, "ExpressionSet"))
  
  # ---- Define paths for saving results and logs ----
  output_path <- here::here("output", "RData", glue::glue("limma_filtered_{chip_name}.RData"))
  log_path    <- here::here("output", "logs", "limma", glue::glue("limma_filter_{chip_name}.log"))
  
  # ---- Load from cache if result already exists ----
  if (file.exists(output_path) && !rerun) {
    message(glue::glue("✅ Using cached limma result for chip: {chip_name}"))
    load(output_path, envir = .GlobalEnv)
    return(get("result", envir = .GlobalEnv))
  }
  
  # ---- Run filtering using limma logic ----
  message(glue::glue("🚧 Running limma filtering for chip: {chip_name}..."))
  source(here::here("R/filters/filter_probes_limma.R"), local = TRUE)
  
  result <- filter_probes_limma(
    eset,
    condition_col = "condition",
    normal_label = "normal",
    cancer_label = "cancer",
    logfc_cutoff = logfc_cutoff,
    pval_cutoff = pval_cutoff,
    logfile = log_path
  )
  
  # ---- Save results to disk ----
  save(result, file = output_path)
  message(glue::glue("💾 Saved limma result to: {output_path}"))
  
  return(result)
}
