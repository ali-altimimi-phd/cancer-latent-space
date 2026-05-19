#' This script establishes the correspondence between empirical measurements
#' (probe intensities) and the structured ontology of biological processes, 
#' thereby enabling the interpretation of statistical complexity 
#' in terms of functional organization within the cell.
NULL
#' Load chip annotations if not already loaded
#'
#' Uses the global `annotations_path` variable to locate the annotations file.
#' If `annotations` does not exist or is NULL, it loads from disk.
#' Logs progress via a provided logger or message if logger is missing.
#'
#' @param logger Optional logger object with `$log()` method.
#' @return The loaded annotations list (invisible).
#' @export
load_annotations_if_needed <- function(logger = NULL) {
  if (!exists("annotations", envir = .GlobalEnv) || is.null(get("annotations", envir = .GlobalEnv))) {
    msg <- "📦 Loading chip annotations..."
    if (!is.null(logger)) logger$log(msg) else message(msg)
    
    if (!exists("annotations_path", envir = .GlobalEnv)) {
      stop("❌ `annotations_path` variable not found in global environment.")
    }
    
    path <- get("annotations_path", envir = .GlobalEnv)
    ann <- readRDS(path)
    
    assign("annotations", ann, envir = .GlobalEnv)
    
    msg2 <- glue::glue("✅ Loaded annotations from: {path}")
    if (!is.null(logger)) logger$log(msg2) else message(msg2)
  } else {
    msg <- "✅ Using existing loaded annotations."
    if (!is.null(logger)) logger$log(msg) else message(msg)
  }
  invisible(get("annotations", envir = .GlobalEnv))
}
