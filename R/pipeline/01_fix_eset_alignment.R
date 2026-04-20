#' Fix ExpressionSet Sample Alignment
#'
#' @description
#' Loads the previously created `global_cancer_eset_list` object, aligns
#' phenotype-data row names to expression-matrix column names by matching GSM
#' accessions embedded in CEL-derived sample identifiers, verifies the result
#' for each platform, and saves a corrected ExpressionSet list.
#'
#' @details
#' This script is intended to run after the preprocessing pipeline has created
#' `global_cancer_eset_list.RData`. The alignment step is necessary because the
#' expression matrix may use CEL-style identifiers such as
#' `GSM1686773_CL2000062805AA.CEL`, whereas `pData(es)` often uses only the GSM
#' accession, such as `GSM1686773`.
#'
#' Logging is handled through the shared pipeline logger helper located in
#' `R/helpers/pipeline_logger.R`. When run standalone, the script initializes
#' its own stage-specific logger.
#'
#' @section Inputs:
#' \itemize{
#'   \item `output/global_cancer/RData/global_cancer_eset_list.RData`
#' }
#'
#' @section Outputs:
#' \itemize{
#'   \item `projects/cancer-latent-space/data/inputs/global_cancer_eset_list_aligned.RData`
#'   \item `projects/cancer-latent-space/output/logs/01_fix_eset_alignment.log`
#' }
#'
#' @seealso [fix_eset_alignment()]
#' @keywords internal
#' @noRd

suppressPackageStartupMessages({
  library(here)
  library(Biobase)
})

source(here::here("R/helpers/pipeline_logger.R"))

log_dir <- here::here("projects", "cancer-latent-space", "output", "logs")
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

if (!exists("logger")) {
  logger <- start_log(logfile = file.path(log_dir, "01_fix_eset_alignment.log"))
}

#' Align `pData` Row Names to Expression-Matrix Column Names
#'
#' @description
#' Reorders and renames phenotype metadata rows so that they match the
#' expression-matrix sample order exactly.
#'
#' @param es A `Biobase::ExpressionSet` object.
#'
#' @return A corrected `Biobase::ExpressionSet` whose `pData(es)` row names are
#'   identical to `colnames(exprs(es))`.
#'
#' @details
#' GSM accessions are extracted from expression-column identifiers using the
#' regular expression `^(GSM[0-9]+).*$`. Metadata rows are then reordered with
#' `match()`, and the resulting phenotype row names are replaced with the full
#' expression identifiers.
#'
#' @examples
#' \dontrun{
#' es_fixed <- fix_eset_alignment(es)
#' identical(colnames(Biobase::exprs(es_fixed)), rownames(Biobase::pData(es_fixed)))
#' }
fix_eset_alignment <- function(es) {
  expr_ids <- colnames(Biobase::exprs(es))
  pheno_ids <- rownames(Biobase::pData(es))

  expr_gsm <- sub("^(GSM[0-9]+).*$", "\\1", expr_ids)
  idx <- match(expr_gsm, pheno_ids)

  if (any(is.na(idx))) {
    missing_expr <- expr_ids[is.na(idx)]
    stop(
      paste0(
        "Unmatched samples detected: ",
        paste(head(missing_expr, 10), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  pd <- Biobase::pData(es)[idx, , drop = FALSE]
  rownames(pd) <- expr_ids
  Biobase::pData(es) <- pd

  es
}

input_file <- here("output",
                   "global_cancer",
                   "RData",
                   "global_cancer_eset_list.RData")
output_file <- here(
  "projects",
  "cancer-latent-space",
  "data",
  "inputs",
  "global_cancer_eset_list_aligned.RData"
)

logger$log(sprintf("Loading input file: %s", input_file), section = "LOAD")
load(input_file)

if (!exists("eset_list")) {
  logger$log("Object 'eset_list' was not found after loading the input file.", section = "ERROR")
  stop("Object 'eset_list' was not found after loading the input file.", call. = FALSE)
}

global_cancer_eset_list <- eset_list

logger$log(
  sprintf("Platforms found: %s", paste(names(global_cancer_eset_list), collapse = ", ")),
  section = "INFO"
)

global_cancer_eset_list_aligned <- logger$timed("Align ExpressionSets", {
  aligned <- lapply(global_cancer_eset_list, fix_eset_alignment)
  names(aligned) <- names(global_cancer_eset_list)
  aligned
})

logger$log("Running alignment verification across platforms", section = "CHECK")

for (nm in names(global_cancer_eset_list_aligned)) {
  es <- global_cancer_eset_list_aligned[[nm]]
  alignment_ok <- identical(
    colnames(Biobase::exprs(es)),
    rownames(Biobase::pData(es))
  )

  logger$log(
    sprintf(
      "Platform=%s | exprs=%s | pData=%s | alignment_ok=%s",
      nm,
      paste(dim(Biobase::exprs(es)), collapse = " x "),
      paste(dim(Biobase::pData(es)), collapse = " x "),
      alignment_ok
    ),
    section = "CHECK"
  )
}

all_ok <- all(vapply(global_cancer_eset_list_aligned, function(es) {
  identical(colnames(Biobase::exprs(es)), rownames(Biobase::pData(es)))
}, logical(1)))

logger$log(sprintf("All platforms aligned: %s", all_ok), section = "CHECK")

logger$timed("Save aligned ExpressionSet list", {
  dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
  save(global_cancer_eset_list_aligned, file = output_file)
})

logger$log(sprintf("Saved corrected object to: %s", output_file), section = "SAVE")
