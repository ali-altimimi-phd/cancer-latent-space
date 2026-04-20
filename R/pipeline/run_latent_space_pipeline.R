#' Run the Cancer Latent Space Preparation Pipeline
#'
#' @description
#' Executes the alignment, VAE-input extraction, and PCA sanity-check scripts in
#' sequence using separate `Rscript` calls.
#'
#' @details
#' Each stage is launched in its own clean R session. This preserves the logic
#' of the individual scripts, avoids environment contamination, and prevents
#' issues caused by script-level calls such as `rm(list = ls())`.
#'
#' Logging is handled through the shared pipeline logger helper located in
#' `R/helpers/pipeline_logger.R`.
#'
#' @section Pipeline order:
#' \enumerate{
#'   \item `01_fix_eset_alignment.R`
#'   \item `02_extract_vae_input_minimal.R`
#'   \item `03_pca_sanity_check_canonical.R`
#' }
#'
#' @section Assumption:
#' The script is run from the project root, or `scripts_dir` is updated to the
#' correct location.
#'
#' @seealso [run_stage()]
#' @keywords internal
#' @noRd

suppressPackageStartupMessages({
  library(here)
})

source(here::here("R/helpers/pipeline_logger.R"))

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

scripts_dir <- here::here("projects", "cancer-latent-space", "R", "pipeline")

scripts_to_run <- c(
  "01_fix_eset_alignment.R",
  "02_extract_vae_input_minimal.R",
  "03_pca_sanity_check_canonical.R"
)

log_dir <- here::here("projects", "cancer-latent-space", "output", "logs")
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(log_dir, "cancer_latent_space_pipeline.log")

logger <- start_log(logfile = log_file)
logger$log("🚀 Starting Cancer Latent Space preparation pipeline", section = "PIPELINE")

#' Run One Pipeline Stage in a Fresh R Session
#'
#' @description
#' Calls a single script with `Rscript`, logs progress, and stops the pipeline
#' immediately if the stage fails.
#'
#' @param script_path Character string giving the path to the script.
#' @param logger Logger object returned by `start_log()`.
#'
#' @return Invisibly returns `TRUE` on success.
#'
#' @examples
#' \dontrun{
#' run_stage("R/cancer-latent-space/01_fix_eset_alignment.R", logger)
#' }
#' @keywords internal
run_stage <- function(script_path, logger) {
  script_name <- basename(script_path)
  
  logger$timed(paste("Running", script_name), {
    logger$log(
      sprintf("Launching stage: %s", script_path),
      section = "STAGE"
    )
    
    exit_status <- system2(
      command = file.path(R.home("bin"), "Rscript"),
      args = shQuote(script_path)
    )
    
    if (!identical(exit_status, 0L)) {
      logger$log(
        sprintf("Stage failed with exit status %s: %s", exit_status, script_name),
        section = "ERROR"
      )
      stop(sprintf("Pipeline failed at stage: %s", script_path), call. = FALSE)
    }
    
    logger$log(
      sprintf("Completed stage successfully: %s", script_name),
      section = "STAGE"
    )
  })
  
  invisible(TRUE)
}

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

for (script_name in scripts_to_run) {
  script_path <- file.path(scripts_dir, script_name)
  
  if (!file.exists(script_path)) {
    logger$log(sprintf("Script not found: %s", script_path), section = "ERROR")
    stop(sprintf("Script not found: %s", script_path), call. = FALSE)
  }
  
  run_stage(script_path, logger)
}

logger$log("✅ Pipeline finished successfully", section = "PIPELINE")