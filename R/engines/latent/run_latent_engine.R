# ------------------------------------------------------------------------------
# File: run_latent_engine.R
# Purpose: Execute latent-space Python backend scripts
# Role: Orchestrates latent preprocessing, VAE training, and latent metrics
# Pipeline: Structural Inference
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

# ==============================================================================
# Latent-space Python backend execution helpers
# ==============================================================================

# These helpers execute the production latent-space Python backend using
# canonical structural-inference configuration values.
#
# R owns:
#
#   - orchestration
#   - configuration
#   - provenance
#   - logging
#   - downstream synthesis integration
#
# Python acts as a computational backend responsible for:
#
#   - preprocessing
#   - VAE training
#   - latent coordinate generation
#   - latent structural metric computation
#
# The latent-space engine is treated as a structural engine parallel to:
#
#   - complexity
#   - entropy
#   - MP spectral analysis
#
# Canonical latent outputs are later consumed by:
#
#   - structural phenotype synthesis
#   - DuckDB ingestion
#   - integrated structural plotting

# ==============================================================================
# Execute a single latent-space Python script
# ==============================================================================
run_latent_python_script <- function(script_name, logger, extra_args = character()) {
  script_path <- file.path(latent_script_dir, script_name)
  
  if (!file.exists(script_path)) {
    logger$log(sprintf("Python script not found: %s", script_path), section = "ERROR")
    stop(sprintf("Python script not found: %s", script_path), call. = FALSE)
  }
  
  args <- c(
    shQuote(script_path),
    "--project-dir", shQuote(here::here()),
    "--chip-id", latent_chip_id,
    "--filter-regime", latent_filter_regime,
    "--top-n", as.character(latent_top_n),
    extra_args
  )
  
  logger$timed(sprintf("Python script %s", script_name), {
    logger$log(sprintf("Launching Python script: %s", script_path), section = "PYTHON")
    
    output <- tryCatch(
      system2(command = latent_python_exe, args = args, stdout = TRUE, stderr = TRUE),
      warning = function(w) structure(conditionMessage(w), status = 1L),
      error = function(e) structure(conditionMessage(e), status = 1L)
    )
    
    if (length(output) > 0) {
      for (line in output) logger$log(as.character(line), section = "PYTHON")
    }
    
    exit_status <- attr(output, "status")
    if (is.null(exit_status)) exit_status <- 0L
    
    if (!identical(as.integer(exit_status), 0L)) {
      logger$log(sprintf("Python script failed with exit status %s: %s", exit_status, script_name), section = "ERROR")
      stop(sprintf("Python script failed: %s", script_name), call. = FALSE)
    }
    
    logger$log(sprintf("Completed Python script successfully: %s", script_name), section = "PYTHON")
  })
  
  invisible(TRUE)
}

# ==============================================================================
# Execute latent-space Python backend stages
# ==============================================================================
run_latent_python_scripts_stage <- function(logger) {
  logger$log("🐍 Executing latent-space Python scripts...", section = "PYTHON")
  
  run_latent_python_script(
    script_name = "run_latent_preprocessing.py",
    logger = logger
  )
  
  run_latent_python_script(
    script_name = "run_latent_training.py",
    logger = logger,
    extra_args = c(
      "--latent-dim", as.character(latent_dim),
      "--epochs", as.character(latent_epochs),
      "--batch-size", as.character(latent_batch_size),
      "--beta", as.character(latent_beta),
      "--learning-rate", as.character(latent_learning_rate)
    )
  )
  
  run_latent_python_script(
    script_name = "run_latent_tables.py",
    logger = logger
  )
  
  run_latent_python_script(
    script_name = "run_latent_metrics.py",
    logger = logger
  )
  
  logger$log("✅ Latent-space Python scripts completed successfully.", section = "PYTHON")
}