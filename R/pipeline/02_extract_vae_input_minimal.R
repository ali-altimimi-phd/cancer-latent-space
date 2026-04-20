#' Extract VAE Input Matrices from Aligned ExpressionSets
#'
#' @description
#' Loads an aligned ExpressionSet list, exports one or more platform-specific
#' matrices for machine-learning use, and writes minimal canonical metadata plus
#' retained feature variances to disk.
#'
#' @details
#' This version intentionally exports a small, clean metadata contract for the
#' Python notebooks:
#'   - sample_id
#'   - geo_accession
#'   - title
#'   - platform_id
#'   - disease_clean
#'   - tissue_clean
#'   - condition
#'   - tissue_label
#'
#' Raw GEO-style metadata and legacy convenience columns are not exported.
#'
#' Logging is handled through the shared pipeline logger helper located in
#' `R/helpers/pipeline_logger.R`. When run standalone, the script initializes
#' its own stage-specific logger.
#'
#' @section Inputs:
#' \itemize{
#'   \item `projects/cancer-latent-space/data/inputs/global_cancer_eset_list_aligned.RData`
#'   \item `R/filters/select_high_variance_probes.R`
#' }
#'
#' @section Outputs:
#' For each selected platform, the script writes:
#' \itemize{
#'   \item `*_vae_input.rds`
#'   \item `*_vae_input.csv`
#'   \item `*_metadata.rds`
#'   \item `*_metadata.csv`
#'   \item `*_feature_variances.rds`
#'   \item `*_feature_variances.csv`
#'   \item `projects/cancer-latent-space/output/logs/02_extract_vae_input_minimal.log`
#' }
#'
#' @keywords internal
#' @noRd

suppressPackageStartupMessages({
  library(here)
  library(Biobase)
})

source(here::here("R/filters/select_high_variance_probes.R"))
source(here::here("R/helpers/pipeline_logger.R"))

log_dir <- here::here("projects", "cancer-latent-space", "output", "logs")
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

if (!exists("logger")) {
  logger <- start_log(logfile = file.path(log_dir, "02_extract_vae_input_minimal.log"))
}

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
input_file <- here::here(
  "projects",
  "cancer-latent-space",
  "data",
  "inputs",
  "global_cancer_eset_list_aligned.RData"
)
output_dir <- here::here(
  "projects",
  "cancer-latent-space",
  "data",
  "inputs"
)

platforms_to_export <- c("hu35ksuba")
n_features <- 3000
filter_method <- "variance"

#' Build Minimal Canonical Metadata for Python Export
#'
#' @param meta_raw Data frame returned by `pData(es)`, already aligned to samples.
#' @param platform_name Character scalar naming the platform.
#'
#' @return A data frame with canonical metadata columns only.
#' @keywords internal
build_export_metadata <- function(meta_raw, platform_name) {
  required_meta_cols <- c(
    "characteristics_ch1",
    "characteristics_ch1.1",
    "condition",
    "tissue_label"
  )

  missing_meta_cols <- setdiff(required_meta_cols, colnames(meta_raw))
  if (length(missing_meta_cols) > 0) {
    stop(
      sprintf(
        "Missing required metadata columns in %s: %s",
        platform_name,
        paste(missing_meta_cols, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  disease_clean <- sub("^disease state: ", "", meta_raw$characteristics_ch1)
  tissue_clean  <- sub("^organism part: ", "", meta_raw$characteristics_ch1.1)

  meta <- data.frame(
    sample_id = rownames(meta_raw),
    geo_accession = if ("geo_accession" %in% colnames(meta_raw)) {
      as.character(meta_raw$geo_accession)
    } else {
      rownames(meta_raw)
    },
    title = if ("title" %in% colnames(meta_raw)) {
      as.character(meta_raw$title)
    } else {
      NA_character_
    },
    platform_id = if ("platform_id" %in% colnames(meta_raw)) {
      as.character(meta_raw$platform_id)
    } else {
      rep(platform_name, nrow(meta_raw))
    },
    disease_clean = as.character(disease_clean),
    tissue_clean = as.character(tissue_clean),
    condition = as.character(meta_raw$condition),
    tissue_label = as.character(meta_raw$tissue_label),
    stringsAsFactors = FALSE,
    row.names = rownames(meta_raw)
  )

  meta
}

#' Extract a VAE-Ready Matrix and Matching Metadata
#'
#' @description
#' Converts a platform-specific `ExpressionSet` into a machine-learning matrix
#' with samples in rows and filtered features in columns.
#'
#' @param es A `Biobase::ExpressionSet` object.
#' @param platform_name A scalar character string naming the platform.
#' @param n_features Integer. Number of highest-variance features to retain.
#' @param filter_method Character string passed to `select_high_variance_probes()`.
#' @param logger Logger object returned by `start_log()`.
#'
#' @return A named list with elements `X`, `meta`, and `vars`.
#' @keywords internal
extract_vae_data <- function(es,
                             platform_name,
                             n_features = 3000,
                             filter_method = "variance",
                             logger) {
  logger$log(sprintf("Processing platform: %s", platform_name), section = "STEP")

  X <- exprs(es)

  if (anyNA(X)) {
    logger$log(sprintf("NA values detected in expression matrix for %s", platform_name), section = "ERROR")
    stop(sprintf("NA values detected in expression matrix for %s", platform_name), call. = FALSE)
  }

  if (!identical(colnames(X), rownames(pData(es)))) {
    logger$log(sprintf("Sample alignment failure in %s", platform_name), section = "ERROR")
    stop(sprintf("Sample alignment failure in %s", platform_name), call. = FALSE)
  }

  logger$log(
    sprintf("Original matrix dimensions (probes x samples): %s", paste(dim(X), collapse = " x ")),
    section = "DATA"
  )
  logger$log(
    sprintf("Expression range: %s", paste(signif(range(X, na.rm = TRUE), 5), collapse = " to ")),
    section = "DATA"
  )

  filter_result <- select_high_variance_probes(
    expr_matrix = X,
    method = filter_method,
    top_n = n_features,
    return_stats = TRUE
  )

  retained_probes <- filter_result$probes
  retained_vars <- filter_result$retained_stats

  if (length(retained_probes) < n_features) {
    logger$log(
      sprintf(
        "Requested %s features but only %s were retained; keeping all retained probes.",
        n_features,
        length(retained_probes)
      ),
      section = "FILTER"
    )
  }

  logger$log(
    sprintf(
      "Selected %s probes using method '%s' with selection mode '%s'.",
      length(retained_probes),
      filter_result$method,
      filter_result$selection_mode
    ),
    section = "FILTER"
  )

  X_filt <- X[retained_probes, , drop = FALSE]
  X_ml <- t(X_filt)

  meta_raw <- pData(es)
  meta_raw <- meta_raw[rownames(X_ml), , drop = FALSE]

  if (!identical(rownames(X_ml), rownames(meta_raw))) {
    logger$log(sprintf("Metadata alignment failure after filtering in %s", platform_name), section = "ERROR")
    stop(sprintf("Metadata alignment failure after filtering in %s", platform_name), call. = FALSE)
  }

  meta <- build_export_metadata(meta_raw, platform_name)

  if (!identical(rownames(X_ml), rownames(meta))) {
    logger$log(sprintf("Canonical metadata alignment failure in %s", platform_name), section = "ERROR")
    stop(sprintf("Canonical metadata alignment failure in %s", platform_name), call. = FALSE)
  }

  logger$log(
    sprintf("Filtered matrix dimensions (samples x features): %s", paste(dim(X_ml), collapse = " x ")),
    section = "FILTER"
  )
  logger$log(
    sprintf("Export metadata columns: %s", paste(colnames(meta), collapse = ", ")),
    section = "FILTER"
  )

  list(
    X = X_ml,
    meta = meta,
    vars = retained_vars
  )
}

#' Save VAE Matrix, Metadata, and Feature Variances
#'
#' @description
#' Writes a platform's filtered matrix, metadata, and retained variances as both
#' RDS and CSV files.
#'
#' @param obj A list returned by [extract_vae_data()].
#' @param platform_name A scalar character string naming the platform.
#' @param output_dir Output directory path.
#' @param logger Logger object returned by `start_log()`.
#'
#' @return No return value. Called for side effects.
#' @keywords internal
save_vae_outputs <- function(obj, platform_name, output_dir, logger) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  X <- obj$X
  meta <- obj$meta
  vars <- obj$vars

  saveRDS(X, file = file.path(output_dir, paste0(platform_name, "_vae_input.rds")))
  write.csv(
    X,
    file = file.path(output_dir, paste0(platform_name, "_vae_input.csv")),
    row.names = TRUE
  )

  saveRDS(meta, file = file.path(output_dir, paste0(platform_name, "_metadata.rds")))
  write.csv(
    meta,
    file = file.path(output_dir, paste0(platform_name, "_metadata.csv")),
    row.names = TRUE
  )

  saveRDS(
    vars,
    file = file.path(output_dir, paste0(platform_name, "_feature_variances.rds"))
  )
  write.csv(
    data.frame(feature = names(vars), variance = as.numeric(vars)),
    file = file.path(output_dir, paste0(platform_name, "_feature_variances.csv")),
    row.names = FALSE
  )

  logger$log(sprintf("Saved outputs for %s", platform_name), section = "SAVE")
}

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
logger$log(sprintf("Loading input file: %s", input_file), section = "LOAD")
load(input_file)

if (!exists("global_cancer_eset_list_aligned")) {
  logger$log("Object 'global_cancer_eset_list_aligned' not found after loading file.", section = "ERROR")
  stop("Object 'global_cancer_eset_list_aligned' not found after loading file.", call. = FALSE)
}

logger$log(
  sprintf("Platforms available: %s", paste(names(global_cancer_eset_list_aligned), collapse = ", ")),
  section = "INFO"
)

for (platform_name in platforms_to_export) {
  if (!platform_name %in% names(global_cancer_eset_list_aligned)) {
    logger$log(sprintf("Platform '%s' not found in aligned ExpressionSet list.", platform_name), section = "ERROR")
    stop(sprintf("Platform '%s' not found in aligned ExpressionSet list.", platform_name), call. = FALSE)
  }

  es <- global_cancer_eset_list_aligned[[platform_name]]

  result <- logger$timed(
    paste("Extract VAE data:", platform_name),
    extract_vae_data(
      es,
      platform_name,
      n_features = n_features,
      filter_method = filter_method,
      logger = logger
    )
  )

  logger$timed(
    paste("Save outputs:", platform_name),
    save_vae_outputs(result, platform_name, output_dir, logger = logger)
  )
}

logger$log("Done.", section = "PIPELINE")
