# ------------------------------------------------------------------------------
# File: export_ml_inputs.R
# Purpose: Export ExpressionSet-derived machine-learning inputs
# Role: Final preprocessing-stage export and validation for latent-space / ML workflows
# Pipeline: Preprocessing
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Export Machine-Learning Inputs from ExpressionSet Objects
#'
#' @description
#' Extracts an expression matrix and metadata from a selected ExpressionSet,
#' validates canonical alignment between expression columns and metadata rows,
#' writes full and variance-global filtered expression matrices, and optionally
#' runs a PCA sanity check.
#'
#' @param eset_list Named list of ExpressionSet objects.
#' @param output_dir Directory for CSV outputs.
#' @param plot_dir Directory for PCA plot output.
#' @param chip_id Character chip/platform key in `eset_list`.
#' @param top_n Integer number of high-variance probes to retain.
#' @param filter_regime Character filter regime. Currently supports "variance_global".
#' @param run_pca_check Logical; whether to produce a PCA plot.
#' @param logger Logger object returned by `start_log()`.
#'
#' @return Invisibly returns a list containing output file paths.
#' @export
export_ml_inputs_from_eset <- function(eset_list,
                                       output_dir,
                                       plot_dir,
                                       chip_id = "hu35ksuba",
                                       top_n = 3000,
                                       filter_regime = "variance_global",
                                       run_pca_check = TRUE,
                                       logger = NULL) {
  log_it <- function(msg, section = "ML_EXPORT") {
    if (!is.null(logger)) {
      logger$log(msg, section = section)
    } else {
      message(msg)
    }
  }
  
  if (!chip_id %in% names(eset_list)) {
    stop(
      sprintf(
        "chip_id '%s' not found in eset_list. Available chips: %s",
        chip_id,
        paste(names(eset_list), collapse = ", ")
      ),
      call. = FALSE
    )
  }
  
  if (!requireNamespace("Biobase", quietly = TRUE)) {
    stop("Package 'Biobase' is required to extract exprs() and pData().",
         call. = FALSE)
  }
  
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
  
  eset <- eset_list[[chip_id]]
  
  log_it(sprintf("Extracting expression matrix for chip: %s", chip_id))
  expr <- Biobase::exprs(eset)
  meta <- Biobase::pData(eset)
  
  if (is.null(colnames(expr)) || is.null(rownames(meta))) {
    stop(
      "Expression matrix colnames and metadata rownames are required for alignment.",
      call. = FALSE
    )
  }
  
  expr_sample_ids <- colnames(expr)
  
  if (!identical(rownames(meta), expr_sample_ids)) {
    log_it(
      "Metadata rownames do not match expression columns; falling back to GSM-based alignment."
    )
    
    expr_gsm <- sub("_.*$", "", expr_sample_ids)
    
    if (!"geo_accession" %in% colnames(meta)) {
      stop("Metadata must contain a 'geo_accession' column for GSM-based alignment.",
           call. = FALSE)
    }
    
    match_idx <- match(expr_gsm, meta$geo_accession)
    
    if (any(is.na(match_idx))) {
      missing_ids <- expr_sample_ids[is.na(match_idx)]
      stop(
        sprintf(
          "Metadata is missing %d expression sample IDs after GSM matching. First missing ID: %s",
          length(missing_ids),
          missing_ids[[1]]
        ),
        call. = FALSE
      )
    }
    
    meta <- meta[match_idx, , drop = FALSE]
    rownames(meta) <- expr_sample_ids
  }
  
  if (!identical(rownames(meta), colnames(expr))) {
    stop(
      "Metadata alignment failed: metadata rownames do not match expression columns.",
      call. = FALSE
    )
  }
  
  log_it(sprintf("Metadata alignment verified for chip: %s", chip_id))
  
  meta$sample_id <- rownames(meta)
  
  if (!"disease_clean" %in% colnames(meta)) {
    if ("disease state:ch1" %in% colnames(meta)) {
      meta$disease_clean <- meta[["disease state:ch1"]]
    } else if ("characteristics_ch1" %in% colnames(meta)) {
      meta$disease_clean <- meta[["characteristics_ch1"]]
    } else {
      meta$disease_clean <- NA_character_
    }
  }
  
  if (!"tissue_clean" %in% colnames(meta)) {
    if ("organism part:ch1" %in% colnames(meta)) {
      meta$tissue_clean <- meta[["organism part:ch1"]]
    } else if ("characteristics_ch1.1" %in% colnames(meta)) {
      meta$tissue_clean <- meta[["characteristics_ch1.1"]]
    } else {
      meta$tissue_clean <- NA_character_
    }
  }
  
  log_it(sprintf(
    "Aligned metadata: %d samples x %d columns",
    nrow(meta),
    ncol(meta)
  ))
  log_it(sprintf(
    "Full expression matrix: %d probes x %d samples",
    nrow(expr),
    ncol(expr)
  ))
  
  expr_samples_by_features <- as.data.frame(t(expr), check.names = FALSE)
  expr_samples_by_features$sample_id <- rownames(expr_samples_by_features)
  expr_samples_by_features <- expr_samples_by_features[, c("sample_id", setdiff(names(expr_samples_by_features), "sample_id"))]
  
  meta_export <- as.data.frame(meta, check.names = FALSE)
  meta_export <- meta_export[, c("sample_id", setdiff(names(meta_export), "sample_id"))]
  
  full_expr_path <- file.path(output_dir, sprintf("%s_expr_full.csv", chip_id))
  metadata_path  <- file.path(output_dir, sprintf("%s_metadata_aligned.csv", chip_id))
  
  utils::write.csv(expr_samples_by_features, full_expr_path, row.names = FALSE)
  utils::write.csv(meta_export, metadata_path, row.names = FALSE)
  
  log_it(sprintf("Wrote full expression matrix: %s", full_expr_path))
  log_it(sprintf("Wrote aligned metadata: %s", metadata_path))
  
  if (!identical(filter_regime, "variance_global")) {
    stop(
      sprintf(
        "Unsupported filter_regime: %s. Currently supported: 'variance_global'.",
        filter_regime
      ),
      call. = FALSE
    )
  }
  
  if (!is.numeric(top_n) || length(top_n) != 1L || top_n <= 0) {
    stop("top_n must be a positive numeric scalar.", call. = FALSE)
  }
  
  top_n <- min(as.integer(top_n), nrow(expr))
  
  probe_variance <- apply(expr, 1, stats::var, na.rm = TRUE)
  top_probes <- names(sort(probe_variance, decreasing = TRUE))[seq_len(top_n)]
  expr_top <- expr[top_probes, , drop = FALSE]
  
  expr_top_samples_by_features <- as.data.frame(t(expr_top), check.names = FALSE)
  expr_top_samples_by_features$sample_id <- rownames(expr_top_samples_by_features)
  expr_top_samples_by_features <- expr_top_samples_by_features[, c("sample_id", setdiff(names(expr_top_samples_by_features), "sample_id"))]
  
  top_expr_path <- file.path(
    output_dir,
    sprintf("%s_expr_top%d_%s.csv", chip_id, top_n, filter_regime)
  )
  
  top_features_path <- file.path(
    output_dir,
    sprintf("%s_top%d_%s_features.csv", chip_id, top_n, filter_regime)
  )
  
  utils::write.csv(expr_top_samples_by_features, top_expr_path, row.names = FALSE)
  utils::write.csv(
    data.frame(
      probe_id = top_probes,
      variance = probe_variance[top_probes],
      rank = seq_along(top_probes),
      row.names = NULL
    ),
    top_features_path,
    row.names = FALSE
  )
  
  log_it(sprintf("Wrote filtered expression matrix: %s", top_expr_path))
  log_it(sprintf("Wrote filtered feature list: %s", top_features_path))
  
  pca_plot_path <- NULL
  
  if (isTRUE(run_pca_check)) {
    log_it("Running PCA check", section = "PCA")
    
    pca <- stats::prcomp(t(expr_top), center = TRUE, scale. = TRUE)
    pca_df <- data.frame(
      sample_id = rownames(pca$x),
      PC1 = pca$x[, 1],
      PC2 = pca$x[, 2],
      stringsAsFactors = FALSE
    )
    
    if ("condition" %in% names(meta)) {
      pca_df$condition <- meta[pca_df$sample_id, "condition"]
    } else {
      pca_df$condition <- "unknown"
    }
    
    if (requireNamespace("ggplot2", quietly = TRUE)) {
      p <- ggplot2::ggplot(pca_df, ggplot2::aes(x = PC1, y = PC2, color = condition)) +
        ggplot2::geom_point(alpha = 0.8, size = 2) +
        ggplot2::theme_minimal() +
        ggplot2::labs(
          title = sprintf("PCA: %s", chip_id),
          subtitle = sprintf("Top %d probes by variance-global regime", top_n),
          x = "PC1",
          y = "PC2",
          color = "Condition"
        )
      
      pca_plot_path <- file.path(
        plot_dir,
        sprintf("%s_pca_top%d_%s.png", chip_id, top_n, filter_regime)
      )
      
      ggplot2::ggsave(
        pca_plot_path,
        p,
        width = 8,
        height = 6,
        dpi = 300
      )
      log_it(sprintf("Wrote PCA plot: %s", pca_plot_path), section = "PCA")
    } else {
      log_it("ggplot2 not installed; PCA plot skipped.", section = "PCA")
    }
  }
  
  invisible(
    list(
      full_expression = full_expr_path,
      filtered_expression = top_expr_path,
      metadata = metadata_path,
      features = top_features_path,
      pca_plot = pca_plot_path
    )
  )
}