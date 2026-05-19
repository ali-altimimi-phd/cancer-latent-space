# ------------------------------------------------------------------------------
# File: filter_probes_limma.R
# Purpose: Perform probe-level differential expression filtering using limma
#   for Affymetrix expression data
# Role: Probe-level differential expression utility
# Pipeline: Preprocessing
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

# ---- Dependencies ------------------------------------------------------------
library(limma)
library(Biobase)
library(tibble)
library(dplyr)
library(glue)
library(here)

#' Filter probes using limma differential expression analysis
#'
#' Fits a linear model using the limma pipeline to identify probes
#' with significant differential expression between two conditions.
#'
#' @param eset An `ExpressionSet` object containing expression and phenotype data.
#' @param condition_col String. Column in `pData(eset)` specifying the experimental condition.
#' @param normal_label String. Label for control/normal samples. Default = "normal".
#' @param cancer_label String. Label for experimental/cancer samples. Default = "cancer".
#' @param logfc_cutoff Numeric. Minimum absolute log2 fold-change threshold.
#' @param pval_cutoff Numeric. Maximum adjusted p-value (FDR) threshold.
#' @param logfile Optional string path. If provided, logs result summary to this file.
#'
#' @return A named list containing:
#'   - `filtered_matrix`: matrix of filtered expression values
#'   - `filtered_probes`: character vector of selected probe IDs
#'   - `stats_table`: tibble with differential expression results for selected probes
#'   - `limma_full`: tibble with full limma results for all probes
filter_probes_limma <- function(eset,
                                condition_col = "condition",
                                normal_label = "normal",
                                cancer_label = "cancer",
                                logfc_cutoff,
                                pval_cutoff,
                                logfile = NULL) {
  
  # --- Input validation ---
  if (!inherits(eset, "ExpressionSet")) {
    stop("Input must be an ExpressionSet object.")
  }
  
  pheno <- pData(eset)
  if (!condition_col %in% names(pheno)) {
    stop(glue("Metadata column '{condition_col}' not found in pData."))
  }
  
  condition <- factor(pheno[[condition_col]])
  if (!all(c(normal_label, cancer_label) %in% levels(condition))) {
    stop(glue("Both '{normal_label}' and '{cancer_label}' must be present in the '{condition_col}' column."))
  }
  
  # --- Design matrix and contrast setup ---
  design <- model.matrix(~ 0 + condition)
  colnames(design) <- levels(condition)
  
  contrast <- makeContrasts(
    contrasts = glue("{cancer_label} - {normal_label}"),
    levels = design
  )
  
  # --- Fit model and apply empirical Bayes smoothing ---
  fit <- lmFit(eset, design)
  fit2 <- contrasts.fit(fit, contrast)
  fit2 <- eBayes(fit2)
  
  # --- Retrieve all limma results ---
  tab <- topTable(fit2, adjust = "fdr", number = Inf) |>
    rownames_to_column("probe_id")
  
  # --- Filter by fold-change and adjusted p-value thresholds ---
  filtered_tab <- tab |>
    filter(abs(logFC) >= logfc_cutoff, adj.P.Val < pval_cutoff)
  
  filtered_probes <- filtered_tab$probe_id
  filtered_exprs <- exprs(eset)[filtered_probes, , drop = FALSE]
  
  # --- Optional logging ---
  if (!is.null(logfile)) {
    log_entry <- glue(
      "[limma filter] {Sys.time()} | Chip: {annotation(eset)} | ",
      "Probes total: {nrow(tab)} | Passed: {length(filtered_probes)} | ",
      "logFC ≥ {logfc_cutoff} & adj.P.Val < {pval_cutoff}\n"
    )
    cat(log_entry, file = logfile, append = TRUE)
  }
  
  # --- Return results ---
  list(
    filtered_matrix = filtered_exprs,
    filtered_probes = filtered_probes,
    stats_table = filtered_tab,
    limma_full = tab
  )
}
