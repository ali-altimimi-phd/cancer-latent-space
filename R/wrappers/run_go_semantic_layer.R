# ==============================================================================
# File: run_go_semantic_layer.R
# Purpose: Run p-value-neutral GO semantic annotation layer
# Role: Preprocessing wrapper
# Pipeline: Preprocessing
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ==============================================================================

#' Run GO Semantic Annotation Layer
#'
#' @description
#' Builds a p-value-neutral, analysis-eligible GO semantic clustering layer from
#' the durable chip annotation object produced during preprocessing.
#'
#' @details
#' This wrapper constructs reusable GO semantic metadata. It does not use
#' biological result p-values and does not perform enrichment or structural
#' inference. The resulting semantic layer can later be joined to biological
#' structural outputs in DuckDB/reporting.
#'
#' @param annotation_path Character. Path to full_chip_annotations.rds.
#' @param output_dir Character. Output directory for semantic layer CSV/RDS files.
#' @param go_mode Character vector. GO ontologies to include, usually c("BP", "MF").
#' @param min_probes Named numeric/integer vector. Minimum probe support by ontology.
#' @param similarity_cutoff Numeric. Semantic similarity threshold for clustering.
#' @param max_terms_per_block Integer. Maximum terms per block for pairwise similarity.
#' @param min_terms_per_ancestor_block Integer. Minimum size for ancestor preblocks.
#' @param large_block_strategy Character. Strategy for oversized semantic blocks.
#' @param logger Optional pipeline logger with a $log method.
#'
#' @return List returned by export_go_semantic_layer().
#' @export
run_go_semantic_layer <- function(annotation_path,
                                  output_dir,
                                  go_mode = c("BP", "MF"),
                                  min_probes = c(BP = 5, MF = 5),
                                  similarity_cutoff = 0.70,
                                  max_terms_per_block = 600,
                                  min_terms_per_ancestor_block = 25,
                                  large_block_strategy = "ancestor_single_cluster",
                                  logger = NULL) {
  source(here::here("R/preprocessing/go_analysis/build_go_semantic_universe.R"), local = TRUE)
  source(here::here("R/preprocessing/go_analysis/build_go_semantic_clusters.R"), local = TRUE)
  source(here::here("R/preprocessing/go_analysis/export_go_semantic_layer.R"), local = TRUE)
  
  export_go_semantic_layer(
    annotation_path = annotation_path,
    output_dir = output_dir,
    go_mode = go_mode,
    min_probes = min_probes,
    similarity_cutoff = similarity_cutoff,
    max_terms_per_block = max_terms_per_block,
    min_terms_per_ancestor_block = min_terms_per_ancestor_block,
    large_block_strategy = large_block_strategy,
    logger = logger
  )
}