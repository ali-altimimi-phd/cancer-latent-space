# ==============================================================================
# File: export_go_semantic_layer.R
# Purpose: Standalone exporter for p-value-neutral GO semantic annotation layer
# Role: Preprocessing-stage artifact builder; safe to test before pipeline integration
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi; generated/refactored with ChatGPT assistance
# Created: 2026
# ==============================================================================

export_go_semantic_layer <- function(annotation_path,
                                     output_dir,
                                     go_mode = c("BP", "MF"),
                                     min_probes = c(BP = 5, MF = 5),
                                     similarity_cutoff = 0.70,
                                     max_terms_per_block = 600,
                                     min_terms_per_ancestor_block = 25,
                                     large_block_strategy = c("ancestor_single_cluster", "singleton"),
                                     logger = NULL) {
  large_block_strategy <- match.arg(large_block_strategy)

  log_msg <- function(...) {
    msg <- paste0(...)
    if (!is.null(logger) && is.function(logger$log)) {
      logger$log(msg, section = "GO_SEMANTIC")
    } else {
      message(msg)
    }
  }

  if (!file.exists(annotation_path)) {
    stop("Annotation file does not exist: ", annotation_path, call. = FALSE)
  }

  fs::dir_create(output_dir)

  log_msg("Loading full chip annotations: ", annotation_path)
  full_chip_annotations <- readRDS(annotation_path)

  log_msg("Building GO semantic universe...")
  go_universe <- build_go_semantic_universe(
    full_chip_annotations = full_chip_annotations,
    go_mode = go_mode,
    min_probes = min_probes
  )

  go_term_dim <- build_go_semantic_term_dim(go_universe)

  log_msg("GO semantic universe rows: ", nrow(go_universe))
  log_msg("Unique GO terms for semantic clustering: ", nrow(go_term_dim))

  log_msg("Computing p-value-neutral GO semantic clusters...")
  go_cluster_membership <- build_go_semantic_clusters(
    go_term_dim = go_term_dim,
    similarity_cutoff = similarity_cutoff,
    max_terms_per_block = max_terms_per_block,
    min_terms_per_ancestor_block = min_terms_per_ancestor_block,
    large_block_strategy = large_block_strategy
  )

  go_cluster_dim <- summarize_go_semantic_clusters(go_cluster_membership)

  provenance <- tibble::tibble(
    artifact = "go_semantic_layer",
    created_at = as.character(Sys.time()),
    annotation_path = annotation_path,
    go_mode = paste(normalize_go_mode(go_mode), collapse = ","),
    min_probes = paste(names(min_probes), unlist(min_probes), sep = ":", collapse = ","),
    similarity_measure = "Wang",
    clustering_method = "GO ancestor prepartition, then Wang semantic similarity + ward.D2 within manageable blocks",
    similarity_cutoff = similarity_cutoff,
    max_terms_per_block = max_terms_per_block,
    min_terms_per_ancestor_block = min_terms_per_ancestor_block,
    large_block_strategy = large_block_strategy,
    note = "Semantic clusters are p-value-neutral. Structural/inferential summaries should be computed downstream by joining result facts to cluster membership."
  )

  readr::write_csv(go_universe, file.path(output_dir, "go_semantic_universe.csv"))
  readr::write_csv(go_term_dim, file.path(output_dir, "go_semantic_term_dim.csv"))
  readr::write_csv(go_cluster_membership, file.path(output_dir, "go_semantic_cluster_membership.csv"))
  readr::write_csv(go_cluster_dim, file.path(output_dir, "go_semantic_cluster_dim.csv"))
  readr::write_csv(provenance, file.path(output_dir, "go_semantic_provenance.csv"))

  saveRDS(go_universe, file.path(output_dir, "go_semantic_universe.rds"))
  saveRDS(go_term_dim, file.path(output_dir, "go_semantic_term_dim.rds"))
  saveRDS(go_cluster_membership, file.path(output_dir, "go_semantic_cluster_membership.rds"))
  saveRDS(go_cluster_dim, file.path(output_dir, "go_semantic_cluster_dim.rds"))
  saveRDS(provenance, file.path(output_dir, "go_semantic_provenance.rds"))

  log_msg("GO semantic layer written to: ", output_dir)
  log_msg("Cluster membership rows: ", nrow(go_cluster_membership))
  log_msg("Cluster dimension rows: ", nrow(go_cluster_dim))

  invisible(list(
    go_semantic_universe = go_universe,
    go_semantic_term_dim = go_term_dim,
    go_semantic_cluster_membership = go_cluster_membership,
    go_semantic_cluster_dim = go_cluster_dim,
    go_semantic_provenance = provenance
  ))
}
