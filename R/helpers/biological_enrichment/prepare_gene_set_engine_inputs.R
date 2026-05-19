# ------------------------------------------------------------------------------
# File: prepare_gene_set_engine_inputs.R
# Purpose: Adapt validated biological gene-set matrices to structural engine format
# Role: Biological interpretation helper
# Pipeline: Biological Interpretation
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

# ==============================================================================
# Biological gene-set engine input adapter
# ==============================================================================
#
# The structural complexity/entropy engines expect a named matrix list and a pair
# of comparison labels. Internally, those engines call get_group_matrices(), which
# converts labels into matrix keys using:
#
#   matrix_key_from_label("normal") -> "m_normal"
#   matrix_key_from_label("tumor")  -> "m_tumor"
#
# In the structural inference pipeline, matrices are stored with full biological
# tissue/disease-derived names. In the biological interpretation pipeline,
# run_pairwise_analysis() owns comparison-level admissibility checks and passes
# only validated normal/tumor matrices and usable probes to this adapter.
#
# Therefore this adapter creates a temporary engine-compatible object:
#
#   matrices_i$m_normal
#   matrices_i$m_tumor
#   comparison_labels = c("normal", "tumor")
#
# This allows biological interpretation to reuse the structural complexity/entropy
# pair-comparison engines without modifying the structural matrix-resolution
# helper used by the structural inference pipeline.
#
# This adapter does not decide whether a gene set should be skipped. Low-probe
# and other biological admissibility decisions belong in run_pairwise_analysis().

prepare_gene_set_engine_inputs <- function(
    mat_normal,
    mat_tumor,
    selected_probes
) {
  if (is.null(mat_normal) || is.null(mat_tumor)) {
    stop("mat_normal and mat_tumor must both be supplied.", call. = FALSE)
  }
  
  if (is.null(rownames(mat_normal)) || is.null(rownames(mat_tumor))) {
    stop("mat_normal and mat_tumor must have probe IDs as rownames.", call. = FALSE)
  }
  
  if (missing(selected_probes) || is.null(selected_probes)) {
    stop("selected_probes must be supplied.", call. = FALSE)
  }
  
  selected_probes_shared <- Reduce(
    intersect,
    list(
      unique(selected_probes),
      rownames(mat_normal),
      rownames(mat_tumor)
    )
  )
  
  mat_normal_subset <- mat_normal[selected_probes_shared, , drop = FALSE]
  mat_tumor_subset  <- mat_tumor[selected_probes_shared, , drop = FALSE]
  
  list(
    matrices_i = list(
      m_normal = mat_normal_subset,
      m_tumor  = mat_tumor_subset
    ),
    comparison_labels = c("normal", "tumor"),
    selected_probes = selected_probes_shared,
    n_shared_probes = length(selected_probes_shared)
  )
}