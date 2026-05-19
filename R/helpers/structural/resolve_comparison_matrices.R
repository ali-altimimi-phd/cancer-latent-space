# ==============================================================================
# Resolve comparison matrices
# ==============================================================================
#
# Purpose:
#   Helper functions for translating comparison maps and filtered-probe objects
#   into the paired normal/tumor matrices required by spectral analyses.
#
# ==============================================================================

#' Load an RDS file or stop with a clear message
#'
#' @param path Path to an RDS file.
#' @return Deserialized R object.
load_required_rds <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf("Required file not found: %s", path), call. = FALSE)
  }
  readRDS(path)
}

#' Extract selected probe IDs from a filtered-probe object
#'
#' @param x Filtered-probe object or character vector.
#' @return Character vector of selected probe IDs.
extract_selected_probes <- function(x) {
  if (is.character(x)) {
    return(x)
  }

  if (is.list(x)) {
    candidate_names <- c(
      "filtered_probes",
      "selected_probes",
      "probes",
      "probe_ids",
      "selected",
      "top_probes",
      "probes_used"
    )

    for (nm in candidate_names) {
      if (!is.null(x[[nm]]) && is.character(x[[nm]])) {
        return(x[[nm]])
      }
    }
  }

  stop("Could not extract selected probes from filtered-probe object.", call. = FALSE)
}

#' Convert a tissue/disease label to the matrix-list key used by the pipeline
#'
#' @param label Label such as "Bladder/normal".
#' @return Matrix key such as "m_Bladder.normal".
matrix_key_from_label <- function(label) {
  paste0("m_", make.names(gsub("/", ".", label)))
}

#' Resolve paired normal/tumor matrices for one comparison
#'
#' @param matrices_i Named list of condition-specific expression matrices.
#' @param comparison_labels Character vector of length two: normal label, tumor label.
#' @param selected_probes Character vector of selected probe IDs.
#' @return List with normal/tumor matrices and source metadata.
get_group_matrices <- function(matrices_i,
                               comparison_labels,
                               selected_probes) {

  if (length(comparison_labels) != 2) {
    stop("comparison_labels must contain exactly two labels: normal and tumor.", call. = FALSE)
  }

  normal_label <- comparison_labels[1]
  tumor_label  <- comparison_labels[2]

  normal_key <- matrix_key_from_label(normal_label)
  tumor_key  <- matrix_key_from_label(tumor_label)

  normal_matrix <- matrices_i[[normal_key]]
  tumor_matrix  <- matrices_i[[tumor_key]]

  if (is.null(normal_matrix)) {
    stop(sprintf("Normal matrix not found: %s", normal_key), call. = FALSE)
  }

  if (is.null(tumor_matrix)) {
    stop(sprintf("Tumor matrix not found: %s", tumor_key), call. = FALSE)
  }

  selected_probes_shared <- Reduce(
    intersect,
    list(
      selected_probes,
      rownames(normal_matrix),
      rownames(tumor_matrix)
    )
  )

  if (length(selected_probes_shared) < 2) {
    stop(
      "Fewer than two shared probes between normal and tumor matrices.",
      call. = FALSE
    )
  }

  list(
    normal = normal_matrix[selected_probes_shared, , drop = FALSE],
    tumor  = tumor_matrix[selected_probes_shared, , drop = FALSE],
    normal_key = normal_key,
    tumor_key = tumor_key,
    n_shared_probes = length(selected_probes_shared)
  )
}
