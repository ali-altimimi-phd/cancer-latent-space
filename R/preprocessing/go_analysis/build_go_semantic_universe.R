# ==============================================================================
# File: build_go_semantic_universe.R
# Purpose: Build a p-value-neutral GO semantic universe from full_chip_annotations.rds
# Role: Preprocessing helper for GO semantic annotation layer
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi; generated/refactored with ChatGPT assistance
# Created: 2026
# ==============================================================================

`%||%` <- function(x, y) if (is.null(x)) y else x

normalize_go_ontology <- function(x) {
  x <- toupper(trimws(as.character(x)))
  x <- stringr::str_replace(x, "^GO_", "")
  dplyr::case_when(
    x %in% c("BP", "BIOLOGICAL_PROCESS", "BIOLOGICAL PROCESS") ~ "BP",
    x %in% c("MF", "MOLECULAR_FUNCTION", "MOLECULAR FUNCTION") ~ "MF",
    x %in% c("CC", "CELLULAR_COMPONENT", "CELLULAR COMPONENT") ~ "CC",
    TRUE ~ NA_character_
  )
}

normalize_go_mode <- function(go_mode = NULL) {
  if (is.null(go_mode) || length(go_mode) == 0) {
    return(c("BP", "MF", "CC"))
  }

  gm <- toupper(trimws(as.character(go_mode)))
  gm <- gm[!is.na(gm) & nzchar(gm)]
  gm <- stringr::str_replace(gm, "^GO_", "")

  if (length(gm) == 0) {
    return(c("BP", "MF", "CC"))
  }

  if (any(gm %in% c("ALL", "GO", "ANY", "UNSPECIFIED", "GO_UNSPECIFIED"))) {
    return(c("BP", "MF", "CC"))
  }

  unsupported <- setdiff(gm, c("BP", "MF", "CC"))
  if (length(unsupported) > 0) {
    stop(
      "Unsupported go_mode value(s): ", paste(unsupported, collapse = ", "),
      ". Use NULL, GO_ALL, GO_BP, GO_MF, GO_CC, BP, MF, and/or CC.",
      call. = FALSE
    )
  }

  unique(gm)
}

check_go_semantic_universe_packages <- function() {
  required_packages <- c("dplyr", "tibble", "purrr", "stringr", "readr", "fs")
  missing_packages <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing_packages) > 0) {
    stop(
      "Missing required packages for GO semantic universe construction: ",
      paste(missing_packages, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

build_go_semantic_universe_one_chip <- function(chip_annotations,
                                                chip_id,
                                                go_mode = c("BP", "MF"),
                                                min_probes = c(BP = 5, MF = 5)) {
  ontologies_to_keep <- normalize_go_mode(go_mode)

  if (!is.list(chip_annotations)) {
    stop("chip_annotations for ", chip_id, " must be a list.", call. = FALSE)
  }
  if (!"annotation_table" %in% names(chip_annotations)) {
    stop("chip_annotations for ", chip_id, " is missing annotation_table.", call. = FALSE)
  }
  if (!"go_counts" %in% names(chip_annotations)) {
    stop("chip_annotations for ", chip_id, " is missing go_counts.", call. = FALSE)
  }

  annotation_table <- chip_annotations$annotation_table
  go_counts <- chip_annotations$go_counts

  required_counts <- c("go_id", "go_ontology", "n_probes")
  missing_counts <- setdiff(required_counts, names(go_counts))
  if (length(missing_counts) > 0) {
    stop("go_counts for ", chip_id, " is missing: ", paste(missing_counts, collapse = ", "), call. = FALSE)
  }

  required_annotation <- c("go_id", "go_name", "root_node")
  missing_annotation <- setdiff(required_annotation, names(annotation_table))
  if (length(missing_annotation) > 0) {
    stop(
      "annotation_table for ", chip_id, " is missing: ",
      paste(missing_annotation, collapse = ", "),
      call. = FALSE
    )
  }

  go_names <- annotation_table |>
    dplyr::filter(!is.na(.data$go_id)) |>
    dplyr::distinct(
      go_id = as.character(.data$go_id),
      go_name = as.character(.data$go_name),
      root_node = as.character(.data$root_node)
    )

  go_universe <- go_counts |>
    dplyr::mutate(
      go_id = as.character(.data$go_id),
      ontology = normalize_go_ontology(.data$go_ontology),
      n_probes = suppressWarnings(as.integer(.data$n_probes))
    ) |>
    dplyr::filter(
      !is.na(.data$go_id),
      grepl("^GO:", .data$go_id),
      !is.na(.data$ontology),
      .data$ontology %in% ontologies_to_keep,
      !is.na(.data$n_probes)
    ) |>
    dplyr::left_join(go_names, by = "go_id") |>
    dplyr::transmute(
      chip = as.character(chip_id),
      ontology = .data$ontology,
      go_id = .data$go_id,
      go_name = dplyr::coalesce(.data$go_name, .data$go_id),
      root_node = dplyr::coalesce(.data$root_node, paste0(.data$ontology, "_UNSPECIFIED")),
      n_probes = .data$n_probes
    ) |>
    dplyr::distinct()

  if (!is.null(min_probes)) {
    if (length(min_probes) == 1 && is.null(names(min_probes))) {
      min_probe_tbl <- tibble::tibble(
        ontology = ontologies_to_keep,
        min_probes_required = as.integer(min_probes)
      )
    } else {
      min_probe_tbl <- tibble::tibble(
        ontology = names(min_probes),
        min_probes_required = as.integer(unlist(min_probes))
      ) |>
        dplyr::filter(!is.na(.data$ontology), nzchar(.data$ontology))
    }

    go_universe <- go_universe |>
      dplyr::left_join(min_probe_tbl, by = "ontology") |>
      dplyr::filter(
        !is.na(.data$min_probes_required),
        .data$n_probes >= .data$min_probes_required
      ) |>
      dplyr::select(-dplyr::any_of("min_probes_required"))
  }

  go_universe
}

build_go_semantic_universe <- function(full_chip_annotations,
                                       go_mode = c("BP", "MF"),
                                       min_probes = c(BP = 5, MF = 5)) {
  check_go_semantic_universe_packages()

  if (!is.list(full_chip_annotations) || length(full_chip_annotations) == 0) {
    stop("full_chip_annotations must be a non-empty named list.", call. = FALSE)
  }

  purrr::imap_dfr(
    full_chip_annotations,
    ~ build_go_semantic_universe_one_chip(
      chip_annotations = .x,
      chip_id = .y,
      go_mode = go_mode,
      min_probes = min_probes
    )
  ) |>
    dplyr::arrange(.data$chip, .data$ontology, .data$root_node, .data$go_id)
}

# This distinct table is the unit used for semantic clustering.  Clustering is
# independent of chip and p-value; chip-specific probe counts remain in the
# universe/membership tables.
build_go_semantic_term_dim <- function(go_universe) {
  go_universe |>
    dplyr::group_by(.data$ontology, .data$go_id) |>
    dplyr::summarise(
      go_name = dplyr::first(stats::na.omit(.data$go_name)) %||% dplyr::first(.data$go_id),
      root_node = dplyr::first(stats::na.omit(.data$root_node)) %||% paste0(dplyr::first(.data$ontology), "_UNSPECIFIED"),
      n_chips = dplyr::n_distinct(.data$chip),
      total_probe_memberships = sum(.data$n_probes, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::arrange(.data$ontology, .data$root_node, .data$go_id)
}
