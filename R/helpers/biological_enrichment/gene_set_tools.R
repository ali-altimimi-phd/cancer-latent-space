# gene_set_utils.R
# -------------------------------------------------------------
# Gene Set Utility Functions for Global Cancer Microarray Pipeline
#
# Adds support for MSigDB Hallmark gene sets while preserving GO and KEGG logic.
# -------------------------------------------------------------

#' Compute dynamic gene set thresholds by chip and source (GO, KEGG, MSIGDB)
#'
#' @param annotations A named list of chip annotations.
#' @param output_log_file Optional log file path.
#' @return A nested list: thresholds[[chip]][[source]]
compute_dynamic_gene_set_thresholds <- function(annotations, output_log_file = NULL) {
  thresholds <- list()
  lines <- c("Gene Set Filtering Thresholds\n=============================\n")
  
  for (chip in names(annotations)) {
    ann <- annotations[[chip]]
    chip_thresholds <- list()
    
    for (source in c("GO", "KEGG", "MSIGDB")) {
      count_tbl <- switch(source,
                          "GO"     = ann$go_counts,
                          "KEGG"   = ann$kegg_counts,
                          "MSIGDB" = ann$msig_counts,
                          stop("Unknown source: ", source)
      )
      
      n_probes <- count_tbl$n_probes
      mean_val <- mean(n_probes, na.rm = TRUE)
      q3_val   <- quantile(n_probes, 0.75, na.rm = TRUE)
      min_probes <- ceiling(max(mean_val, q3_val))
      
      chip_thresholds[[source]] <- list(
        quantile_cutoff = 0.75,
        min_probes = min_probes
      )
      
      lines <- c(lines, glue::glue(
        "{chip} | {source}: mean = {round(mean_val, 2)}, Q3 = {q3_val}, → min_probes = {min_probes}"
      ))
    }
    
    thresholds[[chip]] <- chip_thresholds
  }
  
  if (!is.null(output_log_file)) {
    dir.create(dirname(output_log_file), showWarnings = FALSE, recursive = TRUE)
    writeLines(lines, output_log_file)
  }
  
  return(thresholds)
}

#' Adaptively filter gene sets
#'
#' @param annotation A chip-specific annotation list.
#' @param source One of "GO", "KEGG", "MSIGDB".
#' @param quantile_cutoff Quantile threshold (default = 0.75)
#' @param min_probes Minimum number of probes (default = 5)
#' @param ontology Optional GO sub-ontology filter (e.g., "BP" or "MF")
#' @return A character vector of gene set IDs.
adaptive_gene_set_filter <- function(annotation,
                                     source = c("GO", "KEGG", "MSIGDB"),
                                     quantile_cutoff = 0.75,
                                     min_probes = 5,
                                     ontology = NULL) {
  source <- match.arg(source)
  
  if (source == "GO") {
    count_tbl <- annotation$go_counts
    id_col <- "go_id"
    if (!is.null(ontology)) {
      count_tbl <- count_tbl |>
        dplyr::filter(go_ontology == ontology)
    }
  } else if (source == "KEGG") {
    count_tbl <- annotation$kegg_counts
    id_col <- "PATH"
  } else if (source == "MSIGDB") {
    count_tbl <- annotation$msig_counts
    id_col <- "gs_name"
  }
  
  gene_sets <- count_tbl |>
    dplyr::filter(
      n_probes >= quantile(n_probes, quantile_cutoff, na.rm = TRUE),
      n_probes >= min_probes
    ) |>
    dplyr::pull(.data[[id_col]])
  
  # message("Number of gene sets: ", length(gene_sets))
  return(gene_sets)
}

#' Attach readable gene set names and flags
#'
#' @param df Data frame with `gene_set` column.
#' @param annotations Named list of annotation lists.
#' @return Data frame with new columns: gene_set_normalized, gene_set_name, kegg_cancer, msig_hallmark
attach_gene_set_names <- function(df, annotations) {
  df <- df |>
    dplyr::mutate(
      gene_set_normalized = dplyr::case_when(
        stringr::str_detect(gene_set, "^GO\\.") ~ stringr::str_replace(gene_set, "^GO\\.", "GO:"),
        stringr::str_detect(gene_set, "^X\\d{5}$") ~ stringr::str_replace(gene_set, "^X", ""),
        TRUE ~ gene_set
      )
    )
  
  go_lookup <- dplyr::bind_rows(
    annotations$hu35ksuba$annotation_table,
    annotations$hu6800$annotation_table
  ) |>
    dplyr::filter(!is.na(go_id), !is.na(go_name)) |>
    dplyr::distinct(go_id, go_name, go_ontology) |>
    dplyr::mutate(go_id = stringr::str_replace_all(go_id, "\\.", ":"))
  
  kegg_lookup <- dplyr::bind_rows(
    annotations$hu35ksuba$annotation_table,
    annotations$hu6800$annotation_table
  ) |>
    dplyr::filter(!is.na(PATH), !is.na(path_name)) |>
    dplyr::distinct(PATH, path_name)
  
  msig_lookup <- dplyr::bind_rows(
    annotations$hu35ksuba$annotation_table,
    annotations$hu6800$annotation_table
  ) |>
    dplyr::filter(!is.na(gs_name)) |>
    dplyr::distinct(gs_name) |>
    dplyr::mutate(msig_hallmark = TRUE)
  
  df <- df |>
    dplyr::left_join(go_lookup,   by = c("gene_set_normalized" = "go_id")) |>
    dplyr::left_join(kegg_lookup, by = c("gene_set_normalized" = "PATH")) |>
    dplyr::left_join(msig_lookup, by = c("gene_set_normalized" = "gs_name")) |>
    dplyr::mutate(
      gene_set_name = dplyr::coalesce(go_name, path_name, gene_set_normalized),
      kegg_cancer   = stringr::str_detect(tolower(path_name), "cancer"),
      msig_hallmark = dplyr::coalesce(msig_hallmark, FALSE),
      go_ontology   = dplyr::coalesce(go_ontology, NA_character_)  # propagate MF/BP where relevant
    )
  
  return(df)
}

#' Build fast probe lookup index for GO, KEGG, and MSigDB gene sets
#'
#' @param annotation Annotation list for one chip.
#' @return Named list of probe vectors indexed by gene-set ID.
build_gene_set_probe_index <- function(annotation) {
  tab <- annotation$annotation_table
  
  list(
    GO = tab |>
      dplyr::filter(!is.na(go_id), !is.na(PROBEID)) |>
      dplyr::distinct(go_id, PROBEID) |>
      dplyr::group_by(go_id) |>
      dplyr::summarise(probes = list(unique(PROBEID)), .groups = "drop") |>
      tibble::deframe(),
    
    KEGG = tab |>
      dplyr::filter(!is.na(PATH), !is.na(PROBEID)) |>
      dplyr::distinct(PATH, PROBEID) |>
      dplyr::group_by(PATH) |>
      dplyr::summarise(probes = list(unique(PROBEID)), .groups = "drop") |>
      tibble::deframe(),
    
    MSIGDB = tab |>
      dplyr::filter(!is.na(gs_name), !is.na(PROBEID)) |>
      dplyr::distinct(gs_name, PROBEID) |>
      dplyr::group_by(gs_name) |>
      dplyr::summarise(probes = list(unique(PROBEID)), .groups = "drop") |>
      tibble::deframe()
  )
}

#' Get probe IDs for a gene set using precomputed index
#'
#' @param gene_set_id Gene set identifier.
#' @param gene_set_index Precomputed index from build_gene_set_probe_index().
#' @return Character vector of probe IDs.
get_probes_for_set <- function(gene_set_id, gene_set_index) {
  if (startsWith(gene_set_id, "GO:")) {
    probes <- gene_set_index$GO[[gene_set_id]]
  } else if (stringr::str_detect(gene_set_id, "^[0-9]{4,5}$")) {
    probes <- gene_set_index$KEGG[[gene_set_id]]
  } else if (startsWith(gene_set_id, "HALLMARK_")) {
    probes <- gene_set_index$MSIGDB[[gene_set_id]]
  } else {
    warning(glue::glue("⚠️ Unknown gene set prefix: {gene_set_id}"))
    return(character(0))
  }
  
  unique(na.omit(probes))
}