# ------------------------------------------------------------------------------
# File: annotate_chip_probes.R
# Purpose: Annotate Affymetrix probe identifiers with gene, pathway, and gene
#   set metadata for downstream complexity and entropy analysis
# Role: Probe annotation utility
# Pipeline: Preprocessing
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Annotate the full set of probe IDs for a chip platform
#'
#' Uses Bioconductor annotation resources to attach gene symbols, GO terms,
#' KEGG pathways, and MSigDB H-collection matches to a vector of probe IDs.
#'
#' @param probe_ids Character vector of probe identifiers
#' @param chip_id Character scalar giving the chip platform name
#'   (e.g., `"hu35ksuba"` or `"hu6800"`)
#'
#' @return A list containing merged annotation tables and summary counts
annotate_chip_probes <- function(probe_ids, chip_id) {
  # ---- Chip Database Map ----
  chip_db_map <- list(
    hu35ksuba = "hu35ksuba.db",
    hu6800    = "hu6800.db"
  )
  
  if (!chip_id %in% names(chip_db_map)) {
    stop("❌ Unknown chip ID: ", chip_id)
  }
  
  # ---- Dependency Checks ----
  required_pkgs <- c(
    "AnnotationDbi",
    "dplyr",
    "GOfuncR",
    "clusterProfiler",
    "msigdbr",
    chip_db_map[[chip_id]]
  )
  
  missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0) {
    stop(glue::glue("❌ Required package(s) not installed: {paste(missing_pkgs, collapse = ', ')}"))
  }
  
  db_pkg <- chip_db_map[[chip_id]]
  db <- get(db_pkg, envir = asNamespace(db_pkg))
  
  # ---- GO Annotations (MF + BP) ----
  go_annot_raw <- AnnotationDbi::select(
    db,
    keys = probe_ids,
    columns = c("GO", "ONTOLOGY"),
    keytype = "PROBEID"
  ) |>
    dplyr::filter(ONTOLOGY %in% c("MF", "BP"), !is.na(GO)) |>
    dplyr::distinct(PROBEID, GO, ONTOLOGY) |>
    dplyr::rename(go_id = GO, go_ontology = ONTOLOGY)
  
  go_names <- GOfuncR::get_names(unique(go_annot_raw$go_id))
  go_annot_named <- dplyr::left_join(go_annot_raw, go_names, by = c("go_id" = "go_id"))
  
  # ---- KEGG Annotations ----
  kegg_annot <- AnnotationDbi::select(
    db,
    keys = probe_ids,
    columns = "PATH",
    keytype = "PROBEID"
  ) |>
    dplyr::filter(!is.na(PATH)) |>
    dplyr::distinct(PROBEID, PATH)
  
  kegg_ids <- unique(kegg_annot$PATH)
  
  kegg_pathways <- clusterProfiler::download_KEGG("hsa")$KEGGPATHID2NAME |>
    dplyr::mutate(PATH = gsub("^hsa", "", from)) |>
    dplyr::filter(PATH %in% kegg_ids) |>
    dplyr::rename(path_name = to) |>
    dplyr::distinct(PATH, path_name)
  
  kegg_annot_named <- dplyr::left_join(kegg_annot, kegg_pathways, by = "PATH")
  
  # ---- Gene Symbol Mapping ----
  gene_annot <- AnnotationDbi::select(
    db,
    keys = probe_ids,
    columns = "SYMBOL",
    keytype = "PROBEID"
  ) |>
    dplyr::filter(!is.na(SYMBOL)) |>
    dplyr::distinct(PROBEID, SYMBOL)
  
  # ---- MSigDB H Collection Hits ----
  msigdb_h <- msigdbr::msigdbr(species = "Homo sapiens", collection = "H")
  gene_symbols <- unique(gene_annot$SYMBOL)
  
  msig_gene_hits <- msigdb_h |>
    dplyr::filter(gene_symbol %in% gene_symbols) |>
    dplyr::distinct(gs_name, gene_symbol)
  
  msig_annotated <- dplyr::left_join(
    msig_gene_hits,
    gene_annot,
    by = c("gene_symbol" = "SYMBOL")
  ) |>
    dplyr::filter(!is.na(PROBEID)) |>
    dplyr::distinct(PROBEID, gs_name)
  
  # ---- Final Merge ----
  merged <- gene_annot |>
    dplyr::left_join(go_annot_named, by = "PROBEID", relationship = "many-to-many") |>
    dplyr::left_join(kegg_annot_named, by = "PROBEID", relationship = "many-to-many") |>
    dplyr::left_join(msig_annotated, by = "PROBEID", relationship = "many-to-many")
  
  msig_counts <- msig_annotated |>
    dplyr::count(gs_name, name = "n_probes")
  
  # ---- Return Structured Result ----
  list(
    annotation_table = merged,
    go_counts        = dplyr::count(go_annot_named, go_id, go_ontology, name = "n_probes"),
    kegg_counts      = dplyr::count(kegg_annot_named, PATH, name = "n_probes"),
    msig_counts      = msig_counts
  )
}