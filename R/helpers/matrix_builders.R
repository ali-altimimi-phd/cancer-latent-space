# ------------------------------------------------------------------------------
# File: matrix_builders.R
# Purpose: Build tissue/state expression matrices and predefined comparison maps
#          for the Ramaswamy global cancer analysis dataset.
# Role: Helper
# Pipeline: Analysis
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Matrix Builders for Tissue-Specific Cancer Comparisons
#'
#' These helper functions construct the empirical expression spaces used by the
#' analysis pipeline. The first function partitions a preprocessed `ExpressionSet`
#' into tissue/disease-state-specific expression matrices. The second function
#' defines the curated normal-to-tumor comparison map and retains only comparisons
#' whose required matrices are present for the chip being analyzed.
#'
#' The resulting matrix lists and comparison maps are used downstream for probe
#' filtering, pairwise normal/tumor comparisons, and complexity/entropy analyses.
#'
#' This script is intentionally dataset-specific. It assumes the phenotype data
#' contain Ramaswamy/GEO-derived fields identifying organism part and disease state.
NULL
#' Build Tissue-Specific Expression Matrices
#'
#' Splits an `ExpressionSet` into a list of expression matrices based on tissue and disease state,
#' using the `characteristics_ch1.1` (organism part) and `characteristics_ch1` (disease state)
#' fields from the phenotype data.
#'
#' @param eset An \code{ExpressionSet} object.
#'
#' @return A named list of expression matrices, one per tissue/state combination.
#' Matrix names are formatted as \code{m_<tissue>/<state>} with safe characters.
#'
#' @examples
#' \dontrun{
#'   mat_list <- build_matrix_lists_by_tissue(eset_list$hu35ksuba)
#' }
#'
#' @importFrom Biobase pData exprs
#' @importFrom glue glue
#' @importFrom purrr imap

build_matrix_lists_by_tissue <- function(eset) {
  stopifnot(inherits(eset, "ExpressionSet"))
  
  pheno <- Biobase::pData(eset)
  tissue  <- pheno$characteristics_ch1.1
  disease <- pheno$characteristics_ch1
  
  tissue_clean  <- sub("^organism part: ", "", tissue)
  disease_clean <- sub("^disease state: ", "", disease)
  
  labels <- glue::glue("{tissue_clean}/{disease_clean}")
  mat_split <- split(seq_len(ncol(eset)), labels)
  
  mat_list <- purrr::imap(mat_split, function(cols, label) {
    Biobase::exprs(eset)[, cols, drop = FALSE]
  })
  
  names(mat_list) <- paste0("m_", make.names(names(mat_list)))
  return(mat_list)
}

#' Define Available Predefined Tissue Comparisons
#'
#' Defines curated normal-to-malignant comparison pairs for the analysis dataset
#' and filters them against the matrix labels available for a given chip.
#'
#' Comparisons are specified using human-readable tissue/state labels and are
#' converted internally to the matrix-name convention used by
#' `build_matrix_lists_by_tissue()`.
NULL
#' @return A named list of comparison pairs organized by cancer group (e.g., "carcinomas").
#' Each pair is a character vector of two matrix names: control and case.
#'
#' @examples
#' \dontrun{
#'   comparisons <- define_predefined_comparisons(names(mat_list))
#' }
#'
#' @importFrom purrr map keep
define_predefined_comparisons <- function(matrix_labels) {
  predefined <- list(
    carcinomas = list(
      "BLAD/TCC" = c(
        "Bladder/normal",
        "Bladder/bladder transitional cell carcinoma"
      ),
      "BR/BRAD" = c("Breast/normal", "Breast/breast adenocarcinoma"),
      "COL/COADREAD" = c("Colon/normal", "Colon/colorectal adenocarcinoma"),
      "KID/RCC" = c("Kidney/normal", "Kidney/renal cell carcinoma"),
      "LU/LUAD" = c("Lung/normal", "Lung/lung adenocarcinoma"),
      "OV/OVAD" = c("Ovary/normal", "Ovary/ovarian adenocarcinoma"),
      "PA/PAAD" = c("Pancreas/normal", "Pancreas/pancreatic adenocarcinoma"),
      "PR/PRAD" = c("Prostate/normal", "Prostate/prostate adenocarcinoma"),
      "UT/EAC" = c("Uterus/normal", "Uterus/uterine adenocarcinoma")
    ),
    blastomas = list(
      "Brain/GBM" = c("Brain/normal", "Brain/glioblastoma"),
      "Brain/MB"  = c("Brain/normal", "Brain/medulloblastoma")
    ),
    lymphomas = list(
      "GC/FL"   = c(
        "Lymphoid Tissue/normal",
        "Lymphoid Tissue/Follicular lymphoma"
      ),
      "GC/LBCL" = c(
        "Lymphoid Tissue/normal",
        "Lymphoid Tissue/large B-cell lymphoma"
      )
    ),
    leukemias = list(
      "PB/AML"   = c("Blood/normal", "Blood/acute myeloid leukemia"),
      "PB/B-ALL" = c("Blood/normal", "Bone Marrow/B-cell ALL"),
      "PB/T-ALL" = c("Blood/normal", "Bone Marrow/T-cell ALL")
    )
  )
  
  # Filter to comparisons that exist in matrix_labels
  filtered <- purrr::map(predefined, function(group) {
    purrr::keep(group, function(labels) {
      all(paste0("m_", make.names(labels)) %in% matrix_labels)
    })
  }) |>
    purrr::keep( ~ length(.x) > 0)
  
  return(filtered)
}
