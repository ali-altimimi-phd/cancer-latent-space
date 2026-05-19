# ------------------------------------------------------------------------------
# File: preprocessing_config.R
# Purpose: Configuration for the global cancer preprocessing pipeline
# Role: Defines toggles, download settings, metadata corrections, chip mappings,
#       file paths, logging destinations, and ML/latent-space export settings.
# Pipeline: Preprocessing
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

# ==============================================================================
# 1. Study identifier
# ==============================================================================

# Used to define study-specific input/output directory structure.
study_name <- "global_cancer"


# ==============================================================================
# 2. Chip/platform mappings
# ==============================================================================

# Maps internal chip identifiers to GEO platform identifiers.
geo_chip_map <- list(
  hu35ksuba = "GPL98",
  hu6800    = "GPL80"
)


# ==============================================================================
# 3. Main preprocessing pipeline toggles
# ==============================================================================

# Controls preprocessing stages.
download_enabled <- FALSE
build_esets      <- FALSE
process_metadata <- FALSE
run_annotation   <- FALSE
run_go_semantic_layer <- FALSE
export_ml_inputs <- FALSE
run_ml_pca_check <- FALSE
ingest_preprocessing_annotations_to_duckdb <- TRUE
ingest_gene_set_annotations_to_duckdb <- TRUE
ingest_go_semantic_layer_to_duckdb <- TRUE


# ==============================================================================
# 4. Download settings
# ==============================================================================

# If TRUE, report download targets without downloading files.
# Use FALSE for actual downloading.
dry_run <- TRUE

# Limit number of files downloaded during testing.
# Use NULL to download all files.
download_limit <- 3

# Remote source for CEL files.
ftp_base <- "ftp://ftp.ebi.ac.uk/biostudies/fire/E-GEOD-/928/E-GEOD-68928/Files/"

# GEO accession used for metadata retrieval.
geo_accession <- "GSE68928"


# ==============================================================================
# 5. Metadata normalization fixes
# ==============================================================================

# Dataset-specific tissue-label corrections.
fix_tissue_labels <- c(
  "organism part: Kideny"         = "organism part: Kidney",
  "organism part: Lymphod Tissue" = "organism part: Lymphoid Tissue",
  "organism part: Lymphoid"       = "organism part: Lymphoid Tissue"
)

# Dataset-specific disease-label corrections.
fix_disease_labels <- c(
  "disease state: large-Bcell lymphoma"               = "disease state: large B-cell lymphoma",
  "disease state: bladder transitonal cell carcinoma" = "disease state: bladder transitional cell carcinoma"
)


# ==============================================================================
# 6. Core input, output, and logging paths
# ==============================================================================

# Raw/local input directories.
local_cel_dir <- here::here("data", study_name, "CEL")
local_geo_dir <- here::here("data", study_name, "GEO", geo_accession)

# Preprocessing logs and RData output directories.
logs_dir <- here::here("output", study_name, "logs", "preprocess")
# data_dir <- here::here("output", study_name, "RData")
data_dir <- here::here("data", study_name, "processed", "RData")

# Avoids accidental double nesting such as:
#   output/global_cancer/logs/preprocess/preprocess/...
preprocess_pipeline_logfile <- file.path(
  logs_dir,
  "preprocess_pipeline_log.txt"
)

# Core preprocessing outputs.
eset_path <- file.path(
  data_dir,
  paste0(study_name, "_eset_list.RData")
)

annotations_path <- file.path(
  data_dir,
  "annotations",
  "full_chip_annotations.rds"
)


# ==============================================================================
# 7. GO semantic annotation layer
# ==============================================================================

# Builds a p-value-neutral, analysis-eligible GO semantic clustering layer.
# This is annotation metadata, not biological inference.

go_semantic_output_dir <- file.path(
  data_dir,
  "annotations",
  "go_semantic_layer"
)

go_semantic_ontologies <- c("BP", "MF")

go_semantic_min_probes <- c(
  BP = 5,
  MF = 5
)

go_semantic_similarity_cutoff <- 0.70
go_semantic_max_terms_per_block <- 600
go_semantic_min_terms_per_ancestor_block <- 25
go_semantic_large_block_strategy <- "ancestor_single_cluster"


# ==============================================================================
# 8. ML / latent-space parameters and export output paths
# ==============================================================================

# Chips/platforms exported for downstream Python/Jupyter latent-space workflows.
# Use names(geo_chip_map) to export all configured platforms.
ml_chip_ids      <- names(geo_chip_map)
# Alternatively:
# ml_chip_ids    <- c("hu35ksuba", "hu6800")

ml_filter_method <- "variance"
ml_top_n         <- 3000

# Canonical downstream inputs for Python scripts, notebooks, and ML workflows.
ml_output_dir <- here::here(
  "data",
  study_name,
  "processed",
  "ml_inputs"
)

# PCA sanity-check and latent-related preprocessing plots.
# Currently routed to Quarto resources so they can be included directly in reports.
ml_plot_dir <- here::here(
  "quarto",
  "resources",
  "plots",
  "latent"
)


# ==============================================================================
# 10. Ensure required directories exist
# ==============================================================================

dir.create(dirname(preprocess_pipeline_logfile), recursive = TRUE, showWarnings = FALSE)

dir.create(local_cel_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(local_geo_dir, recursive = TRUE, showWarnings = FALSE)

dir.create(dirname(eset_path), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(annotations_path), recursive = TRUE, showWarnings = FALSE)

dir.create(ml_output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ml_plot_dir, recursive = TRUE, showWarnings = FALSE)

dir.create(go_semantic_output_dir, recursive = TRUE, showWarnings = FALSE)