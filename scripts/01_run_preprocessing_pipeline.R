# ------------------------------------------------------------------------------
# File: 01_run_preprocessing_pipeline.R
# Purpose: Execute the preprocessing pipeline
# Role: Top-level preprocessing runner
# Pipeline: Preprocessing
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Run Preprocessing Pipeline for Global Cancer Microarray Data
#'
#' @description
#' Executes the preprocessing workflow for Affymetrix CEL files, including
#' downloading, normalization, GEO metadata attachment/cleaning, probe annotation,
#' and export of machine-learning-ready expression/metadata files.
#'
#' @details
#' The machine-learning export stage prepares files consumed by downstream
#' Python/Jupyter workflows. It does not train the VAE and does not perform
#' latent-space analysis.
#'
#' @return No return value. Writes output to disk.
#' @export
run_preprocessing_pipeline <- function() {
  # ---- Config ----
  source(here::here("R/config/global_cancer/preprocessing_config.R"))

  # ---- Logging ----
  source(here::here("R/helpers/pipeline_logger.R"))
  logger <- start_log(preprocess_pipeline_logfile)
  logger$log("🚀 Starting Preprocessing Pipeline...", section = "PIPELINE")

  # ---- Step 1: Download CEL files (optional) ----
  if (download_enabled) {
    logger$log("📥 Downloading CEL files via FTP...", section = "DOWNLOAD")
    source(here::here("R/preprocessing/download_cel_files.R"))
    downloaded <- download_cel_files(
      ftp_base = ftp_base,
      local_cel_dir = local_cel_dir,
      download_limit = download_limit,
      dry_run = dry_run,
      logger = logger
    )
    logger$log(glue::glue("✅ FTP download complete. {length(downloaded)} files processed."), section = "DOWNLOAD")
  } else {
    logger$log("⏭️ FTP download skipped (download_enabled = FALSE).", section = "DOWNLOAD")
  }

  
  # ---- Step 2: Build ExpressionSets from CEL files (optional) ----
  if (isTRUE(build_esets)) {
    logger$log("🧬 Building ExpressionSet objects from CEL files...", section = "ESET")
    source(here::here("R/preprocessing/build_expression_sets.R"))
    eset_list <- build_expression_sets(
      cel_dir = local_cel_dir,
      log_dir = logs_dir,
      logger = logger
    )
  } else {
    logger$log("⏭️ ExpressionSet building skipped (build_esets = FALSE).", section = "ESET")
  }

  if (!build_esets) {
    logger$log(glue::glue("📦 Loading ExpressionSets from {eset_path}"), section = "ESET")
    load(eset_path)
  }

  if (!exists("eset_list")) {
    stop("eset_list was not created or loaded. Check build_esets and eset_path.", call. = FALSE)
  }

  
  # ---- Step 3: Attach and Process GEO Metadata ----
  if (isTRUE(process_metadata)) {
    logger$log("🧬 Processing GEO metadata: attach → clean → label → align...", section = "METADATA")
    
    source(here::here("R/preprocessing/geo_metadata/attach_geo_metadata.R"))
    eset_list <- attach_geo_metadata(
      eset_list = eset_list,
      geo_dir = local_geo_dir,
      logger = logger
    )

    source(here::here("R/preprocessing/geo_metadata/clean_and_label_metadata.R"), local = TRUE)
    eset_list <- clean_and_label_metadata(
      eset_list = eset_list,
      tissue_fixes = fix_tissue_labels,
      disease_fixes = fix_disease_labels,
      logger = logger
    )

    source(here::here("R/preprocessing/geo_metadata/repair_eset_metadata_alignment.R"), local = TRUE)
    
    eset_list <- repair_eset_metadata_alignment(
      eset_list = eset_list,
      logger = logger
    )
    
    logger$log("✅ GEO metadata processing complete.", section = "METADATA")
  } else {
    logger$log("⏭️ GEO metadata processing skipped (process_metadata = FALSE).", section = "METADATA")
  }

  
  # ---- Step 4: Annotate all chip probes ----
  if (isTRUE(run_annotation)) {
    logger$log("🔎 Annotating all probes per chip...", section = "ANNOTATION")
    source(here::here("R/preprocessing/wrappers/run_annotate_chip_probes.R"), local = TRUE)

    annotations <- run_annotate_chip_probes(
      eset_list        = eset_list,
      chip_ids         = names(geo_chip_map),
      annotations_path = annotations_path
    )

    logger$log("✅ Probe annotation complete.", section = "ANNOTATION")
  } else {
    logger$log("⏭️ Annotation step skipped (run_annotation = FALSE).", section = "ANNOTATION")
  }

  # ---- Step 4B: Build GO semantic annotation layer ----
  if (isTRUE(run_go_semantic_layer)) {
    logger$log(
      "🧠 Building p-value-neutral GO semantic annotation layer...",
      section = "GO_SEMANTIC"
    )
    
    source(here::here("R/wrappers/run_go_semantic_layer.R"), local = TRUE)
    
    go_semantic_layer <- run_go_semantic_layer(
      annotation_path = annotations_path,
      output_dir = go_semantic_output_dir,
      go_mode = go_semantic_ontologies,
      min_probes = go_semantic_min_probes,
      similarity_cutoff = go_semantic_similarity_cutoff,
      max_terms_per_block = go_semantic_max_terms_per_block,
      min_terms_per_ancestor_block = go_semantic_min_terms_per_ancestor_block,
      large_block_strategy = go_semantic_large_block_strategy,
      logger = logger
    )
    
    logger$log(
      glue::glue(
        "✅ GO semantic layer complete: {nrow(go_semantic_layer$go_semantic_cluster_membership)} term memberships."
      ),
      section = "GO_SEMANTIC"
    )
  } else {
    logger$log(
      "⏭️ GO semantic layer skipped (run_go_semantic_layer = FALSE).",
      section = "GO_SEMANTIC"
    )
  }
  
  
  # ---- Step 5: Save ExpressionSets ----
  save(eset_list, file = eset_path)
  logger$log(glue::glue("💾 ExpressionSets saved to {basename(eset_path)}"), section = "SAVE")

  
  # ---- Step 6: Export ML / latent-space inputs ----
  if (isTRUE(export_ml_inputs)) {
    logger$log(
      "🧠 Exporting machine-learning / latent-space input files...",
      section = "ML_EXPORT"
    )
    
    source(here::here("R/preprocessing/export_ml_inputs.R"), local = TRUE)
    
    for (chip_id in ml_chip_ids) {
      logger$log(
        sprintf("🧬 Exporting ML inputs for chip: %s", chip_id),
        section = "ML_EXPORT"
      )
      
      export_ml_inputs_from_eset(
        eset_list = eset_list,
        output_dir = ml_output_dir,
        plot_dir = ml_plot_dir,
        chip_id = chip_id,
        top_n = ml_top_n,
        filter_method = ml_filter_method,
        run_pca_check = run_ml_pca_check,
        logger = logger
      )
    }
    
    logger$log("✅ ML input export complete.", section = "ML_EXPORT")
  } else {
    logger$log(
      "⏭️ ML input export skipped (export_ml_inputs = FALSE).",
      section = "ML_EXPORT"
    )
  }

  # ---- Step 7: DuckDB preprocessing annotation ingestion ----------------------
  
  preprocessing_duckdb_ingestion_results <- NULL
  
  if (isTRUE(ingest_preprocessing_annotations_to_duckdb)) {
    
    logger$timed("DuckDB preprocessing annotation ingestion", {
      
      logger$log(
        "🦆 Ingesting preprocessing annotation artifacts into DuckDB...",
        section = "DUCKDB"
      )
      
      source(
        here::here(
          "R",
          "warehouse",
          "ingest",
          "ingest_preprocessing_annotations_to_duckdb.R"
        )
      )
      
      preprocessing_duckdb_ingestion_results <-
        ingest_preprocessing_annotations_to_duckdb(
          study_name = study_name,
          annotation_path = annotations_path,
          go_semantic_dir = go_semantic_output_dir,
          ingest_gene_set_annotations = ingest_gene_set_annotations_to_duckdb,
          ingest_go_semantic_layer = ingest_go_semantic_layer_to_duckdb,
          logger = logger
        )
      
      logger$log(
        "✅ Preprocessing annotation DuckDB ingestion completed.",
        section = "DUCKDB"
      )
    })
    
  } else {
    
    logger$log(
      "⏭️ Preprocessing annotation DuckDB ingestion skipped.",
      section = "DUCKDB"
    )
  }

    
  # ---- Step 8: End ----------------------------------------------------------
  logger$log("🏁 Preprocessing pipeline completed successfully.", section = "PIPELINE")

  invisible(TRUE)
}

if (sys.nframe() == 0) {
  run_preprocessing_pipeline()
}
