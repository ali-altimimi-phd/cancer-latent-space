# ------------------------------------------------------------------------------
# File: run_pairwise_analysis.R
# Purpose: Execute pairwise biological gene-set comparisons from precomputed
#          admissible gene-set inputs
# Role: Internal biological interpretation helper
# Pipeline: Biological Interpretation
# Project: Global Cancer Structural Inference Framework
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Build Admissible Gene-Set Inputs for Pairwise Biological Engines
#'
#' Precomputes comparison-specific gene-set inputs for a single
#' chip x filter-regime x gene-set-mode context.
#'
#' This function centralizes the expensive admissibility work that was formerly
#' repeated inside each engine run. It intersects gene-set probe memberships with
#' the comparison-specific filtered matrices once, records skipped terms once,
#' and returns engine-ready inputs that can be shared by complexity, entropy, and
#' future structural engines.
#'
#' Two probe-count thresholds are enforced:
#'
#' 1. Engine viability threshold: comparisons with fewer than two probes in one
#'    or both matrices are skipped because structural descriptors are unstable or
#'    undefined.
#' 2. Biological admissibility threshold: gene sets are retained only when the
#'    number of usable probes after intersection is at least `min_probes`.
#'
#' @param comparison_list List of comparison metadata and matrices, usually from
#'   build_comparison_input_list().
#' @param gene_sets Character vector of biological gene-set identifiers to test,
#'   or the sentinel value "FULL" for whole-matrix comparisons.
#' @param gene_set_indices Named list of gene-set probe indices by chip.
#' @param gene_set_mode Gene-set mode: GO_BP, GO_MF, KEGG, MSIGDB, or FULL.
#' @param filter_regime Concrete filtered-probe regime label, such as limma or
#'   variance_top3000.
#' @param min_probes Minimum number of usable probes required for a gene set
#'   within a given comparison.
#' @param logger Optional detail logger supplied by the wrapper.
#' @param verbose Logical. Whether to print progress messages when logger is NULL.
#'
#' @return A named list with `inputs` and `skipped` components. The `inputs`
#'   component is a comparison-nested list of engine-ready gene-set inputs.
build_admissible_gene_set_inputs <- function(
    comparison_list,
    gene_sets,
    gene_set_indices = NULL,
    gene_set_mode,
    filter_regime,
    min_probes,
    logger = NULL,
    verbose = TRUE
) {
  log_detail <- function(msg) {
    if (!is.null(logger)) {
      logger$log(msg, section = "BIO_PAIRWISE_DETAIL")
    } else if (isTRUE(verbose)) {
      message(msg)
    }
  }

  if (!(is.character(gene_sets) && length(gene_sets) >= 1)) {
    stop(
      "gene_sets must be a non-empty character vector of biological gene-set IDs.",
      call. = FALSE
    )
  }

  if (is.null(gene_set_indices)) {
    stop(
      "gene_set_indices must be supplied for biological interpretation.",
      call. = FALSE
    )
  }

  if (missing(gene_set_mode) || is.null(gene_set_mode)) {
    stop("gene_set_mode is required.", call. = FALSE)
  }

  if (missing(filter_regime) || is.null(filter_regime)) {
    stop("filter_regime is required.", call. = FALSE)
  }

  if (missing(min_probes) || is.null(min_probes) || is.na(min_probes)) {
    stop("min_probes is required.", call. = FALSE)
  }

  admissible_inputs <- list()
  skipped <- list()

  for (cmp in comparison_list) {
    cmp_name <- cmp$name
    group    <- cmp$group
    chip     <- cmp$chip
    mat1     <- cmp$m1
    mat2     <- cmp$m2

    if (nrow(mat1) < 2 || nrow(mat2) < 2) {
      reason <- paste0(
        "<2 probes in one or both comparison matrices; ",
        "engine cannot compute stable pairwise descriptor"
      )

      log_detail(glue::glue(
        "⚠️ Skipping comparison during admissibility build: {cmp_name} | chip={chip} | reason={reason}"
      ))

      skipped[[paste(cmp_name, "comparison_level", sep = "::")]] <- tibble::tibble(
        comparison = cmp_name,
        group = group,
        chip = chip,
        filter_regime = filter_regime,
        gene_set_mode = gene_set_mode,
        gene_set_name = NA_character_,
        n_gene_set_probes = NA_integer_,
        n_usable_probes = NA_integer_,
        reason = reason
      )

      next
    }

    log_detail(glue::glue(
      "🧮 Building admissible gene-set inputs: comparison={cmp_name}; chip={chip}; filter_regime={filter_regime}; gene_set_mode={gene_set_mode}"
    ))

    cmp_inputs <- list()

    for (gs in gene_sets) {
      result_id <- make.names(gs)

      if (identical(gs, "FULL") || identical(gene_set_mode, "FULL")) {
        probes <- rownames(mat1)
      } else {
        if (is.null(gene_set_indices[[chip]])) {
          stop("Gene-set index missing for chip: ", chip, call. = FALSE)
        }

        probes <- get_probes_for_set(gs, gene_set_indices[[chip]])
      }

      engine_input <- prepare_gene_set_engine_inputs(
        mat_normal = mat1,
        mat_tumor = mat2,
        selected_probes = probes
      )

      n_gene_set_probes <- length(unique(probes))
      n_usable_probes <- length(engine_input$selected_probes)

      if (n_usable_probes < min_probes) {
        reason <- sprintf(
          "usable gene-set probes below biological admissibility threshold: %s < %s",
          n_usable_probes,
          min_probes
        )

        log_detail(glue::glue(
          "⚠️ Skipped gene set during admissibility build: comparison={cmp_name}; chip={chip}; mode={gene_set_mode}; gene_set={gs}; usable_probes={n_usable_probes}; min_probes={min_probes}"
        ))

        skipped[[paste(cmp_name, result_id, sep = "::")]] <- tibble::tibble(
          comparison = cmp_name,
          group = group,
          chip = chip,
          filter_regime = filter_regime,
          gene_set_mode = gene_set_mode,
          gene_set_name = gs,
          n_gene_set_probes = n_gene_set_probes,
          n_usable_probes = n_usable_probes,
          reason = reason
        )

        next
      }

      cmp_inputs[[result_id]] <- list(
        comparison = cmp_name,
        group = group,
        chip = chip,
        filter_regime = filter_regime,
        gene_set_mode = gene_set_mode,
        gene_set_name = gs,
        result_id = result_id,
        n_gene_set_probes = n_gene_set_probes,
        n_usable_probes = n_usable_probes,
        matrices_i = engine_input$matrices_i,
        comparison_labels = engine_input$comparison_labels,
        selected_probes = engine_input$selected_probes
      )
    }

    admissible_inputs[[cmp_name]] <- cmp_inputs

    log_detail(glue::glue(
      "✅ Admissibility built: comparison={cmp_name}; chip={chip}; retained_gene_sets={length(cmp_inputs)}"
    ))
  }

  skip_df <- if (length(skipped) > 0) {
    dplyr::bind_rows(skipped)
  } else {
    tibble::tibble()
  }

  list(
    inputs = admissible_inputs,
    skipped = skip_df
  )
}

#' Execute Pairwise Biological Comparisons from Admissible Gene-Set Inputs
#'
#' Internal helper used by `run_biological_pairwise_comparisons()`.
#'
#' This function applies one analytic engine to a precomputed set of admissible
#' gene-set inputs. It does not perform gene-set admissibility filtering. That
#' work is handled upstream by `build_admissible_gene_set_inputs()` so that
#' multiple engines can reuse the same comparison-specific intersections.
#'
#' @param admissible_inputs Comparison-nested list returned in the `inputs`
#'   component of `build_admissible_gene_set_inputs()`.
#' @param skipped Data frame of skipped comparison/gene-set records returned in
#'   the `skipped` component of `build_admissible_gene_set_inputs()`.
#' @param engine Engine to use: "complexity" or "entropy".
#' @param biological_resampling_seed Optional random seed used by biological
#'   permutation/bootstrap routines.
#' @param run_complexity_permutation Logical; whether to run complexity
#'   permutation testing.
#' @param complexity_n_perm Number of complexity permutations.
#' @param complexity_permutation_metric Complexity metric for permutation.
#' @param complexity_permutation_unit Resampling unit for complexity permutation.
#' @param run_complexity_bootstrap Logical; whether to run complexity bootstrap.
#' @param complexity_n_boot Number of complexity bootstrap replicates.
#' @param complexity_bootstrap_metric Complexity metric for bootstrap.
#' @param complexity_bootstrap_unit Resampling unit for complexity bootstrap.
#' @param complexity_covariance_space Covariance orientation for complexity.
#' @param run_entropy_permutation Logical; whether to run entropy permutation.
#' @param entropy_n_perm Number of entropy permutations.
#' @param entropy_permutation_metric Entropy metric for permutation.
#' @param entropy_permutation_unit Resampling unit for entropy permutation.
#' @param run_entropy_bootstrap Logical; whether to run entropy bootstrap.
#' @param entropy_n_boot Number of entropy bootstrap replicates.
#' @param entropy_bootstrap_metric Entropy metric for bootstrap.
#' @param entropy_bootstrap_unit Resampling unit for entropy bootstrap.
#' @param entropy_covariance_space Covariance orientation for entropy.
#' @param logger Optional detail logger supplied by the wrapper.
#' @param verbose Logical. Whether to print progress messages when logger is NULL.
#'
#' @return A named list with result and skip components.
run_pairwise_analysis <- function(
    admissible_inputs,
    skipped = tibble::tibble(),
    engine = c("complexity", "entropy"),

    biological_resampling_seed = NULL,

    run_complexity_permutation = FALSE,
    complexity_n_perm = 0,
    complexity_permutation_metric = "kappa",
    complexity_permutation_unit = "sample_label",
    run_complexity_bootstrap = FALSE,
    complexity_n_boot = 0,
    complexity_bootstrap_metric = "kappa",
    complexity_bootstrap_unit = "sample",
    complexity_covariance_space = "sample",

    run_entropy_permutation = FALSE,
    entropy_n_perm = 0,
    entropy_permutation_metric = "spectral",
    entropy_permutation_unit = "sample_label",
    run_entropy_bootstrap = FALSE,
    entropy_n_boot = 0,
    entropy_bootstrap_metric = "spectral",
    entropy_bootstrap_unit = "sample",
    entropy_covariance_space = "sample",

    logger = NULL,
    verbose = TRUE
) {
  engine <- match.arg(engine)

  log_detail <- function(msg) {
    if (!is.null(logger)) {
      logger$log(msg, section = "BIO_PAIRWISE_DETAIL")
    } else if (isTRUE(verbose)) {
      message(msg)
    }
  }

  if (!is.list(admissible_inputs)) {
    stop("admissible_inputs must be a comparison-nested list.", call. = FALSE)
  }

  results <- list()

  for (cmp_name in names(admissible_inputs)) {
    cmp_inputs <- admissible_inputs[[cmp_name]]
    cmp_result <- list()

    if (length(cmp_inputs) == 0) {
      results[[cmp_name]] <- cmp_result
      next
    }

    for (result_id in names(cmp_inputs)) {
      input <- cmp_inputs[[result_id]]

      log_detail(glue::glue(
        "🔬 Running {engine}: comparison={input$comparison}; chip={input$chip}; filter_regime={input$filter_regime}; gene_set_mode={input$gene_set_mode}; gene_set={input$gene_set_name}"
      ))

      result <- switch(
        engine,

        complexity = compare_pair_complexity(
          matrices_i = input$matrices_i,
          comparison_labels = input$comparison_labels,
          selected_probes = input$selected_probes,
          comparison = input$comparison,
          group = input$group,
          chip = input$chip,
          filter_regime = input$filter_regime,
          gene_set_mode = input$gene_set_mode,
          gene_set_name = input$gene_set_name,
          run_permutation = run_complexity_permutation,
          n_perm = complexity_n_perm,
          permutation_metric = complexity_permutation_metric,
          permutation_unit = complexity_permutation_unit,
          run_bootstrap = run_complexity_bootstrap,
          n_boot = complexity_n_boot,
          bootstrap_metric = complexity_bootstrap_metric,
          bootstrap_unit = complexity_bootstrap_unit,
          covariance_space = complexity_covariance_space,
          seed = biological_resampling_seed
        ),

        entropy = compare_pair_entropy(
          matrices_i = input$matrices_i,
          comparison_labels = input$comparison_labels,
          selected_probes = input$selected_probes,
          comparison = input$comparison,
          group = input$group,
          chip = input$chip,
          filter_regime = input$filter_regime,
          gene_set_mode = input$gene_set_mode,
          gene_set_name = input$gene_set_name,
          run_permutation = run_entropy_permutation,
          n_perm = entropy_n_perm,
          permutation_metric = entropy_permutation_metric,
          permutation_unit = entropy_permutation_unit,
          run_bootstrap = run_entropy_bootstrap,
          n_boot = entropy_n_boot,
          bootstrap_metric = entropy_bootstrap_metric,
          bootstrap_unit = entropy_bootstrap_unit,
          covariance_space = entropy_covariance_space,
          seed = biological_resampling_seed
        )
      )

      cmp_result[[result_id]] <- result

      log_detail(glue::glue(
        "✅ Completed gene set: comparison={input$comparison}; chip={input$chip}; mode={input$gene_set_mode}; gene_set={input$gene_set_name}; engine={engine}; usable_probes={input$n_usable_probes}"
      ))
    }

    results[[cmp_name]] <- cmp_result
  }

  if (nrow(skipped) > 0 && !"engine" %in% names(skipped)) {
    skipped$engine <- engine
    skipped <- skipped[, c(
      "comparison", "group", "chip", "filter_regime", "gene_set_mode",
      "gene_set_name", "engine", "n_gene_set_probes", "n_usable_probes",
      "reason"
    )]
  }

  list(
    results = results,
    skipped = skipped
  )
}
