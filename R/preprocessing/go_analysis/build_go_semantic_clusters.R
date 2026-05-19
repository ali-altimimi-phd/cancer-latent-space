# ==============================================================================
# File: build_go_semantic_clusters.R
# Purpose: Compute p-value-neutral GO semantic clusters from a GO term universe
# Role: Preprocessing helper for GO semantic annotation layer
# Project: Cancer Complexity Analysis
# Author: Ali M. Al-Timimi; generated/refactored with ChatGPT assistance
# Created: 2026
# ============================================================================== 

# ---- Package checks -----------------------------------------------------------

check_go_semantic_cluster_packages <- function() {
  required_packages <- c(
    "dplyr", "tibble", "purrr", "stringr",
    "AnnotationDbi", "GO.db", "GOSemSim", "org.Hs.eg.db"
  )

  missing_packages <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]

  if (length(missing_packages) > 0) {
    stop(
      "Missing required packages for GO semantic clustering: ",
      paste(missing_packages, collapse = ", "),
      call. = FALSE
    )
  }

  invisible(TRUE)
}

# ---- String helpers -----------------------------------------------------------

sanitize_cluster_part <- function(x) {
  x <- as.character(x)
  x <- stringr::str_replace_all(x, "[^A-Za-z0-9]+", "_")
  x <- stringr::str_replace_all(x, "_+", "_")
  x <- stringr::str_replace_all(x, "^_|_$", "")
  ifelse(is.na(x) | !nzchar(x), "UNSPECIFIED", x)
}

# ---- Semantic data ------------------------------------------------------------

make_semdata_by_ontology <- function(ontologies) {
  stats::setNames(
    lapply(ontologies, function(ont) {
      message("Preparing GOSemSim semantic data for ontology: ", ont)
      GOSemSim::godata(annoDb = "org.Hs.eg.db", ont = ont)
    }),
    ontologies
  )
}

get_go_ancestor_map <- function(ontology) {
  ontology <- toupper(as.character(ontology))

  ancestor_obj <- switch(
    ontology,
    BP = GO.db::GOBPANCESTOR,
    MF = GO.db::GOMFANCESTOR,
    CC = GO.db::GOCCANCESTOR,
    stop("Unsupported ontology for ancestor map: ", ontology, call. = FALSE)
  )

  AnnotationDbi::as.list(ancestor_obj)
}

get_go_term_names <- function(go_ids) {
  go_ids <- unique(as.character(go_ids))
  go_ids <- go_ids[!is.na(go_ids) & nzchar(go_ids)]

  if (length(go_ids) == 0) {
    return(tibble::tibble(go_id = character(), go_name = character()))
  }

  out <- AnnotationDbi::select(
    GO.db::GO.db,
    keys = go_ids,
    keytype = "GOID",
    columns = c("GOID", "TERM")
  ) |>
    tibble::as_tibble() |>
    dplyr::transmute(
      go_id = as.character(.data$GOID),
      go_name = as.character(.data$TERM)
    ) |>
    dplyr::distinct(.data$go_id, .keep_all = TRUE)

  out
}

# ---- Ancestor-based prepartitioning ------------------------------------------

assign_go_ancestor_blocks_one_ontology <- function(ontology_df,
                                                   sem_data,
                                                   max_terms_per_block = 600,
                                                   min_terms_per_ancestor_block = 25) {
  ontology <- unique(ontology_df$ontology)
  if (length(ontology) != 1 || is.na(ontology)) {
    stop("Ancestor block assignment requires one ontology.", call. = FALSE)
  }

  ontology_df <- ontology_df |>
    dplyr::distinct(.data$ontology, .data$go_id, .keep_all = TRUE)

  term_ids <- ontology_df$go_id
  ancestor_map <- get_go_ancestor_map(ontology)

  # Candidate ancestors include each term itself plus its GO ancestors.
  candidate_tbl <- purrr::map_dfr(term_ids, function(term_id) {
    ancestors <- ancestor_map[[term_id]]
    candidates <- unique(c(term_id, ancestors))
    candidates <- candidates[!is.na(candidates) & nzchar(candidates)]

    if (length(candidates) == 0) {
      return(tibble::tibble(go_id = term_id, ancestor_go_id = NA_character_))
    }

    tibble::tibble(
      go_id = term_id,
      ancestor_go_id = candidates
    )
  }) |>
    dplyr::filter(!is.na(.data$ancestor_go_id))

  if (nrow(candidate_tbl) == 0) {
    return(ontology_df |>
      dplyr::mutate(
        semantic_preblock_id = paste0(.data$ontology, "_", sanitize_cluster_part(.data$root_node)),
        semantic_preblock_go_id = NA_character_,
        semantic_preblock_name = .data$root_node,
        semantic_preblock_method = "root_node_fallback_no_go_ancestors",
        semantic_preblock_n_terms = dplyr::n()
      ))
  }

  ancestor_counts <- candidate_tbl |>
    dplyr::count(.data$ancestor_go_id, name = "ancestor_n_terms")

  ic_tbl <- tibble::tibble(
    ancestor_go_id = names(sem_data@IC),
    ancestor_ic = as.numeric(sem_data@IC)
  )

  candidate_scored <- candidate_tbl |>
    dplyr::left_join(ancestor_counts, by = "ancestor_go_id") |>
    dplyr::left_join(ic_tbl, by = "ancestor_go_id") |>
    dplyr::mutate(
      ancestor_ic = dplyr::coalesce(.data$ancestor_ic, 0),
      usable_size = !is.na(.data$ancestor_n_terms) &
        .data$ancestor_n_terms >= min_terms_per_ancestor_block &
        .data$ancestor_n_terms <= max_terms_per_block
    )

  chosen <- candidate_scored |>
    dplyr::filter(.data$usable_size) |>
    dplyr::arrange(.data$go_id, dplyr::desc(.data$ancestor_ic), .data$ancestor_n_terms, .data$ancestor_go_id) |>
    dplyr::group_by(.data$go_id) |>
    dplyr::slice(1) |>
    dplyr::ungroup() |>
    dplyr::select(
      .data$go_id,
      semantic_preblock_go_id = .data$ancestor_go_id,
      semantic_preblock_n_terms = .data$ancestor_n_terms,
      semantic_preblock_ic = .data$ancestor_ic
    )

  # Terms without a usable intermediate ancestor fall back to root_node.
  out <- ontology_df |>
    dplyr::left_join(chosen, by = "go_id") |>
    dplyr::mutate(
      semantic_preblock_go_id = dplyr::coalesce(.data$semantic_preblock_go_id, NA_character_),
      semantic_preblock_method = dplyr::if_else(
        is.na(.data$semantic_preblock_go_id),
        "root_node_fallback_no_usable_ancestor",
        "go_ancestor_prepartition"
      ),
      semantic_preblock_id = dplyr::if_else(
        is.na(.data$semantic_preblock_go_id),
        paste0(.data$ontology, "_", sanitize_cluster_part(.data$root_node)),
        paste0(.data$ontology, "_", sanitize_cluster_part(.data$semantic_preblock_go_id))
      )
    )

  ancestor_names <- get_go_term_names(out$semantic_preblock_go_id)

  out |>
    dplyr::left_join(
      ancestor_names |>
        dplyr::rename(
          semantic_preblock_go_id = .data$go_id,
          ancestor_term_name = .data$go_name
        ),
      by = "semantic_preblock_go_id"
    ) |>
    dplyr::mutate(
      semantic_preblock_name = dplyr::coalesce(.data$ancestor_term_name, .data$root_node),
      semantic_preblock_n_terms = dplyr::coalesce(.data$semantic_preblock_n_terms, NA_integer_)
    ) |>
    dplyr::select(-dplyr::any_of("ancestor_term_name"))
}

assign_go_ancestor_blocks <- function(go_term_dim,
                                      semdata,
                                      max_terms_per_block = 600,
                                      min_terms_per_ancestor_block = 25) {
  go_term_dim |>
    dplyr::group_by(.data$ontology) |>
    dplyr::group_split() |>
    purrr::map_dfr(function(ontology_df) {
      ont <- unique(ontology_df$ontology)
      assign_go_ancestor_blocks_one_ontology(
        ontology_df = ontology_df,
        sem_data = semdata[[ont]],
        max_terms_per_block = max_terms_per_block,
        min_terms_per_ancestor_block = min_terms_per_ancestor_block
      )
    })
}

# ---- Pairwise semantic clustering within prepartitioned blocks ----------------

cluster_go_term_block <- function(block_df,
                                  sem_data,
                                  similarity_cutoff = 0.70,
                                  max_terms_per_block = 600,
                                  large_block_strategy = c("ancestor_single_cluster", "singleton")) {
  large_block_strategy <- match.arg(large_block_strategy)

  ontology <- unique(block_df$ontology)
  preblock_id <- unique(block_df$semantic_preblock_id)
  preblock_name <- unique(block_df$semantic_preblock_name)

  if (length(ontology) != 1 || is.na(ontology)) {
    stop("Each semantic block must have exactly one ontology.", call. = FALSE)
  }
  if (length(preblock_id) != 1 || is.na(preblock_id)) {
    preblock_id <- paste0(ontology, "_UNSPECIFIED")
  }
  if (length(preblock_name) != 1 || is.na(preblock_name)) {
    preblock_name <- preblock_id
  }

  block_df <- dplyr::distinct(block_df, .data$ontology, go_id, .keep_all = TRUE)
  n_terms <- nrow(block_df)
  block_label <- sanitize_cluster_part(preblock_id)

  if (n_terms == 0) {
    return(tibble::tibble())
  }

  if (n_terms == 1) {
    return(block_df |>
      dplyr::mutate(
        semantic_cluster_id = paste0(block_label, "_singleton_1"),
        semantic_block_id = block_label,
        semantic_block_name = preblock_name,
        cluster_method = "singleton_one_term",
        semantic_similarity_cutoff = similarity_cutoff,
        semantic_block_n_terms = n_terms
      ))
  }

  valid_terms <- intersect(block_df$go_id, names(sem_data@IC))
  invalid_terms <- setdiff(block_df$go_id, valid_terms)

  if (length(valid_terms) > max_terms_per_block) {
    if (large_block_strategy == "ancestor_single_cluster") {
      return(block_df |>
        dplyr::mutate(
          semantic_cluster_id = paste0(block_label, "_largeblock"),
          semantic_block_id = block_label,
          semantic_block_name = preblock_name,
          cluster_method = paste0("ancestor_large_block_no_pairwise_gt_", max_terms_per_block),
          semantic_similarity_cutoff = similarity_cutoff,
          semantic_block_n_terms = n_terms
        ))
    }

    return(block_df |>
      dplyr::mutate(
        semantic_cluster_id = paste0(block_label, "_singleton_", dplyr::row_number()),
        semantic_block_id = block_label,
        semantic_block_name = preblock_name,
        cluster_method = paste0("singleton_large_block_no_pairwise_gt_", max_terms_per_block),
        semantic_similarity_cutoff = similarity_cutoff,
        semantic_block_n_terms = n_terms
      ))
  }

  clustered <- tibble::tibble()

  if (length(valid_terms) >= 2) {
    message("  Pairwise Wang similarity: ", ontology, " / ", preblock_name, " (", length(valid_terms), " terms)")

    sim_matrix <- GOSemSim::mgoSim(
      valid_terms,
      valid_terms,
      semData = sem_data,
      measure = "Wang",
      combine = NULL
    )

    sim_matrix[is.na(sim_matrix)] <- 0
    
    # Wang semantic similarities are not guaranteed to produce a Euclidean
    # distance matrix. Stabilize before hierarchical clustering.
    sim_matrix <- (sim_matrix + t(sim_matrix)) / 2
    sim_matrix[sim_matrix < 0] <- 0
    sim_matrix[sim_matrix > 1] <- 1
    diag(sim_matrix) <- 1
    
    dist_matrix <- stats::as.dist(1 - sim_matrix)
    
    hc <- stats::hclust(dist_matrix, method = "average")
    
    clusters <- stats::cutree(hc, h = 1 - similarity_cutoff)
    
    clustered <- tibble::tibble(
      go_id = names(clusters),
      semantic_cluster_id = paste0(block_label, "_", as.integer(clusters)),
      semantic_block_id = block_label,
      semantic_block_name = preblock_name,
      cluster_method = "GOSemSim_Wang_average_within_GO_ancestor_block",
      semantic_similarity_cutoff = similarity_cutoff,
      semantic_block_n_terms = n_terms
    )
  } else if (length(valid_terms) == 1) {
    clustered <- tibble::tibble(
      go_id = valid_terms,
      semantic_cluster_id = paste0(block_label, "_singleton_1"),
      semantic_block_id = block_label,
      semantic_block_name = preblock_name,
      cluster_method = "singleton_only_one_valid_term",
      semantic_similarity_cutoff = similarity_cutoff,
      semantic_block_n_terms = n_terms
    )
  }

  if (length(invalid_terms) > 0) {
    invalid_clustered <- tibble::tibble(
      go_id = invalid_terms,
      semantic_cluster_id = paste0(block_label, "_unmapped_", seq_along(invalid_terms)),
      semantic_block_id = block_label,
      semantic_block_name = preblock_name,
      cluster_method = "singleton_not_in_semdata",
      semantic_similarity_cutoff = similarity_cutoff,
      semantic_block_n_terms = n_terms
    )
    clustered <- dplyr::bind_rows(clustered, invalid_clustered)
  }

  dplyr::left_join(block_df, clustered, by = "go_id")
}

# ---- Public API ----------------------------------------------------------------

build_go_semantic_clusters <- function(go_term_dim,
                                       similarity_cutoff = 0.70,
                                       max_terms_per_block = 600,
                                       min_terms_per_ancestor_block = 25,
                                       large_block_strategy = c("ancestor_single_cluster", "singleton")) {
  check_go_semantic_cluster_packages()
  large_block_strategy <- match.arg(large_block_strategy)

  required_cols <- c("ontology", "go_id", "go_name", "root_node")
  missing_cols <- setdiff(required_cols, names(go_term_dim))
  if (length(missing_cols) > 0) {
    stop("go_term_dim is missing columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  go_term_dim <- go_term_dim |>
    dplyr::filter(!is.na(.data$ontology), !is.na(go_id), grepl("^GO:", go_id)) |>
    dplyr::mutate(root_node = dplyr::coalesce(.data$root_node, paste0(.data$ontology, "_UNSPECIFIED"))) |>
    dplyr::distinct(.data$ontology, go_id, .keep_all = TRUE)

  ontologies <- sort(unique(go_term_dim$ontology))
  semdata <- make_semdata_by_ontology(ontologies)

  message("Assigning GO ancestor prepartition blocks...")
  go_term_dim <- assign_go_ancestor_blocks(
    go_term_dim = go_term_dim,
    semdata = semdata,
    max_terms_per_block = max_terms_per_block,
    min_terms_per_ancestor_block = min_terms_per_ancestor_block
  )

  go_term_dim |>
    dplyr::group_by(.data$ontology, .data$semantic_preblock_id) |>
    dplyr::group_split() |>
    purrr::map_dfr(function(block) {
      ont <- unique(block$ontology)
      cluster_go_term_block(
        block_df = block,
        sem_data = semdata[[ont]],
        similarity_cutoff = similarity_cutoff,
        max_terms_per_block = max_terms_per_block,
        large_block_strategy = large_block_strategy
      )
    }) |>
    dplyr::arrange(.data$ontology, .data$semantic_block_id, .data$semantic_cluster_id, go_id)
}

summarize_go_semantic_clusters <- function(go_cluster_membership) {
  go_cluster_membership |>
    dplyr::group_by(
      .data$ontology,
      .data$semantic_block_id,
      .data$semantic_block_name,
      .data$root_node,
      .data$semantic_cluster_id
    ) |>
    dplyr::summarise(
      n_terms = dplyr::n_distinct(.data$go_id),
      representative_go_ids = paste(utils::head(sort(unique(.data$go_id)), 10), collapse = "; "),
      representative_terms = paste(utils::head(unique(.data$go_name), 10), collapse = "; "),
      cluster_method = dplyr::first(.data$cluster_method),
      semantic_similarity_cutoff = dplyr::first(.data$semantic_similarity_cutoff),
      semantic_block_n_terms = dplyr::first(.data$semantic_block_n_terms),
      preblock_method = dplyr::first(.data$semantic_preblock_method),
      preblock_go_id = dplyr::first(.data$semantic_preblock_go_id),
      .groups = "drop"
    ) |>
    dplyr::arrange(.data$ontology, .data$semantic_block_id, .data$semantic_cluster_id)
}
