# ==============================================================================
# File: generate_biological_summary_reports_from_duckdb.R
# Purpose: Generate chip × engine biological enrichment summaries from DuckDB
# Role: Development/debugging reporting layer for biological enrichment results
# ============================================================================== 

library(DBI)
library(duckdb)
library(here)

# ---- Paths --------------------------------------------------------------------

view_script_path <- here::here(
  "R/warehouse/views/create_biological_reporting_views.R"
)

db_path <- here::here(
  "output/global_cancer/warehouse/global_cancer_results.duckdb"
)

report_dir <- here::here(
  "output/global_cancer/biological_enrichment/reports/summary_txt"
)

# ---- Source view module -------------------------------------------------------

if (!file.exists(view_script_path)) {
  stop("View script not found: ", view_script_path, call. = FALSE)
}

source(view_script_path)

# ---- Formatting helpers -------------------------------------------------------

fmt_num <- function(x, digits = 4) {
  if (length(x) == 0 || is.na(x)) return("NA")
  formatC(as.numeric(x), digits = digits, format = "fg", flag = "#")
}

fmt_p <- function(x) {
  if (length(x) == 0 || is.na(x)) return("NA")
  x <- as.numeric(x)
  if (x < 0.001) return("<0.001")
  formatC(x, digits = 3, format = "fg", flag = "#")
}

first_present <- function(df, candidates) {
  candidate <- candidates[candidates %in% names(df)][1]
  if (length(candidate) == 0 || is.na(candidate)) return(NULL)
  candidate
}

safe_value <- function(row, column, default = NA) {
  if (is.null(column) || !column %in% names(row)) return(default)
  value <- row[[column]][1]
  if (length(value) == 0) default else value
}

safe_text <- function(x, default = "NA") {
  if (length(x) == 0 || is.na(x) || identical(x, "")) default else as.character(x)
}

engine_descriptor_columns <- function(engine) {
  switch(
    engine,
    complexity = c(
      "kappa_delta",
      "effrank_delta",
      "sparsity_delta",
      "composite_kappa_delta"
    ),
    entropy = c(
      "shannon_delta",
      "spectral_delta"
    ),
    character(0)
  )
}

format_descriptor_line <- function(row, engine) {
  cols <- engine_descriptor_columns(engine)
  cols <- cols[cols %in% names(row)]

  if (length(cols) == 0) {
    return("- descriptors: unavailable")
  }

  values <- vapply(
    cols,
    function(col) paste0(col, "=", fmt_num(row[[col]][1])),
    character(1)
  )

  paste0("- descriptors: ", paste(values, collapse = "; "))
}

format_term_line <- function(row, engine) {
  name <- safe_text(row$report_gene_set_name[1], row$gene_set_id[1])
  id <- safe_text(row$gene_set_id[1])
  p <- fmt_p(row$p_report[1])
  direction <- safe_text(row$report_direction[1])

  descriptor_col <- switch(
    engine,
    complexity = first_present(row, c("kappa_delta", "composite_kappa_delta")),
    entropy = first_present(row, c("spectral_delta", "shannon_delta")),
    NULL
  )

  descriptor_text <- if (!is.null(descriptor_col)) {
    paste0(" | ", descriptor_col, "=", fmt_num(row[[descriptor_col]][1]))
  } else {
    ""
  }

  paste0(
    "- ", name,
    " [", id, "]",
    ": p=", p,
    " | direction=", direction,
    descriptor_text
  )
}

# ---- Database helpers ---------------------------------------------------------

relation_exists <- function(con, relation_name) {
  DBI::dbGetQuery(
    con,
    sprintf(
      "
      SELECT COUNT(*) AS n
      FROM information_schema.tables
      WHERE table_name = %s
      ",
      DBI::dbQuoteString(con, relation_name)
    )
  )$n[[1]] > 0
}

get_available_chips <- function(con) {
  DBI::dbGetQuery(
    con,
    "
    SELECT DISTINCT chip
    FROM vw_report_biological_aggregated_results
    ORDER BY chip
    "
  )$chip
}

get_structural_comparison_rows <- function(con, chip, engine) {
  # Optional structural-comparison context.
  # This intentionally stays best-effort because structural synthesis view names
  # may evolve independently of biological enrichment reporting.
  candidate_views <- c(
    "vw_structural_phenotype_wide_with_latent_overlay",
    "vw_structural_phenotype_wide"
  )

  structural_view <- candidate_views[vapply(candidate_views, function(view) relation_exists(con, view), logical(1))][1]

  if (length(structural_view) == 0 || is.na(structural_view)) {
    return(data.frame())
  }

  fields <- DBI::dbListFields(con, structural_view)

  required <- c("chip", "comparison")
  if (!all(required %in% fields)) return(data.frame())

  wanted <- unique(c(
    "comparison",
    "chip",
    "filter_regime",
    paste0(engine, "__p_perm"),
    paste0(engine, "__p_report"),
    paste0(engine, "__kappa_delta"),
    paste0(engine, "__spectral_delta"),
    "complexity__kappa_delta",
    "entropy__spectral_delta",
    "mp_spectral__spectral_entropy_delta",
    "latent__centroid_distance"
  ))

  selected <- wanted[wanted %in% fields]
  if (length(selected) == 0) return(data.frame())

  sql <- sprintf(
    "SELECT %s FROM %s WHERE chip = ? ORDER BY comparison, filter_regime",
    paste(DBI::dbQuoteIdentifier(con, selected), collapse = ", "),
    DBI::dbQuoteIdentifier(con, structural_view)
  )

  DBI::dbGetQuery(con, sql, params = list(chip))
}

get_biological_rows <- function(con, chip, engine, p_threshold = 0.05) {
  DBI::dbGetQuery(
    con,
    "
    SELECT *
    FROM vw_report_biological_aggregated_results
    WHERE chip = ?
      AND engine = ?
      AND p_report IS NOT NULL
      AND p_report <= ?
    ORDER BY comparison, filter_regime, gene_set_mode, p_report, report_gene_set_name
    ",
    params = list(chip, engine, p_threshold)
  )
}

get_biological_row_counts <- function(con, chip, engine) {
  DBI::dbGetQuery(
    con,
    "
    SELECT
      filter_regime,
      gene_set_mode,
      COUNT(*) AS n_rows,
      SUM(CASE WHEN p_report IS NOT NULL THEN 1 ELSE 0 END) AS n_with_p,
      SUM(CASE WHEN p_report IS NOT NULL AND p_report <= 0.05 THEN 1 ELSE 0 END)
        AS n_significant_005
    FROM vw_report_biological_aggregated_results
    WHERE chip = ?
      AND engine = ?
    GROUP BY filter_regime, gene_set_mode
    ORDER BY filter_regime, gene_set_mode
    ",
    params = list(chip, engine)
  )
}

# ---- Report assembly ----------------------------------------------------------

table_lines <- function(df) {
  if (nrow(df) == 0) return("  none")

  apply(
    df,
    1,
    function(row) {
      paste0(
        "  - ",
        paste(names(row), row, sep = "=", collapse = "; ")
      )
    }
  )
}

append_structural_context <- function(lines, structural_rows, comparison, filter_regime, engine) {
  if (nrow(structural_rows) == 0) {
    return(c(lines, "  - structural comparison-level result: unavailable"))
  }

  match_rows <- structural_rows[structural_rows$comparison == comparison, , drop = FALSE]

  if ("filter_regime" %in% names(match_rows)) {
    match_rows <- match_rows[match_rows$filter_regime == filter_regime, , drop = FALSE]
  }

  if (nrow(match_rows) == 0) {
    return(c(lines, "  - structural comparison-level result: unavailable"))
  }

  row <- match_rows[1, , drop = FALSE]

  candidate_cols <- switch(
    engine,
    complexity = c("complexity__kappa_delta", "complexity__p_perm"),
    entropy = c("entropy__spectral_delta", "entropy__p_perm"),
    character(0)
  )
  candidate_cols <- c(candidate_cols, "mp_spectral__spectral_entropy_delta", "latent__centroid_distance")
  candidate_cols <- candidate_cols[candidate_cols %in% names(row)]

  if (length(candidate_cols) == 0) {
    return(c(lines, "  - structural comparison-level result: present, but no recognized descriptor columns found"))
  }

  context <- vapply(
    candidate_cols,
    function(col) paste0(col, "=", fmt_num(row[[col]][1])),
    character(1)
  )

  c(lines, paste0("  - structural comparison-level result: ", paste(context, collapse = "; ")))
}

build_chip_engine_report <- function(con, chip, engine, p_threshold = 0.05, max_terms_per_section = 50) {
  biological_rows <- get_biological_rows(con, chip, engine, p_threshold = p_threshold)
  structural_rows <- get_structural_comparison_rows(con, chip, engine)
  row_counts <- get_biological_row_counts(con, chip, engine)

  lines <- c(
    paste0("# Biological enrichment summary: ", chip, " — ", engine),
    "",
    "Generated from DuckDB reporting views.",
    "",
    "Interpretation levels:",
    "- Structural comparison-level context summarizes normal/tumor structural descriptors when available.",
    "- Biological term-level results list GO/KEGG/MSIGDB terms with gene-set-level p-values.",
    "",
    "Run coverage:",
    table_lines(row_counts),
    ""
  )

  if (nrow(biological_rows) == 0) {
    return(c(
      lines,
      paste0("No biological terms found at p <= ", fmt_p(p_threshold), ".")
    ))
  }

  group_keys <- unique(
    biological_rows[, c("comparison", "filter_regime"), drop = FALSE]
  )
  group_keys <- group_keys[order(group_keys$comparison, group_keys$filter_regime), , drop = FALSE]

  for (i in seq_len(nrow(group_keys))) {
    comparison_i <- group_keys$comparison[i]
    filter_i <- group_keys$filter_regime[i]

    lines <- c(
      lines,
      paste0("## Comparison: ", comparison_i),
      paste0("### Filter regime: ", filter_i),
      "",
      "Structural comparison-level context:"
    )

    lines <- append_structural_context(
      lines = lines,
      structural_rows = structural_rows,
      comparison = comparison_i,
      filter_regime = filter_i,
      engine = engine
    )

    subset_i <- biological_rows[
      biological_rows$comparison == comparison_i &
        biological_rows$filter_regime == filter_i,
      ,
      drop = FALSE
    ]

    for (mode_i in c("GO_BP", "GO_MF", "KEGG", "MSIGDB")) {
      mode_rows <- subset_i[subset_i$gene_set_mode == mode_i, , drop = FALSE]

      if (nrow(mode_rows) == 0) next

      mode_rows <- mode_rows[order(mode_rows$p_report, mode_rows$report_gene_set_name), , drop = FALSE]
      if (nrow(mode_rows) > max_terms_per_section) {
        mode_rows <- mode_rows[seq_len(max_terms_per_section), , drop = FALSE]
      }

      lines <- c(
        lines,
        "",
        paste0("Biological term-level results: ", mode_i),
        vapply(
          seq_len(nrow(mode_rows)),
          function(j) format_term_line(mode_rows[j, , drop = FALSE], engine),
          character(1)
        )
      )
    }

    lines <- c(lines, "")
  }

  lines
}

write_chip_engine_report <- function(con, chip, engine, report_dir, p_threshold = 0.05) {
  lines <- build_chip_engine_report(
    con = con,
    chip = chip,
    engine = engine,
    p_threshold = p_threshold
  )

  filename <- paste0(
    "biological_", engine, "_", chip, "_summary.txt"
  )

  path <- file.path(report_dir, filename)
  writeLines(lines, con = path, useBytes = TRUE)
  path
}

# ---- Main ---------------------------------------------------------------------

generate_biological_summary_reports_from_duckdb <- function(
    db_path = here::here("output/global_cancer/warehouse/global_cancer_results.duckdb"),
    report_dir = here::here("output/global_cancer/biological_enrichment/reports/summary_txt"),
    engines = c("complexity", "entropy"),
    chips = NULL,
    p_threshold = 0.05,
    print_validation = TRUE) {

  dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

  con <- DBI::dbConnect(duckdb::duckdb(), db_path)
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)

  create_biological_reporting_views(con)

  if (isTRUE(print_validation)) {
    validation <- validate_biological_reporting_views(con)
    message("Biological reporting view validation:")
    print(validation)
  }

  if (is.null(chips)) {
    chips <- get_available_chips(con)
  }

  output_paths <- character(0)

  for (engine in engines) {
    for (chip in chips) {
      output_paths <- c(
        output_paths,
        write_chip_engine_report(
          con = con,
          chip = chip,
          engine = engine,
          report_dir = report_dir,
          p_threshold = p_threshold
        )
      )
    }
  }

  message("✅ Biological summary reports written:")
  for (path in output_paths) {
    message("  - ", path)
  }

  invisible(output_paths)
}

# ---- Run when sourced interactively or called by Rscript ----------------------

if (sys.nframe() == 0) {
  generate_biological_summary_reports_from_duckdb(
    db_path = db_path,
    report_dir = report_dir
  )
}
