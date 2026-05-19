# ==============================================================================
# inspect_structural_rds_inventory.R
# Purpose: inspect structural inference RDS contents before downstream debugging
# ==============================================================================

library(tidyverse)

rds_root <- file.path(
  "output", "global_cancer", "structural_inference", "RData"
)

rds_files <- list.files(
  rds_root,
  pattern = "\\.rds$",
  recursive = TRUE,
  full.names = TRUE
)

inspect_rds <- function(path) {
  obj <- readRDS(path)
  
  top_class <- paste(class(obj), collapse = ", ")
  top_names <- if (is.list(obj)) names(obj) else NA_character_
  
  tibble(
    file = path,
    basename = basename(path),
    rel_path = gsub(paste0("^", normalizePath(rds_root, winslash = "/"), "/?"), "", normalizePath(path, winslash = "/")),
    top_class = top_class,
    top_length = if (is.list(obj)) length(obj) else NA_integer_,
    top_names = paste(top_names, collapse = " | ")
  )
}

inventory <- map_dfr(rds_files, inspect_rds)

print(inventory, n = Inf, width = Inf)

# ---- Deeper inspection for list-like result RDS files -------------------------

inspect_nested <- function(path) {
  obj <- readRDS(path)
  
  if (!is.list(obj)) {
    return(tibble(
      file = path,
      item = NA_character_,
      class = paste(class(obj), collapse = ", "),
      nrow = if (is.data.frame(obj)) nrow(obj) else NA_integer_,
      ncol = if (is.data.frame(obj)) ncol(obj) else NA_integer_,
      columns = if (is.data.frame(obj)) paste(names(obj), collapse = " | ") else NA_character_
    ))
  }
  
  map_dfr(names(obj), function(nm) {
    x <- obj[[nm]]
    
    tibble(
      file = path,
      item = nm,
      class = paste(class(x), collapse = ", "),
      nrow = if (is.data.frame(x)) nrow(x) else NA_integer_,
      ncol = if (is.data.frame(x)) ncol(x) else NA_integer_,
      columns = if (is.data.frame(x)) paste(names(x), collapse = " | ") else NA_character_
    )
  })
}

nested_inventory <- map_dfr(rds_files, inspect_nested)

print(nested_inventory, n = Inf, width = Inf)

# ---- Search specifically for filter_regime / selection / comparison fields ----

extract_field_values <- function(path) {
  obj <- readRDS(path)
  
  tables <- list()
  
  if (is.data.frame(obj)) {
    tables[[basename(path)]] <- obj
  } else if (is.list(obj)) {
    tables <- keep(obj, is.data.frame)
  }
  
  map_dfr(names(tables), function(nm) {
    tbl <- tables[[nm]]
    
    tibble(
      file = path,
      item = nm,
      nrow = nrow(tbl),
      has_chip = "chip" %in% names(tbl),
      chips = if ("chip" %in% names(tbl)) paste(sort(unique(tbl$chip)), collapse = " | ") else NA_character_,
      has_filter_regime = "filter_regime" %in% names(tbl),
      filter_regimes = if ("filter_regime" %in% names(tbl)) paste(sort(unique(tbl$filter_regime)), collapse = " | ") else NA_character_,
      has_selection_regime = "selection_regime" %in% names(tbl),
      selection_regimes = if ("selection_regime" %in% names(tbl)) paste(sort(unique(tbl$selection_regime)), collapse = " | ") else NA_character_,
      has_comparison = "comparison" %in% names(tbl),
      n_comparisons = if ("comparison" %in% names(tbl)) n_distinct(tbl$comparison) else NA_integer_
    )
  })
}

field_inventory <- map_dfr(rds_files, extract_field_values)

print(field_inventory, n = Inf, width = Inf)

# ---- Save audit tables --------------------------------------------------------

audit_dir <- file.path(
  "output", "global_cancer", "structural_inference", "diagnostics", "rds_inventory"
)

dir.create(audit_dir, recursive = TRUE, showWarnings = FALSE)

write_csv(inventory, file.path(audit_dir, "rds_top_level_inventory.csv"))
write_csv(nested_inventory, file.path(audit_dir, "rds_nested_inventory.csv"))
write_csv(field_inventory, file.path(audit_dir, "rds_field_inventory.csv"))

message("Saved RDS inventory to: ", audit_dir)