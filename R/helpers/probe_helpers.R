# probe_helpers.R
# -------------------------------------------------
# Helper functions to extract filtered probe counts from chip result objects
# -------------------------------------------------

#' Get filtered probe count for a specific comparison from a result object
#'
#' @param res_obj A result object (e.g., res_hu35ksuba)
#' @param comparison_name Comparison string (e.g., "PB/T-ALL")
#' @return Integer; number of filtered probes (or NA if not found)
#' @export
get_filtered_probe_count <- function(res_obj, comparison_name) {
  summary_list <- res_obj$`__summary__`$probe_hits_by_group
  
  if (is.null(summary_list)) {
    warning("Missing `probe_hits_by_group` in result object.")
    return(NA)
  }
  
  parsed <- tibble::tibble(
    group = sub("::.*", "", names(summary_list)),
    comparison = sub(".*::", "", names(summary_list)),
    probe_count = sapply(summary_list, length)
  )
  
  match_row <- dplyr::filter(parsed, comparison == comparison_name)
  if (nrow(match_row) == 0) {
    message("No match found for comparison: ", comparison_name)
    return(NA)
  }
  
  return(match_row$probe_count)
}

#' Collect filtered probe counts across multiple chips and comparisons
#'
#' @param chips Character vector of chip IDs (e.g., c("hu35ksuba", "hu6800"))
#' @param chip_label_map Named character vector mapping chip IDs to labels (e.g., c(hu35ksuba = "GPL98"))
#' @return A tibble with: group, comparison, chip, chip_label, probe_count
#' @export
collect_probe_counts <- function(chips,
                                 chip_label_map = c(hu35ksuba = "GPL98", hu6800 = "GPL80")) {
  all_data <- list()
  
  for (chip_id in chips) {
    res_name <- paste0("res_", chip_id)
    if (!exists(res_name, envir = .GlobalEnv)) {
      warning("Missing result object: ", res_name)
      next
    }
    
    res <- get(res_name, envir = .GlobalEnv)
    summary_list <- res$`__summary__`$probe_hits_by_group
    
    if (is.null(summary_list)) {
      warning("Missing `probe_hits_by_group` in ", res_name)
      next
    }
    
    chip_data <- tibble::tibble(
      group = sub("::.*", "", names(summary_list)),
      comparison = sub(".*::", "", names(summary_list)),
      probe_count = sapply(summary_list, length),
      chip = chip_id,
      chip_label = if (!is.null(chip_label_map[[chip_id]])) chip_label_map[[chip_id]] else chip_id
    )
    
    all_data[[length(all_data) + 1]] <- chip_data
  }
  
  dplyr::bind_rows(all_data)
}

#' Collect filtered probe ID sets per group and comparison
#'
#' @param chips Character vector of chip IDs (e.g., c("hu35ksuba", "hu6800"))
#' @return A tibble with: group, comparison, filtered_probes (vector column)
#' @export
collect_filtered_probe_sets <- function(chips) {
  probe_list <- list()
  
  for (chip_id in chips) {
    res_name <- paste0("res_", chip_id)
    if (!exists(res_name, envir = .GlobalEnv)) {
      warning("Result object not found: ", res_name)
      next
    }
    
    res <- get(res_name, envir = .GlobalEnv)
    summary_list <- res$`__summary__`$probe_hits_by_group
    if (is.null(summary_list)) next
    
    for (nm in names(summary_list)) {
      group <- sub("::.*", "", nm)
      comparison <- sub(".*::", "", nm)
      probes <- summary_list[[nm]]
      key <- paste(group, comparison, sep = "::")
      
      # Merge probes from multiple chips into same (group::comparison) set
      if (!key %in% names(probe_list)) {
        probe_list[[key]] <- probes
      } else {
        probe_list[[key]] <- union(probe_list[[key]], probes)
      }
    }
  }
  
  tibble::tibble(
    group = sub("::.*", "", names(probe_list)),
    comparison = sub(".*::", "", names(probe_list)),
    filtered_probes = unname(probe_list)
  )
}

#' Compare filtered probe sets across two chip result objects by comparison
#'
#' For each shared group::comparison pair, reports the number of shared and unique probes.
#'
#' @param res_obj A list of two result objects (e.g., list(res_hu35ksuba, res_hu6800))
#' @param chips A character vector of length 2 naming the corresponding chip IDs.
#' @param report_path Optional file path to write the report.
#'
#' @return A tibble with: group, comparison, shared, unique_chip1, unique_chip2
#' @export
compare_probe_sets_by_comparison <- function(res_obj,
                                             chips = c("hu35ksuba", "hu6800"),
                                             report_path = NULL) {
  if (length(res_obj) != 2 || length(chips) != 2) {
    stop("`res_obj` and `chips` must both be length 2.")
  }
  
  # Extract probe sets by group::comparison
  get_probe_map <- function(res) {
    lst <- res$`__summary__`$probe_hits_by_group
    if (is.null(lst)) stop("Missing `probe_hits_by_group` in result object")
    lst
  }
  
  probes1 <- get_probe_map(res_obj[[1]])
  probes2 <- get_probe_map(res_obj[[2]])
  
  all_keys <- union(names(probes1), names(probes2))
  parsed <- tibble::tibble(
    group = sub("::.*", "", all_keys),
    comparison = sub(".*::", "", all_keys),
    key = all_keys
  )
  
  shared_info <- parsed |>
    dplyr::rowwise() |>
    dplyr::mutate(
      probes_1 = list(probes1[[key]] %||% character(0)),
      probes_2 = list(probes2[[key]] %||% character(0)),
      shared = length(intersect(probes_1, probes_2)),
      unique_chip1 = length(setdiff(probes_1, probes_2)),
      unique_chip2 = length(setdiff(probes_2, probes_1))
    ) |>
    dplyr::ungroup() |>
    dplyr::select(group, comparison, shared, unique_chip1, unique_chip2)
  
  if (!is.null(report_path)) {
    report_lines <- c(
      glue::glue("🔍 Probe Set Comparison by Group/Comparison — {Sys.time()}"),
      "------------------------------------------------------"
    )
    
    for (i in seq_len(nrow(shared_info))) {
      row <- shared_info[i, ]
      report_lines <- c(report_lines,
                        glue::glue(
                          "[{row$group} / {row$comparison}] ",
                          "Shared: {row$shared}, ",
                          "{chips[1]}-unique: {row$unique_chip1}, ",
                          "{chips[2]}-unique: {row$unique_chip2}"
                        )
      )
    }
    
    writeLines(c(report_lines, "------------------------------------------------------"), con = report_path)
  }
  
  return(shared_info)
}
