# ------------------------------------------------------------------------------
# File: write_probe_count_tables_html.R
# Purpose: Generate filtered-probe count summary tables by cancer group and write
#   them as HTML fragments and CSV outputs for reporting.
# Role: Helper (filtered probe-count table writer)
# Pipeline: Reporting
# Project: Global Cancer Complexity
# Author: Ali M. Al-Timimi
# Created: 2026
# ------------------------------------------------------------------------------

#' Write Filtered-Probe Count Tables
#'
#' Generates filtered-probe count summaries by comparison and cancer group for
#' the supported Affymetrix chip platforms, then writes group-level HTML
#' fragments and a CSV summary table.
#'
#' The HTML files are intended for inclusion in Quarto-generated reports. The
#' CSV output provides a structured tabular record of filtered-probe counts by
#' comparison and chip platform.
#'
#' @param res_hu35ksuba Result object for the hu35ksuba chip.
#' @param res_hu6800 Result object for the hu6800 chip.
#' @param geo_chip_map Named list mapping internal chip IDs to GEO platform IDs.
#'   Defaults to \code{list(hu35ksuba = "GPL98", hu6800 = "GPL80")}.
#' @param output_dir Directory where HTML fragments will be written.
#' @param data_dir Root directory where the CSV summary will be written under
#'   \code{filter_data/}.
#'
#' @details
#' This helper extracts probe-count summaries from the \code{__summary__}
#' element of each chip-specific result object, reshapes the counts into a wide
#' comparison-by-chip table, writes the full summary to CSV, and writes one HTML
#' fragment per cancer group.
#'
#' @return Invisibly returns a named character vector of HTML file paths, indexed
#' by cancer group.

write_probe_count_tables_html <- function(res_hu35ksuba,
                                          res_hu6800,
                                          geo_chip_map = list(hu35ksuba = "GPL98", hu6800 = "GPL80"),
                                          output_dir = "quarto/resources/tables/probes",
                                          data_dir = "data") {
  fs::dir_create(output_dir)
  fs::dir_create(file.path(data_dir, "filter_data"))
  
  res_list <- list(hu35ksuba = res_hu35ksuba, hu6800 = res_hu6800)
  
  all_probe_data <- purrr::map_dfr(names(geo_chip_map), function(chip_name) {
    chip_data <- res_list[[chip_name]]
    summary_data <- chip_data$`__summary__`$probe_hits_by_group
    
    if (is.null(summary_data)) {
      warning(paste("No probe summary for chip:", chip_name))
      return(NULL)
    }
    
    tibble::tibble(
      chip = chip_name,
      group_comparison = names(summary_data),
      n_probes = lengths(summary_data)
    ) |>
      dplyr::mutate(
        group = tolower(sub(":.*", "", group_comparison)),
        comparison = sub("^[^:]+:", "", group_comparison),
        comparison = sub("^:+", "", comparison)
      )
  })
  
  wide_df <- all_probe_data |>
    tidyr::pivot_wider(
      id_cols = c(group, comparison),
      names_from = chip,
      values_from = n_probes,
      values_fill = NA
    ) |>
    dplyr::arrange(group, comparison)
  
  # Save wide summary
  readr::write_csv(
    wide_df,
    file.path(data_dir, "filter_data", "filtered_probe_counts_by_comparison.csv")
  )
  
  # Generate HTML tables per group
  cancer_groups <- c("carcinomas", "blastomas", "leukemias", "lymphomas")
  out_paths <- purrr::map_chr(cancer_groups, function(grp) {
    df <- wide_df |> dplyr::filter(group == grp)
    
    html_text <- if (nrow(df) == 0) {
      paste0("<p><em>No probe counts for ", stringr::str_to_title(grp), ".</em></p>")
    } else {
      tbl <- df |> 
        dplyr::select(Comparison = comparison, tidyselect::all_of(names(geo_chip_map))) |>
        dplyr::rename_with(
          ~ paste("Number of Filtered Probes", paste0("(", geo_chip_map[.], ")")),
          .cols = names(geo_chip_map)
        )
      
      html_table <- kableExtra::kbl(
        tbl,
        format = "html",
        align = c("l", rep("r", ncol(tbl) - 1)),
        na = "NA"
      ) |>
        kableExtra::kable_styling(
          bootstrap_options = c("striped", "hover"),
          full_width = TRUE
        ) |>
        as.character()
      
      paste0("<h4>", stringr::str_to_title(grp), "</h4>\n", html_table)
    }
    
    out_file <- file.path(output_dir, paste0(grp, "_probe_counts.html"))
    writeLines(html_text, out_file)
    out_file
  })
  
  names(out_paths) <- cancer_groups
  invisible(out_paths)
}
