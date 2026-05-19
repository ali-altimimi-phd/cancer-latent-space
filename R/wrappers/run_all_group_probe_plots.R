#' Generate bar and UpSet plots for each cancer group
#'
#' For each group (e.g. carcinomas, leukemias), this wrapper creates:
#' 1. A grouped bar plot of filtered probe counts by chip
#' 2. An UpSet plot showing overlap of filtered probes across comparisons
#'
#' @param probe_df Data frame of filtered probe counts (from collect_probe_counts)
#' @param output_dir Where to save output PNGs (default = "quarto/resources/plots/probes/groups/")
#' @param plot_utils_path Path to helper file containing `save_png_light()`
#'
#' @return NULL. All plots saved to disk.
#' @export
run_all_group_probe_plots <- function(probe_df,
                                      output_dir = "quarto/resources/plots/probes/groups",
                                      plot_utils_path = "R/helpers/plot_utils.R") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  groups <- unique(probe_df$group)
  
  for (grp in groups) {
    grp_df <- dplyr::filter(probe_df, group == grp)
    
    # -------------------
    # 1. Bar Plot
    # -------------------
    bar_plot <- plot_probe_counts_by_group(grp_df)
    
    bar_file <- file.path(output_dir, paste0(tolower(gsub("\\W+", "_", grp)), "_bar_plot.png"))
    save_png_light(bar_plot, filename = bar_file, width = 10, height = 6)
    
    # -------------------
    # 2. UpSet Plot
    # -------------------
    all_upset_df <- collect_filtered_probe_sets(chips = unique(probe_df$chip))
    upset_df <- dplyr::filter(all_upset_df, group == grp)
    
    if (nrow(upset_df) >= 2) {
      upset_plot <- plot_upset_probe_overlap_by_group(upset_df)
      
      upset_file <- file.path(output_dir, paste0(tolower(gsub("\\W+", "_", grp)), "_upset_plot.png"))
      save_png_light(upset_plot, filename = upset_file, width = 8, height = 5)
    } else {
      message("Skipping UpSet plot for group '", grp, "' (less than 2 comparisons)")
    }
  }
}
