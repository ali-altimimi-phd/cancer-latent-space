# ==============================================================================
# Structural Robustness Plot Helpers
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(ggrepel)
  library(here)
})


load_robustness_outputs <- function(study_name = "global_cancer") {
  
  base_dir <- here::here(
    "output",
    study_name,
    "structural_inference",
    "tables",
    "robustness"
  )
  
  list(
    robustness_summary = readr::read_csv(
      file.path(base_dir, "structural_robustness_summary.csv"),
      show_col_types = FALSE
    ),
    
    trajectory_summary = readr::read_csv(
      file.path(base_dir, "structural_robustness_trajectories.csv"),
      show_col_types = FALSE
    ),
    
    boundary_assignments = readr::read_csv(
      file.path(base_dir, "structural_robustness_boundary_assignments.csv"),
      show_col_types = FALSE
    )
  )
}


create_plot_dir <- function(study_name = "global_cancer") {
  
  outdir <- here::here(
    "output",
    study_name,
    "structural_inference",
    "plots",
    "robustness"
  )
  
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  
  outdir
}