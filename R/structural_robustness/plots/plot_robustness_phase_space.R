# ==============================================================================
# Plot Robustness Phase Space
# ==============================================================================
#
# Purpose:
#   Visualize structural robustness geometry.
#
# Interpretation:
#
#   x-axis:
#     mean quadrant margin
#
#   y-axis:
#     maximum trajectory length
#
#   lower-right:
#     deep structural attractors
#
#   lower-left:
#     stable boundary systems
#
#   upper regions:
#     preprocessing/platform-sensitive systems
#
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(ggrepel)
  library(readr)
  library(here)
})

source(here::here(
  "R",
  "structural_robustness",
  "plots",
  "plot_helpers.R"
))


# ==============================================================================
# 1. Load data
# ==============================================================================

plots <- load_robustness_outputs()

robustness_summary <- plots$robustness_summary

plot_dir <- create_plot_dir()


# ==============================================================================
# 2. Build plot
# ==============================================================================

plot_df <- robustness_summary %>%
  dplyr::filter(
    !is.na(mean_quadrant_margin),
    !is.na(max_trajectory_length)
  )

missing_df <- robustness_summary %>%
  dplyr::filter(
    is.na(mean_quadrant_margin) |
      is.na(max_trajectory_length)
  )

if (nrow(missing_df) > 0) {
  message("Unavailable robustness cases excluded from phase-space plot:")
  print(as.data.frame(missing_df))
}

p <- plot_df %>%
  ggplot(
    aes(
      x = mean_quadrant_margin,
      y = max_trajectory_length,
      color = robustness_class
    )
  ) +
  geom_hline(
    yintercept = 0.75,
    linetype = "dashed",
    alpha = 0.5
  ) +
  geom_vline(
    xintercept = 0.5,
    linetype = "dashed",
    alpha = 0.5
  ) +
  geom_point(
    size = 4,
    alpha = 0.9
  ) +
  ggrepel::geom_text_repel(
    aes(label = comparison),
    size = 4,
    max.overlaps = Inf
  ) +
  labs(
    title = "Structural Robustness Phase Space",
    subtitle = paste(
      "Margin-aware robustness geometry across",
      "chips and filtering regimes"
    ),
    x = "Mean Quadrant Margin",
    y = "Maximum Structural Trajectory Length",
    color = "Robustness Class"
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "right"
  )


# ==============================================================================
# 3. Save
# ==============================================================================

ggsave(
  filename = file.path(
    plot_dir,
    "robustness_phase_space.png"
  ),
  plot = p,
  width = 12,
  height = 8,
  dpi = 300
)

ggsave(
  filename = file.path(
    plot_dir,
    "robustness_phase_space.pdf"
  ),
  plot = p,
  width = 12,
  height = 8
)


# ==============================================================================
# 4. Print
# ==============================================================================

print(p)