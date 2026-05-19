# ==============================================================================
# Plot Cross-Regime Structural Trajectories
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
# 1. Load
# ==============================================================================

plots <- load_robustness_outputs()

boundary_assignments <- plots$boundary_assignments

plot_dir <- create_plot_dir()


# ==============================================================================
# 2. Plot
# ==============================================================================

plot_df <- boundary_assignments %>%
  dplyr::filter(
    !is.na(robustness_x),
    !is.na(robustness_y)
  ) %>%
  dplyr::arrange(
    comparison,
    chip,
    factor(
      filter_regime,
      levels = c("limma", "variance_global", "variance_comparison")
    )
  )

missing_df <- boundary_assignments %>%
  dplyr::filter(
    is.na(robustness_x) |
      is.na(robustness_y)
  )

if (nrow(missing_df) > 0) {
  message("Unavailable trajectory points excluded:")
  print(as.data.frame(missing_df))
}

p <- plot_df %>%
  ggplot(
    aes(
      x = robustness_x,
      y = robustness_y,
      group = interaction(comparison, chip),
      color = filter_regime
    )
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.4) +
  geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.4) +
  geom_path(
    linewidth = 0.8,
    alpha = 0.7,
    arrow = arrow(length = unit(0.10, "inches"))
  ) +
  geom_point(size = 2.5) +
  facet_wrap(~ comparison, scales = "free") +
  labs(
    title = "Cross-Regime Structural Trajectories",
    subtitle = "Filtering-regime perturbation paths within structural state-space",
    x = "Scaled Complexity Axis",
    y = "Scaled Spectral Entropy Axis",
    color = "Filter Regime"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )


# ==============================================================================
# 3. Save
# ==============================================================================

ggsave(
  filename = file.path(
    plot_dir,
    "cross_regime_trajectories.png"
  ),
  plot = p,
  width = 13,
  height = 9,
  dpi = 300
)

ggsave(
  filename = file.path(
    plot_dir,
    "cross_regime_trajectories.pdf"
  ),
  plot = p,
  width = 13,
  height = 9
)


print(p)