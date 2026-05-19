# ==============================================================================
# Plot Boundary Margin Heatmap
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(here)
})

source(here::here(
  "R",
  "structural_robustness",
  "plots",
  "plot_helpers.R"
))


plots <- load_robustness_outputs()

boundary_assignments <- plots$boundary_assignments

plot_dir <- create_plot_dir()


p <- boundary_assignments %>%
  ggplot(
    aes(
      x = filter_regime,
      y = comparison,
      fill = quadrant_margin
    )
  ) +
  
  geom_tile(color = "white") +
  
  facet_wrap(~ chip) +
  
  labs(
    title = "Quadrant Boundary Margin Heatmap",
    subtitle = "Distance from structural phase boundaries",
    
    x = "Filter Regime",
    y = "Comparison",
    fill = "Quadrant Margin"
  ) +
  
  theme_bw(base_size = 12) +
  
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


ggsave(
  filename = file.path(
    plot_dir,
    "boundary_margin_heatmap.png"
  ),
  plot = p,
  width = 12,
  height = 8,
  dpi = 300
)

ggsave(
  filename = file.path(
    plot_dir,
    "boundary_margin_heatmap.pdf"
  ),
  plot = p,
  width = 12,
  height = 8
)


print(p)