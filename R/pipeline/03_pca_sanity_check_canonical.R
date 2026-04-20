#' Run PCA Sanity Checks on the VAE Input Matrix
#'
#' @description
#' Loads a VAE-ready expression matrix and matching canonical metadata, verifies
#' sample alignment, removes pooled normal samples, computes principal
#' components on cancer samples only, reports explained variance, and saves both
#' static and interactive diagnostic plots.
#'
#' @details
#' This version expects canonical exported metadata from `02_extract_vae_input.R`
#' and uses `disease_clean` directly rather than deriving labels from raw GEO
#' columns.
#'
#' It saves three outputs:
#' \itemize{
#'   \item a cancer-only PCA scatterplot,
#'   \item a scree plot with cumulative variance explained,
#'   \item an interactive Plotly PCA scatterplot with dynamic legend filtering.
#' }
#'
#' Logging is handled through the shared pipeline logger helper located in
#' `R/helpers/pipeline_logger.R`. When run standalone, the script initializes
#' its own stage-specific logger.
#'
#' @section Inputs:
#' \itemize{
#'   \item `projects/cancer-latent-space/data/processed/hu35ksuba_vae_input.rds`
#'   \item `projects/cancer-latent-space/data/processed/hu35ksuba_metadata.rds`
#' }
#'
#' @section Outputs:
#' \itemize{
#'   \item `projects/cancer-latent-space/output/plots/preprocessing/pca_cancer_only.png`
#'   \item `projects/cancer-latent-space/output/plots/preprocessing/pca_scree.png`
#'   \item `projects/cancer-latent-space/output/plots/preprocessing/pca_cancer_only_interactive.html`
#'   \item `projects/cancer-latent-space/output/logs/03_pca_sanity_check.log`
#' }
#'
#' @keywords internal
#' @noRd

suppressPackageStartupMessages({
  library(here)
  library(ggplot2)
  library(Polychrome)
  library(plotly)
  library(htmlwidgets)
})

source(here::here("R/helpers/pipeline_logger.R"))

log_dir <- here::here("projects", "cancer-latent-space", "output", "logs")
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

if (!exists("logger")) {
  logger <- start_log(logfile = file.path(log_dir, "03_pca_sanity_check.log"))
}

input_x <- here("projects", "cancer-latent-space", "data", "inputs", "hu35ksuba_vae_input.rds")
input_meta <- here("projects", "cancer-latent-space", "data", "inputs", "hu35ksuba_metadata.rds")

out_dir <- here("projects", "cancer-latent-space", "output", "plots", "preprocessing")
output_plot <- file.path(out_dir, "pca_cancer_only.png")
output_scree <- file.path(out_dir, "pca_scree.png")
output_html <- file.path(out_dir, "pca_cancer_only_interactive.html")

logger$log(sprintf("Loading VAE matrix: %s", input_x), section = "LOAD")
X <- readRDS(input_x)

logger$log(sprintf("Loading metadata: %s", input_meta), section = "LOAD")
meta <- readRDS(input_meta)

logger$log(sprintf("Matrix dimensions: %s", paste(dim(X), collapse = " x ")), section = "DATA")
logger$log(sprintf("Metadata dimensions: %s", paste(dim(meta), collapse = " x ")), section = "DATA")

if (!identical(rownames(X), rownames(meta))) {
  logger$log("Sample alignment check failed between matrix and metadata.", section = "ERROR")
  stop("Sample alignment check failed between matrix and metadata.", call. = FALSE)
}

logger$log("Sample alignment check passed.", section = "CHECK")

required_cols <- c("disease_clean", "condition", "tissue_label")
missing_cols <- setdiff(required_cols, colnames(meta))
if (length(missing_cols) > 0) {
  logger$log(
    sprintf("Required metadata columns not found: %s", paste(missing_cols, collapse = ", ")),
    section = "ERROR"
  )
  stop(
    sprintf("Required metadata columns not found: %s", paste(missing_cols, collapse = ", ")),
    call. = FALSE
  )
}

label_col <- "disease_clean"
normal_label <- "normal"

keep_idx <- !is.na(meta[[label_col]]) & meta[[label_col]] != normal_label

logger$log(
  sprintf(
    "Cancer-only subset retained %s of %s samples; excluded %s pooled normal samples.",
    sum(keep_idx), nrow(meta), sum(!keep_idx)
  ),
  section = "FILTER"
)

X_cancer <- X[keep_idx, , drop = FALSE]
meta_cancer <- meta[keep_idx, , drop = FALSE]

if (!identical(rownames(X_cancer), rownames(meta_cancer))) {
  logger$log("Sample alignment check failed after cancer-only filtering.", section = "ERROR")
  stop("Sample alignment check failed after cancer-only filtering.", call. = FALSE)
}

pca_result <- logger$timed("Run cancer-only PCA", {
  X_scaled <- scale(X_cancer)
  pca <- prcomp(X_scaled, center = FALSE, scale. = FALSE)
  var_explained <- pca$sdev^2 / sum(pca$sdev^2)
  cum_var_explained <- cumsum(var_explained)
  list(
    pca = pca,
    var_explained = var_explained,
    cum_var_explained = cum_var_explained
  )
})

pca <- pca_result$pca
var_explained <- pca_result$var_explained
cum_var_explained <- pca_result$cum_var_explained

logger$log(
  sprintf("Top variance explained: %s", paste(round(var_explained[1:10], 4), collapse = ", ")),
  section = "RESULT"
)

build_pca_plot_data <- function(pca, meta, label_col) {
  if (!(label_col %in% colnames(meta))) {
    stop(paste("Column not found:", label_col), call. = FALSE)
  }

  pca_df <- data.frame(
    PC1 = pca$x[, 1],
    PC2 = pca$x[, 2],
    sample = rownames(meta),
    label = as.factor(meta[[label_col]]),
    stringsAsFactors = FALSE
  )

  pca_df$tooltip <- paste(
    "Sample:", pca_df$sample,
    "<br>Type:", pca_df$label,
    "<br>PC1:", round(pca_df$PC1, 2),
    "<br>PC2:", round(pca_df$PC2, 2)
  )

  pca_df
}

make_pca_plot <- function(pca_df, var_explained) {
  set.seed(42)
  num_colors <- length(unique(pca_df$label))
  custom_palette <- createPalette(
    num_colors,
    c("#4285F4", "#EA4335", "#FBBC05")
  )
  names(custom_palette) <- unique(pca_df$label)

  ggplot(pca_df, aes(x = PC1, y = PC2, color = label, text = tooltip)) +
    geom_point(size = 2, alpha = 0.8) +
    coord_fixed() +
    scale_color_manual(values = as.vector(custom_palette)) +
    labs(
      title = "Cancer-Only PCA of Expression Data (hu35ksuba)",
      subtitle = paste("Dimensionality reduction of", num_colors, "cancer categories"),
      x = paste0("PC1 (", round(var_explained[1] * 100, 1), "%)"),
      y = paste0("PC2 (", round(var_explained[2] * 100, 1), "%)"),
      color = "Cancer Type"
    ) +
    theme_minimal() +
    theme(
      legend.position = "right",
      legend.text = element_text(size = 8)
    )
}

make_scree_plot <- function(var_explained, cum_var_explained, n_components = 15) {
  n_show <- min(length(var_explained), n_components)
  scree_df <- data.frame(
    PC = seq_len(n_show),
    explained = var_explained[seq_len(n_show)] * 100,
    cumulative = cum_var_explained[seq_len(n_show)] * 100
  )

  ggplot(scree_df, aes(x = PC)) +
    geom_col(aes(y = explained)) +
    geom_line(aes(y = cumulative), linewidth = 0.8) +
    geom_point(aes(y = cumulative), size = 1.8) +
    scale_x_continuous(breaks = scree_df$PC) +
    labs(
      title = "Scree Plot of Cancer-Only PCA",
      subtitle = paste("Top", n_show, "principal components"),
      x = "Principal Component",
      y = "Variance Explained (%)"
    ) +
    theme_minimal()
}

logger$log(sprintf("Using metadata label column: %s", label_col), section = "PLOT")

pca_df <- build_pca_plot_data(pca, meta_cancer, label_col)
p_static <- make_pca_plot(pca_df, var_explained)
p_scree <- make_scree_plot(var_explained, cum_var_explained)
p_interactive <- ggplotly(p_static, tooltip = "text")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

logger$timed("Save cancer-only PCA plot", {
  ggsave(
    output_plot,
    plot = p_static,
    width = 10,
    height = 7,
    dpi = 300
  )
})

logger$timed("Save scree plot", {
  ggsave(
    output_scree,
    plot = p_scree,
    width = 9,
    height = 6,
    dpi = 300
  )
})

logger$timed("Save interactive PCA plot", {
  saveWidget(
    p_interactive,
    file = output_html,
    selfcontained = TRUE
  )
})

if (interactive()) {
  logger$log("Interactive session detected; rendering Plotly PCA to screen.", section = "PLOT")
  print(p_interactive)
}

logger$log(sprintf("Saved cancer-only PCA plot to: %s", output_plot), section = "SAVE")
logger$log(sprintf("Saved scree plot to: %s", output_scree), section = "SAVE")
logger$log(sprintf("Saved interactive PCA HTML to: %s", output_html), section = "SAVE")
