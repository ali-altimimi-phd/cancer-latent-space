# Structural Phenotype Synthesis Design

## Purpose

The structural synthesis layer converts engine-specific outputs from MP spectral analysis, complexity analysis, entropy analysis, and optional latent-space geometry into a unified descriptor table. The goal is not only to summarize tumor-versus-normal behavior, but also to preserve the provenance of the feature space used to define that behavior.

This matters because the structural engines can now be run under multiple probe-selection regimes, while the current latent-space model remains tied to a single feature space: global variance top 3000 probes, currently on `hu35ksuba`.

## Core design principle

Do not treat all feature spaces as interchangeable.

A structural metric estimated on one probe-selection regime is valid within that regime. A latent metric is only directly comparable to structural metrics computed on the same feature space used to train or project the latent model.

Therefore, synthesis is split into three products:

1. all-regime structural synthesis,
2. latent-aligned synthesis,
3. regime-comparison synthesis.

## Product 1: all-regime structural synthesis

Files:

- `structural_phenotype_long.csv`
- `structural_phenotype_wide.csv`
- `structural_phenotype_heatmap_long.csv`
- `structural_phenotype_summary.csv`
- `structural_phenotype_tables.rds`, object `all_regimes`

This is the full descriptor-first synthesis. It includes all available structural regimes requested by the pipeline, such as:

- `limma`
- `variance_top3000`
- future comparison-local limma regimes
- future comparison-local variance regimes

This table is used to ask:

> What structural phenotype is observed under each feature-selection regime?

It is appropriate for MP, entropy, and complexity comparisons across feature-selection logic. It is also the correct source for robustness and sensitivity analysis.

## Product 2: latent-aligned synthesis

Files:

- `structural_phenotype_latent_aligned_long.csv`
- `structural_phenotype_latent_aligned_wide.csv`
- `structural_phenotype_latent_aligned_heatmap_long.csv`
- `structural_phenotype_latent_aligned_summary.csv`
- `structural_phenotype_tables.rds`, object `latent_aligned`

This product restricts the synthesis to the feature-space provenance compatible with the current VAE/Python latent layer.

Current defaults:

- `latent_chip_id = "hu35ksuba"`
- `latent_filter_regime = "variance_top3000"`
- `latent_aligned_chips = "hu35ksuba"`
- `latent_aligned_filter_regimes = "variance_top3000"`

This table is used to ask:

> How do MP, entropy, complexity, and latent geometry agree or diverge in the same feature space?

This is the only synthesis product that should be used for direct structural-versus-latent interpretation until additional VAE models are trained on the other regimes.

## Product 3: regime-comparison synthesis

File:

- `structural_phenotype_regime_comparison_wide.csv`
- `structural_phenotype_tables.rds`, object `regime_comparison`

This product excludes latent metrics and pivots structural metrics across filter regimes. It is designed for stability analysis.

This table is used to ask:

> Is the inferred tumor structural phenotype stable under limma versus variance selection, and under global versus comparison-local probe spaces?

This is a methodological validation layer, not a biological interpretation layer by itself.

## Interpretation rules

### Rule 1: latent comparison requires feature-space alignment

Latent metrics should only be compared directly against structural metrics from the same chip and filter regime used by the latent model. For the current model, this means `hu35ksuba + variance_top3000`.

### Rule 2: local regimes are robustness evidence

Comparison-local limma or comparison-local variance regimes are valuable because they test whether a structural conclusion depends on a global probe set. However, they should not be merged with the current VAE metrics unless a corresponding local-regime VAE is trained.

### Rule 3: all-regime synthesis is descriptive, not automatically integrative

The all-regime table is useful because it preserves all structural measurements in one standardized schema. But interpretation should still respect `chip`, `filter_regime`, `comparison`, `engine`, and `metric` provenance.

### Rule 4: disagreement across regimes is informative

If a comparison changes quadrant, direction, or rank across regimes, that does not necessarily indicate a failure. It may indicate that the structural behavior is scale-sensitive, feature-selection-sensitive, or dominated by a subset of comparison-specific probes.

## Recommended reporting layout

A future Quarto/reporting section can use the synthesis products as follows:

1. **Primary structural phenotype**  
   Use the all-regime structural synthesis to summarize MP, entropy, and complexity by chip and filter regime.

2. **Latent-space overlay**  
   Use only the latent-aligned synthesis to compare structural descriptors against VAE geometry.

3. **Feature-selection robustness**  
   Use the regime-comparison synthesis to identify stable versus regime-sensitive structural phenotypes.

4. **Biological interpretation**  
   Defer GO/KEGG/Hallmark interpretation until structural behavior is classified and regime stability is known.

## Practical notes

The modified synthesis script remains backward-compatible. Existing objects remain available at the top level of `structural_phenotype_tables.rds`:

- `long`
- `wide`
- `heatmap_long`
- `summary`

The new structured objects are added alongside these aliases:

- `all_regimes`
- `latent_aligned`
- `regime_comparison`
- `metadata`

This allows existing reporting code to continue working while newer reporting code can explicitly select the appropriate synthesis product.
