# Structural Validation Layer

This module runs after structural synthesis and before biological interpretation.

It builds cross-engine calibration and discordance mapping outputs from:

```sql
vw_structural_phenotype_wide_with_latent_overlay
```

## Scripts

| Script | Purpose |
|---|---|
| `run_structural_validation.R` | Main runner |
| `01_build_cross_engine_calibration.R` | Pairwise Spearman calibration among canonical metrics |
| `02_build_engine_discordance_map.R` | Directional sign calls and engine agreement fractions |
| `03_summarize_discordance_archetypes.R` | Discordance archetype classification and summary views |

## DuckDB outputs

- `cross_engine_metric_correlations`
- `cross_engine_directional_concordance`
- `cross_engine_discordance_archetypes`

## DuckDB views

- `vw_cross_engine_discordance_summary`
- `vw_cross_engine_archetype_summary`
- `vw_cross_engine_discordant_runs`

## CSV outputs

Written to:

```text
output/global_cancer/structural_inference/tables/validation
```

## Run

From the project root:

```r
source("R/structural_validation/run_structural_validation.R")
```
