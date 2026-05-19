# Structural Concordance Layer

This module is downstream of the structural synthesis layer.

It does **not** recompute MP, complexity, entropy, or latent-space metrics. Instead, it reads the DuckDB synthesis view:

```r
vw_structural_phenotype_wide_with_latent_overlay
```

and builds:

| Table | Purpose |
|---|---|
| `canonical` | One row per `chip × filter_regime × group × comparison` with harmonized structural metrics |
| `quadrant_assignments` | Four-quadrant structural classification with distance/confidence measures |
| `engine_correlations` | Pairwise cross-engine metric concordance |
| `quadrant_stability` | Reproducibility of quadrant assignment across chips and filter regimes |
| `concordance_inventory` | Row counts and provenance for the concordance layer |

By default, the quadrant plane is:

```r
x_metric = "complexity__effrank_delta"
y_metric = "mp__spectral_entropy_delta"
```

This can be changed in `run_structural_concordance.R` if the final formal quadrant definition uses a different complexity coordinate.

## Run

From the project root:

```r
source("R/structural_concordance/run_structural_concordance.R")
```

## Outputs

CSV outputs:

```text
output/global_cancer/structural_inference/tables/concordance/tables/
```

RDS output:

```text
output/global_cancer/structural_inference/tables/concordance/RData/structural_concordance_tables.rds
```

If `materialize_to_duckdb = TRUE`, the following DuckDB tables are also created:

```text
structural_concordance_canonical
structural_quadrant_assignments
structural_engine_correlations
structural_quadrant_stability
structural_concordance_inventory
```

## Intended next use

After this layer is stable, biological interpretation should attach GO/KEGG/Hallmark/GO-semantic summaries to quadrant/archetype assignments rather than interpreting gene sets independently of structural phenotype.
