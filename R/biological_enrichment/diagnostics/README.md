---
title: "κ-Bootstrap Instability Technical Notes"
format:
  html:
    html-math-method: mathjax
---

# Technical Assessment of κ-Bootstrap Instability in the Structural Complexity Engine

## Overview

During diagnostic evaluation of the refactored structural complexity engine within the Global Cancer Structural Inference Framework, a pronounced instability was observed in bootstrap-derived confidence intervals and resampling distributions associated with \$\\kappa\$-based complexity metrics.

In contrast:

- entropy-based descriptors,

- effective-rank descriptors,

- and Marchenko–Pastur (MP) spectral descriptors

displayed comparatively stable bootstrap behavior under identical resampling conditions.

This prompted an investigation into whether the instability reflected:

1.  a programming or implementation error,

2.  an inappropriate resampling strategy,

3.  or an intrinsic mathematical property of condition-number statistics in high-dimensional transcriptomic spaces.

The following assessment summarizes:

- the observed results,

- code-review findings,

- mathematical interpretation,

- and recommended methodological refinements.

# 1. Observed Empirical Behavior

## Stable descriptors

The following descriptor families demonstrated relatively stable bootstrap and resampling behavior:

- Shannon entropy,

- spectral entropy,

- effective rank,

- MP spectral metrics,

- latent-space geometric metrics.

These metrics produced:

- comparatively smooth bootstrap distributions,

- reasonable confidence intervals,

- and acceptable convergence behavior under reduced bootstrap/permutation counts.

## Unstable descriptors

In contrast, the following descriptors exhibited pronounced instability:

- SVD-based condition number (\$\\kappa\$),

- covariance condition number,

- composite \$\\kappa\$ metrics.

Observed behaviors included:

- extremely wide bootstrap confidence intervals,

- heavy-tailed bootstrap distributions,

- large replicate-to-replicate variance,

- sensitivity to bootstrap sample composition,

- and occasional near-explosive condition-number values.

This instability persisted despite:

- correct matrix extraction,

- proper comparison-map alignment,

- reproducible seeds,

- and otherwise stable descriptor computation.

# 2. Code Review Findings

The reviewed implementation included:

- SVD-based condition-number computation,

- covariance-space conditioning,

- sample-level bootstrap resampling,

- permutation testing,

- and composite descriptor aggregation.

Reviewed files included:

- `core_complexity_metrics.R`,

- `statistical_complexity_helpers.R`,

- `compare_pair_complexity.R`,

- `run_pairwise_complexity.R`.

## Major conclusion of code review

No major programming defect or logical failure was identified.

Specifically, the review found no evidence of:

- label leakage,

- permutation corruption,

- transpose/orientation mistakes,

- matrix misalignment,

- malformed null distributions,

- resampling-seed collisions,

- covariance-orientation inconsistency,

- or improper matrix extraction.

The implementation was internally coherent and mathematically valid.

The instability therefore appears to arise primarily from:

- the mathematical properties of condition numbers,

- combined with the geometry of transcriptomic data,

- and amplified by bootstrap resampling.

# 3. Mathematical Explanation of the Instability

## 3.1 Condition-number behavior

The core SVD-based complexity metric is:

\\kappa = \\frac{\\sigma\_{\\max}}{\\sigma\_{\\min}}

where:

- \$\\sigma\_{\\max}\$ is the largest singular value,

- \$\\sigma\_{\\min}\$ is the smallest singular value.

Condition numbers are therefore dominated by the behavior of the smallest singular value.

## 3.2 High-dimensional transcriptomic geometry

Transcriptomic matrices operate in a:

[\
p \\gg n\
]

regime:

- probes/features greatly exceed samples,

- covariance structure is highly collinear,

- eigenspectra are heavy-tailed,

- and many spectral directions are weakly identified.

Under these conditions:

- matrices are already near rank-deficient,

- and \$\\sigma\_{\\min}\$ is often extremely small or poorly constrained.

Thus even minor perturbations can produce massive swings in:

[\
\\kappa\
]

## 3.3 Bootstrap-induced manifold collapse

The current bootstrap implementation performs sample-level resampling with replacement.

This necessarily:

- duplicates columns/samples,

- creates exact collinearities,

- removes leverage samples,

- reduces effective rank,

- and artificially pinches the local data manifold.

Consequently:

- \$\\sigma\_{\\min} \\to 0\$,

- while \$\\sigma\_{\\max}\$ remains comparatively stable.

This produces explosive behavior in:

[\
\\kappa\
]

The bootstrap therefore is not merely estimating ordinary statistical uncertainty.

Instead, it is effectively measuring:

- susceptibility to spectral degeneracy,

- local manifold collapse,

- and conditioning fragility.

This reinterpretation is critical.

# 4. Why Entropy and Effective Rank Remain Stable

Entropy and effective-rank metrics behave differently because they integrate over the entire eigenspectrum rather than depending on a single extreme eigenvalue.

Spectral entropy:

H = -\\sum p_i \\log p_i

Effective rank:

r\_{eff}=\\exp(H)

These statistics:

- average across spectral mass,

- are self-averaging,

- and are comparatively resistant to local singular perturbations.

Thus:

- entropy measures global spectral organization,

- while \$\\kappa\$ measures edge instability and degeneracy sensitivity.

The differing stability profiles therefore support — rather than weaken — the validity of the overall framework.

If the pipeline itself were fundamentally flawed, instability would be expected across all descriptor families simultaneously.

Instead, the observed pattern is precisely what random matrix theory and numerical linear algebra would predict.

# 5. Interpretation of κ Within the Structural Inference Framework

The original framing treated \$\\kappa\$ as a conventional complexity metric.

However, the observed behavior suggests that \$\\kappa\$ is more appropriately interpreted as a descriptor of:

- degeneracy susceptibility,

- conditioning fragility,

- anisotropic collapse,

- manifold instability,

- or spectral sensitivity.

This reframing aligns:

- the mathematics,

- the bootstrap behavior,

- and the biological interpretation.

Rather than representing a stable measure of “complexity,” \$\\kappa\$ may instead function as a sensitive detector of transcriptomic manifold collapse or near-singular organization.

# 6. Implementation Factors That Amplify Instability

Although no major coding errors were identified, several implementation choices likely amplify the intrinsic instability.

## 6.1 Sample-level bootstrap with replacement

Current behavior:

- duplicates samples,

- destabilizes covariance geometry,

- reduces effective rank.

This is likely the largest amplification factor.

## 6.2 Raw smallest-singular-value conditioning

The current implementation uses the raw smallest singular value directly:

\\kappa = \\frac{\\sigma\_{\\max}}{\\max(\\sigma\_{\\min},\\epsilon)}

where \$\\epsilon\$ is machine precision.

This creates strong sensitivity to numerical degeneracy.

## 6.3 Covariance-space conditioning

Covariance conditioning further amplifies instability because:

\\kappa(\\Sigma) \\approx \\kappa(X)\^2

Thus covariance-space condition numbers are inherently more unstable than raw matrix conditioning.

## 6.4 Internal rounding during metric computation

Metrics are currently rounded internally during computation rather than only during reporting.

Although not catastrophic, this:

- discretizes bootstrap distributions,

- increases ties,

- and slightly perturbs confidence-interval estimation.

# 7. Recommended Methodological Refinements

## 7.1 Promote entropy/effective-rank descriptors to the primary inferential layer

These descriptors demonstrated:

- superior stability,

- better convergence,

- and stronger large-scale geometric interpretability.

Recommended primary structural descriptors:

- spectral entropy,

- Shannon entropy,

- effective rank,

- MP spectral metrics,

- latent geometric metrics.

# 7.2 Reposition κ as a secondary sensitivity layer

Rather than abandoning \$\\kappa\$, reinterpret it as a:

- degeneracy index,

- conditioning-instability metric,

- or manifold-fragility descriptor.

This transforms instability from a nuisance into a biologically meaningful signal.

# 7.3 Use logarithmic condition numbers

Recommended transformation:

\\log\_{10}(\\kappa)

Advantages:

- compresses extreme tails,

- stabilizes variance,

- improves interpretability,

- and aligns more naturally with multiplicative biological scales.

# 7.4 Introduce regularized or truncated condition numbers

Recommended replacement:

\\kappa\_{reg}=\\frac{\\sigma_1+\\delta}{\\sigma_k+\\delta}

where:

- \$\\delta\$ is a small regularization parameter,

- or \$k\$ is chosen using effective-rank truncation.

This prevents tiny unstable singular values from dominating the metric.

# 7.5 Effective-rank-based truncation

One promising strategy is:

[\
k \\approx r\_{eff}\
]

yielding:

\\kappa\_{trunc}=\\frac{\\sigma_1}{\\sigma\_{r\_{eff}}}

This links:

- entropy,

- effective rank,

- and conditioning

into a unified geometric framework.

Potential advantages:

- improved stability,

- biologically meaningful spectral truncation,

- reduced numerical degeneracy sensitivity.

# 7.6 Replace ordinary bootstrap with subsampling

Instead of:

- bootstrap with replacement,

consider:

- \$k\$-out-of-\$n\$ subsampling without replacement.

This better preserves:

- covariance geometry,

- local manifold structure,

- and effective rank.

# 8. Emerging Structural Interpretation Framework

The observed results suggest a natural hierarchy of structural descriptors.

| Descriptor Family   | Structural Role                        |
|---------------------|----------------------------------------|
| MP spectral metrics | Bulk covariance organization           |
| Entropy metrics     | Global spectral dispersion             |
| Effective rank      | Occupied dimensionality                |
| Latent geometry     | Nonlinear manifold organization        |
| κ / truncated κ     | Degeneracy and instability sensitivity |

This emerging architecture is internally coherent and mathematically defensible.

# 9. Final Conclusion

The observed \$\\kappa\$ bootstrap instability does not appear to reflect a major implementation error.

Rather, it emerges from:

- the intrinsic behavior of condition numbers,

- the geometry of high-dimensional transcriptomic matrices,

- and the interaction between bootstrap resampling and near-singular spectral structure.

The instability therefore appears to be:

- mathematically expected,

- biologically interpretable,

- and potentially scientifically informative.

Within the evolving Global Cancer Structural Inference Framework:

- entropy and MP descriptors appear well-suited for stable primary structural inference,

- while \$\\kappa\$ and related conditioning metrics may serve as specialized indicators of transcriptomic degeneracy, manifold collapse, and conditioning fragility.

The resulting framework is substantially richer than a single-metric “complexity” interpretation and supports a multi-layered view of transcriptomic structural organization.
