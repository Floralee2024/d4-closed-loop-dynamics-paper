# Claims and limitations

## Supported descriptive claims

1. D4a's selected entropy/future-MI proxy has a q90 landmark closer to normalized MSE utility than its usage/rank proxy in the eight reported cells. D4a trains once per seed/variant, not once per C. The null-adjusted diagnostic has lower average q90 error, and the slope-based comparison differs.
2. D4b nonzero-residual conditions have larger K-balanced gaps between the mixed-q rank proxy and the outcome/lower-state NMI association proxy on the archived per-C retraining grid.
3. D4b full-grid utility ranges are lower in nonzero-residual groups, but this includes a structural endpoint difference at C=0. Post-hoc exclusion of that endpoint does not preserve the direction across splits: OOD-inverted r=0.1 has a larger range than r=0.
4. The two studies use different control variables, models, proxies, utility scales and threshold estimators. Neither their marker values nor their scientific constructs are interchangeable.

## Measurement and aggregation boundaries

D4a uses unsmoothed min-to-max q90 with structural/utility gain gates of 0.02/0.10. D4b uses interior three-point smoothing and first-point-to-maximum q90 with gates 0.05/0.10. Boundary hits are observed-grid limitations, not estimates of an identified latent censored critical point.

D4b gap summaries average valid rows within K before equal weighting of K=8 and K=16. Event rates retain every planned row in the denominator and code undefined comparisons as no observed event. Valid-only pooled rates are separately labeled in the revision supplement. These are different estimands, not arithmetic corrections to the archived tables.

“Equivalence” is not directly tested. D4a's entropy term can reward collapsed symbol sequences. D4b's default dynamics score has no transition term; it combines separately curve-normalized outcome and lower-state NMI. Its rank proxy measures mixed q rather than the same discrete IDs. Utility range is max minus min, not local sensitivity.

## Claims not established

- A shared dynamics quantity across D4a and D4b.
- Useful dynamics learning or symbolic equivalence from proxy values alone.
- A universal critical constant, a sharp phase transition or estimator-independent utility prediction.
- A fixed-model residual intervention effect or demonstrated semantic-bypass mechanism.
- Universal reduction in utility range after removing C=0.
- Generalization to natural tasks or action-conditioned closed-loop control.

Per-C retraining is a valid protocol to describe training outcomes; it is not inherently invalid. A fixed-model experiment addresses a different estimand. Common random numbers and magnitude-matched residual controls would help discriminate interpretations but are not supplied by this revision.

## Reproducibility and remaining work

D4b raw curves, configs and provenance are available. D4a original raw curves and landmarks were recovered from the exact SOURCES.md path, with an alignment file byte-identical to the released table. All 960 checked q90 markers reproduce. These metric tables do not constitute sample-level reruns.

`results/revision_posthoc/` is explicitly exploratory. It provides endpoint sensitivity, denominator diagnostics and 2,000-resample base-seed bootstrap intervals. These intervals condition on the current world, grid and missing-threshold rule; they do not fix censoring, selection or multiplicity. Original data and training code remain unchanged.

D4a raw artifacts have been recovered without rerunning training. Of 160 utility q90 markers, 140 are at C=0; marker alignment must not be described as coincident emergence. Minimum recorded normalized usage entropy is 0.755, inconsistent with complete collapse in these evaluations. Any new mechanism experiment should prospectively define its measurement object and controls. The archived frozen-model probes show why a C-dependent readout is needed, but do not establish a causal mechanism or replicate the original proxies.
