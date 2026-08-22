# X-Trans windowed alias-hypothesis classification

## Decision

**NO-GO as the primary route to an alias-aware demosaicer.** Neighbor context
occasionally chooses a better sparse approximation, but the gain is supplied
by generic support/color continuity rather than the predicted X-Trans Fourier
phase trajectory. It does not recover the missing saturated red/gray proposal,
it worsens the dense sweep, and its confidence score is not calibrated. Most
decisively, the known exact luma/chroma null remains exact and spatially
coherent after stacking both CFA-aligned and non-aligned translated windows.

This result does not invalidate the preceding finding that complete replica
constellations are more informative than isolated peaks. It shows that local
translation consistency does not supply enough *new* information to select a
unique RGB source once materially different sparse explanations survive.

No RawTherapee demosaicing or GUI code is changed by this experiment.

## Experimental contract

The implementation imports the exact 18-carrier lattice, 18x54 operator,
orthonormal `L/C1/C2` basis, family ordering, least-squares fit, and OMP
machinery from `tools/xtrans_alias` and `tools/xtrans_sparse_alias`. For each
window/family it generates fixed-cardinality support proposals with a limited
OMP beam plus forced-exclusion alternatives. Hypotheses record support,
coefficients, residual, energy, singular values, minimum singular value, and
condition number.

Primary analysis uses:

- 48x48 windows;
- separable square-root periodic Hann analysis taper;
- a 12-pixel shift and four-neighbor connectivity;
- maximum support K=5;
- eight materially distinct proposals from beam width eight;
- exact CFA-phase dictionaries, including non-six-pixel origins.

The shifted sampling equation was checked directly for aligned `(0,0)` and
`(6,6)` origins and the non-aligned `(7,5)` origin; maximum numerical error is
below `5e-17`.

## Hypothesis generation

[Proposal-survival plot](images/xtrans-alias-context/proposal-survival.png)

Exact top-K truth survival among local proposals is:

| Scene | top 1 | top 3 | top 5 | top 8 | near-equal proposals |
| --- | ---: | ---: | ---: | ---: | ---: |
| vertical red/green | 0% | 6.7% | 33.3% | 100% | 100% |
| saturated red/gray | 0% | 0% | 0% | 0% | 93.3% |
| diagonal red/green | 0% | 91.1% | 100% | 100% | 100% |
| saturated quadrant | 13.3% | 13.3% | 37.8% | 42.2% | 100% |
| dense frequency sweep | 48.9% | 68.9% | 71.1% | 86.7% | 86.7% |
| windowed half-carrier | 0% | 0% | 0% | 0% | 100% |

“Near equal” means the second residual is within 0.01 of the best. The mean
best/second support Jaccard is about 0.45–0.67 depending on the scene, so these
are genuinely different explanations rather than support permutations.

The primary saturated red/gray failure occurs before contextual selection:
the true dominant five-atom support is absent from all eight proposals. Beam
widths 4, 8, and 16 do not change this. On a representative edge family K=3
is present at rank one for every beam width, while K=4 and K=5 are absent even
at N=8. Context cannot recover a proposal that local generation discarded.

## Context terms and ablations

[Context-ablation plot](images/xtrans-alias-context/context-ablation.png)

Mean coefficient error (lower is better):

| Scene | local | support only | phase only | support + phase | all terms |
| --- | ---: | ---: | ---: | ---: | ---: |
| vertical red/green | 0.3358 | 0.1327 | 0.1329 | 0.1327 | 0.1327 |
| saturated red/gray | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| diagonal red/green | 0.0996 | 0.0996 | 0.0996 | 0.0996 | 0.0996 |
| saturated quadrant | 0.3317 | **0.1176** | 0.2349 | 0.1392 | 0.1838 |
| dense frequency sweep | **0.0931** | 0.1022 | 0.1022 | 0.1022 | 0.1022 |
| windowed half-carrier | 0.8087 | 1.1725 | 0.6670 | 1.2531 | 1.2530 |

Color-only reaches 0.0685 on the sweep, but that is an approximation-error
accident rather than correct source identification. The full contextual score
is not the best contextual method on the quadrant, and the phase term supplies
no independent gain there. Iteration also is not reliably better: on the
quadrant its error is 0.1988 versus 0.1838 for one-pass reranking, and on the
vertical edge it loses the one-pass improvement and returns to 0.3358.

For the saturated quadrant, a separate 24-pixel-shift grid distinguishes long
edges from the corner. Full context reduces long-edge error from 0.5301 to
0.3303, but leaves the corner unchanged at 0.0968. This supports the narrow
claim that neighboring support agreement can help coherent edges. Support-only
selection nevertheless performs better, so the gain does not justify the
alias-specific phase machinery.

## Phase evolution and grouped control

For a known `(1/3,1/6)` source, the translation-predicted complex phase error
over 72 phases is at most `1.14e-15`; a deliberately wrong quarter-cycle
trajectory scores 0.5. Thus the phase formula and grouped-quadrature trajectory
are correct in isolation.

The complete windowed classifier still regresses the half-carrier scene. Hann
sidebands make its five-component local truth denser, the correct support is
not retained in the top eight, and the combined score rises from 0.8087 local
error to 1.2530. Phase-only improves to 0.6670, but cannot rescue the complete
selection pipeline. The prior 99.72% single-group recovery result itself is
unchanged; contextual proposal generation is the new failure.

## Intrinsic multi-window ambiguity

The known one-luma/four-chroma dependency uses five columns with single-window
rank four. Stacking a globally coherent sinusoidal model gives:

| Windows | count | rank | nullity | truth/competitor observation difference |
| --- | ---: | ---: | ---: | ---: |
| single | 1 | 4 | 1 | `7.58e-16` |
| shifts `(0,0),(6,0),(12,0),(0,6)` | 4 | 4 | 1 | `9.33e-16` |
| shifts `(0,0),(1,0),(0,1),(7,5)` | 4 | 4 | 1 | `1.02e-15` |

Both source interpretations obey the correct global phase evolution. Even a
non-CFA-aligned window does not add rank. The repeated windows add a scene
prior through consistency; they do not add independent measurement information
for this exact model. Therefore some single-window ambiguities remain exact
across spatial context, as the prompt required the experiment to establish.

## Window size, shift, noise, and propagation

[Window/shift plot](images/xtrans-alias-context/window-shift-ablation.png)

Window sizes 24, 48, and 96 were tested with shifts 6, 12, 24 where valid, plus
non-aligned shift 7. On the saturated red/gray diagnostic, full context selects
the same coefficients as local recovery in every configuration. Errors vary
from 0.0258 to 0.1887 because window size/placement changes spectral density,
not because context resolves the ambiguity. There is no evidence that 48x48
is uniquely favorable.

At 40 dB observation SNR, context remains neutral. At 30 and 20 dB it reduces
mean coefficient error relative to independent selection, but exact top-K
support recovery remains zero at every SNR. This is consistent with continuity
acting as a stabilizer rather than discovering the physical source. Injecting
one severely corrupted center window did not change the selected supports of
its neighbors in this small test; iterative selection did not propagate the
corruption, but it also did not recover correct supports.

The explicit support-transition scene changes from C1 to C2 at fixed spatial
frequency. Full context detects all 15 evaluated row/family transitions with
zero grid-step localization error and improves coefficient error from 0.4176
to 0.3667. Support-only obtains exactly the same result, while phase-only
regresses to 0.6934. The test prevents interpreting constant-support preference
as success, but again attributes the useful behavior to generic support
continuity rather than Fourier phase.

## Natural RGB windows

[Natural structure plot](images/xtrans-alias-context/natural-structure-recovery.png)

The same four authenticated scikit-image 0.26.0 sources are analyzed in linear
RGB. The study contains 144 selected high-energy family/window samples.

| Metric | value |
| --- | ---: |
| locally ambiguous families | 90.28% |
| plausible supports within residual 0.01 | 6.61 mean |
| local coefficient error | 0.09898 |
| full-context coefficient error | 0.09536 |
| oracle top-support error | 0.07380 |
| blind-to-oracle gap closed, G | 14.36% |
| ambiguous families improved | 3.08% |
| ambiguous families regressed | 0% |

The small aggregate improvement is real but sparse: 96.9% of ambiguous
families are unchanged. Approximate irregular-texture samples improve by
0.0104 coefficient error; corners improve by only 0.0008; single edges,
saturated chromatic edges, and periodic texture are unchanged.

Most importantly, support-only, color-only, support+phase, and the full score
all produce the same 0.09536 natural error. Phase-only is worse than local at
0.10231. Exact support rate remains 40.28% for every ablation. The natural gain
therefore comes from generic continuity/tie-breaking, not the physically
predicted phase observable.

The structure labels are deterministic gradient/color heuristics, not semantic
ground truth. They are useful only for approximate breakdown, and the corpus
remains four sRGB photographs rather than camera-linear RAW targets.

The exact dominant top-five support is present in the best 1/3/5/8 natural
proposals for 40.3%/61.1%/69.4%/75.0% of samples. Thus N=8 leaves a quarter of
the natural targets unreachable before context, while retaining many plausible
alternatives for the remaining samples.

## Confidence calibration

[Calibration plot](images/xtrans-alias-context/confidence-calibration.png) and
[spatial maps](images/xtrans-alias-context/spatial-confidence-maps.png)

The desired monotonic calibration fails. On natural windows, the highest 10%
by contextual confidence have 0.1090 mean coefficient error, worse than the
highest 25% (0.0848), highest 50% (0.0833), and even the full population
(0.0954). The high-confidence 10% also contain 28.6% incorrect top-K supports.

Synthetic confidence is likewise non-monotonic. The intrinsically dense sweep
has mean confidence 1.65 and the unresolved saturated red/gray edge 1.67,
both higher than the observable vertical red/green edge at 1.15. The classifier
therefore does exactly what the negative control forbids: it can be highly
confident in a coherent but non-identifiable explanation.

## Answers to the experiment questions

- **Is truth usually in top N?** Sometimes. Natural and scene-specific values
  are recorded for N=1/3/5/8, but saturated red/gray and the windowed
  half-carrier have 0% inclusion even at N=8. Increasing beam width does not
  fix their K=4/5 proposal failure.
- **How many plausible hypotheses?** Natural families average 6.61 of eight
  within 0.01 residual; 90.3% are locally ambiguous. Their supports are
  materially different.
- **Does support agreement help?** Yes on selected coherent edges and a small
  subset of irregular natural texture.
- **Does phase evolution help more?** No. Phase-only regresses natural error,
  and support-only matches or beats the full score wherever context helps.
- **Does color add independent value?** No consistent independent value is
  demonstrated; it often produces the same tie-break as support continuity.
- **Can saturated red/gray be resolved?** No. Its true top-five support never
  survives local proposal generation, and every tested context configuration
  leaves the result unchanged.
- **Which ambiguities remain intrinsic?** The explicit luma/four-chroma null
  remains rank deficient and exactly coherent across aligned and non-aligned
  translated windows. The dense sweep also remains wrong and overconfident.
- **Does context help natural images?** It closes 14.36% of the mean
  blind-to-oracle coefficient gap, but changes only 3.08% of ambiguous samples
  and supplies no phase-specific advantage.
- **Is confidence usable?** No. Error is non-monotonic with retained confidence,
  and the ambiguous negative controls can score higher than observable edges.

## Why this is NO-GO rather than PARTIAL

The numerical natural-window gain alone would merit PARTIAL if it identified a
new X-Trans-specific observable. It does not: support-only obtains the entire
gain, phase-only is worse, the primary ambiguous edge is not proposed, and the
exact contextual null proves that multiple globally coherent RGB sources can
remain measurement-identical. The confidence failure also prevents safely
routing only identifiable regions to this method.

Accordingly, do not build an alias-family demosaicer from this classifier. The
replica operator remains valuable as a diagnostic or as a feature inside a
method with genuinely additional information, but local support/phase
continuity should not be pursued as the primary inverse prior.

Because the classifier fails its family-level gates, image-space overlap-add
reconstruction and comparison with Markesteijn/B/C are deliberately not used
to manufacture a visually plausible result. The earlier image metrics remain
the relevant comparison: this experiment tests selection, and that selection
does not pass.

## Reproducibility

The canonical result is
`images/xtrans-alias-context/context-characterization.json`. Its SHA-256 is:

```text
fb2e4a0497cb88c7121a1bff81d31f7c613ba2b609995a5fd6c97eb1d2a3d765
```

The generator uses authenticated natural sources, fixed seeds, canonical JSON,
and deterministic unoptimized PNG output. It records no timestamp, host,
absolute path, or runtime. Native tests independently cover CFA phase
dictionaries, phase evolution, exact contextual null preservation, contextual
tie-breaking, and genuine support transitions.
