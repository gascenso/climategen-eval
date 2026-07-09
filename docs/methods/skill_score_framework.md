# A skill-score framework for evaluating generative models in climate science

*Methods. This document defines a general, sample-size-stable framework for interpreting
suites of evaluation metrics for generative models in climate science. It is written to be
lifted into a paper's methods section: the formulation is domain-agnostic, validated on a
synthetic process with known failure modes, and instantiated on a tropical-cyclone
brightness-temperature case study.*

---

## 1. Problem and desiderata

A generative model for a climate field is typically assessed with a *suite* of metrics —
some measuring agreement of distributions (e.g. Wasserstein or Jensen–Shannon distances
between value histograms), others measuring domain-specific physical realism (e.g. radial
structure, spectral content, coherent features). In a mature benchmark setting one
interprets a metric value by comparison to a long history of prior results. In climate
science that history usually does not exist: the variable, region, or architecture is new,
and a raw number such as "the brightness-temperature histogram Wasserstein-1 distance is
2 K" is uninterpretable on its own.

We require a framework that:

1. **is interpretable from the dataset and a stated protocol** — it must say whether a
   value is close to physically realistic or far from it, without an external benchmark;
2. **rests on a principled theoretical basis**;
3. **applies to a heterogeneous suite of metrics**, from distributional distances to
   physical-realism functionals, under whatever criteria are appropriate for the domain;
4. **is cheap to adopt** — no model retraining, no bespoke losses, no held-out benchmark.

The central hazard the framework is designed around is the conflation of **statistical
significance** (*can we detect that the model is not the real process?*) with **effect
size** (*how far from real is the model?*). The first grows with the evaluation sample
size and tends to certainty; the second is an intrinsic property of the two distributions.
A trustworthy evaluation must report the second. The crux of the matter is establishing
what effect size can be deemed acceptable or "within physically plausible variability."

We are explicit at the outset about a boundary the framework respects (§13): it **does not
define realism**. It normalizes the evidence supplied by a chosen suite of realism probes
against interpretable endpoints. Its upper endpoint is **dataset-defined**; its lower
endpoint is **protocol-defined** through a pre-registered, interpretable baseline panel.

---

## 2. Significance versus effect size

Let `m` denote any lower-is-better metric — a non-negative pseudo-distance between two
samples or laws, with `m = 0` exactly when the property it probes is matched. (Higher-is-
better quantities are re-oriented in §10.) Let `P` be the real data law and `Q` the model
law, and define the **population discrepancy**

```
   m*(Q) = m(Q, P) ≥ 0,     with m*(Q) = 0  iff  Q matches P on the probed property.
```

`m*` is the *effect size*: an intrinsic, sample-size-independent property of the model.

We never observe `m*`; we observe an estimate `m̂(N)` from `N` generated samples and the
available real sample. For any consistent estimator,

```
   m̂_model(N) → m*_model       (a positive constant for any imperfect model),
```

while the **real-vs-real self-distance** — the same metric between two finite samples of
the *same* law `P` — satisfies

```
   m̂_real-vs-real(N) → 0        as N → ∞,
```

because a distribution has zero distance from itself; the finite-sample value is sampling
noise that vanishes at a metric-dependent rate `N^{−α}` (e.g. `α = 1/2` for one-dimensional
Wasserstein under a finite second moment; `α = 1` for an unbiased squared-MMD U-statistic).

Consequently, a test that asks *"is `m̂_model(N)` outside the real-vs-real fluctuation
band?"* returns **yes with probability → 1** for every imperfect model as `N` grows. That
is a correct test of `H₀: Q = P`, and it is useless as a *measure of quality*, because its
verdict is governed more by `N` than by how good the model is.

> **Stance.** Report a normalized **effect size** with interpretable endpoints — the
> data-defined perfect anchor and a protocol-defined naive anchor. Retain significance only
> in a subordinate role (§9), never as the scale.

---

## 3. Setup and the perfect-model anchor

Work class-conditionally where the task is conditional (e.g. per intensity class): all
quantities below are computed within a class and aggregated as a profile across classes
and metrics (§11). Fix an evaluation sample size `N` — the number of generated samples per
class — and compute every quantity at that same `N`.

For each metric `m`, the **real-vs-real null** is the distribution `M₀(N)` of `m̂` between
two independent size-`N` samples of `P`. It has two roles.

**Perfect-model anchor (skill = 1).** A model drawing exactly from `P` is, at the metric
level, indistinguishable from a second real sample; its expected finite-sample metric value
is the self-distance. We therefore take

```
   a₁(N) = median of M₀(N)
```

as the perfect anchor. This is the key reinterpretation underpinning the framework: the
real-vs-real distribution is **not the ruler** — it is the **finite-sample behaviour of a
perfect generator at sample size `N`**. Its *median* is the perfect anchor; its *spread*
is a significance quantity (§9), not a unit of the scale.

**Resolution ribbon.** Fix one global significance level `ε` (e.g. `ε = 0.05`),
pre-registered, and report sensitivity to `ε ∈ {0.01, 0.05, 0.10}` (§11). The upper edge of
the central null is the quantile `r(N) = Q_{1−ε}(M₀(N))` (lower edge `Q_ε` for higher-is-
better metrics). The metric-space band `[a₁(N), r(N)]` is the set of values **not resolvably
worse than real** at sample size `N`. The single level `ε` does double duty: it sets the
model's resolution band near skill 1 (§9) **and** the admissibility margin for baselines at
the bad end (§5). The band narrows as `N` grows, which is the correct behaviour — more data
resolves finer departures.

---

## 4. Naive baselines as the bad anchor

The lower anchor is supplied not by one universal "minimal model" — which does not exist
across heterogeneous metrics and use cases — but by a small, pre-registered **panel of
property-preserving nulls** `B`. Each baseline `b ∈ B` is a surrogate generator that
**preserves one named low-order or structural property of the real class-conditional
distribution while deliberately destroying others**, so its behaviour across the metric
suite is predictable from what it preserves. (We avoid the term "sufficient statistic":
for arbitrary climate-field laws these properties are not sufficient statistics in the
classical sense, which would require a specified parametric family.) Admissibility
conditions for the panel are given in §12. Each baseline yields a law `R_b` and an estimate
`m̂_b(N)`, computed by the *same* pipeline at the *same* `N` as the model.

**The panel as a basis for a failure-mode taxonomy.** Rather than claim universality, we
organize the panel around a taxonomy of the ways a generative climate model can fail, and
require the panel to **span** that taxonomy — at least one baseline that provably fails on
each axis:

| Failure axis (what a model can get wrong) | Anchoring null (preserves → destroys) |
|---|---|
| value marginals (1-point distribution) | a Gaussian/structural null that mismatches the marginal |
| second-order spatial structure (local covariance) | per-pixel-moment null (correct marginals, wrong dependence) |
| spectral content / global second-order structure | a spectrum-destroying null |
| coherent structures (organized features) | phase-randomized surrogate (keeps spectrum, destroys coherence) |
| conditioning / label fidelity | wrong-condition real samples (a *conditional-label null*, §12) |
| diversity / mode coverage | a low-diversity or deterministic null |

The role of a baseline is to define **the level of performance attainable on a given axis
without genuinely modelling that axis's property** — the "skill = 0" reference. The crucial
subtlety, developed next, is that a baseline can play this role **only for the metrics whose
property it does not preserve** — i.e. only on its own failure axes.

---

## 5. Per-metric admissibility and the reference anchor

A baseline must not anchor skill = 0 for a metric whose property it reproduces. If it did,
its metric value would sit at the floor, the bad and perfect anchors would coincide, the
scale would collapse, and — because a good model's value also sits near the floor — the
model would be scored at or below zero on an axis it may in fact match well. This is the
same divergence pathology as §8, entering from the baseline side, and it is a *false
penalty*, not merely a degenerate scale.

The resolution is that **admissibility is per metric**, and is judged by *resolvability*
against the real-vs-real null at the same level `ε` that defines the ribbon — not by a point
estimate, and not by an arbitrary effect-size threshold:

> **Admissibility (CI-separation).** Baseline `b` may anchor skill = 0 for metric `k` at
> sample size `N` iff `b` is **significantly worse than real on `k` at level `ε`** — i.e.
> the bootstrap distribution of `m̂_b(k,N)` is separated from the real-vs-real null
> `M₀(k,N)`, with `b`'s lower confidence edge lying outside the ribbon `[a₁, r]`.
> Equivalently, `b`'s population discrepancy `m*_b(k) > 0` and is resolvable at this `N`.
> Baselines whose interval overlaps the ribbon are "as good as real" on axis `k` — within
> sampling resolution — and are disqualified there.

This is parameter-free in the sense that matters: it introduces **no new knob beyond the
`ε`** already fixed for the ribbon, and it cannot be gamed — a baseline is either
statistically resolvable from real at the chosen `N` or it is not, and `ε` is global,
pre-registered, and reported with a sensitivity sweep (§11).

Define the **admissible set** `A_k(N) = { b ∈ B : b CI-separated from M₀(k,N) at level ε }`
and the **reference anchor** as the *strongest* admissible baseline — the hardest naive
competitor that still genuinely lacks the property:

```
   a₀(k, N) = min_{b ∈ A_k(N)}  m̂_b(k, N).
```

Four properties make this the right construction.

- **Conservative and non-gameable in selection.** Taking the minimum over admissible
  baselines anchors skill = 0 at the *strongest* naive model on each axis, so skill cannot
  be inflated by introducing a weak reference, and beating skill = 0 is a genuine
  achievement.
- **Denominator fragility is bounded, without a hyperparameter.** Because admissibility
  requires CI-separation, the selected `a₀` is separated from `a₁` by at least one
  `ε`-level of the metric's own sampling noise. The denominator `a₀ − a₁` therefore cannot
  be "infinitesimally positive"; its minimum margin *is* the natural noise scale of the
  metric, not an arbitrary `δ`. This is the structural answer to the otherwise-valid worry
  that a barely-worse-than-real baseline could produce an unstable, inflated skill scale.
- **Well-posed by construction.** Every admissible baseline has resolvable `m*_b(k) > 0`,
  so `a₀(k,N) − a₁(k,N)` is bounded away from zero (`O(1)` asymptotically). Admissibility
  and well-posedness are *the same condition*.
- **Automatic and general.** The assignment of baselines to metrics is data-driven — no
  per-pair manual analysis — and reuses only the null and `ε`. The panel-design burden
  reduces to *spanning* the taxonomy (§4); the assignment is free.

**Empty admissible set.** If `A_k(N) = ∅`, metric `k` cannot be skill-scored at this `N`,
and this is reported as an informative diagnostic rather than forced into a number: either
the panel does not span axis `k` (no null distorts it resolvably), or `N` is too small to
resolve any baseline from real. Both are worth knowing — the former flags a gap in the
panel relative to the taxonomy, the latter a non-discriminating metric or an under-powered
evaluation.

---

## 6. The skill score

For a lower-is-better metric `k`, define the **climate generative skill score**

```
                 a₀(k,N) − m̂_model(k,N)
   S_k(N)  =  ──────────────────────────────
                 a₀(k,N) − a₁(k,N)
```

with `a₁` the perfect anchor (§3) and `a₀` the per-metric reference anchor (§5).

| Value         | Meaning                                                        |
|---------------|----------------------------------------------------------------|
| `S = 1`       | at the perfect-model anchor (median real-vs-real self-distance) |
| `0 < S < 1`   | between the strongest admissible naive baseline and real       |
| `S = 0`       | as good as the strongest admissible naive baseline             |
| `S < 0`       | **worse** than every admissible naive baseline on this axis    |
| `S > 1`       | **below** the median self-distance — see interpretation note   |

This is the skill-score form `(reference − model)/(reference − perfect)` of Murphy (1988)
and Nash–Sutcliffe (1970), specialized to generative evaluation by two endpoints: the
data-defined irreducible self-distance of the real data, and the protocol-defined strongest
naive null that genuinely lacks the probed property. Values are **never clipped**: negative
skill ("worse than naive") and `S > 1` are both informative and must remain visible.

**Interpretation of `S > 1`.** Because `a₁` is the *median* self-distance, a model can land
below it (`m̂_model < a₁`) and obtain `S > 1`, by chance or because it is overly smooth /
low-diversity and matches a narrow metric better than two finite real samples match each
other. `S > 1` is therefore **not** evidence of "super-realism": it is flagged and
cross-checked against the diversity axis and complementary metrics, since generative models
can exploit smoothed or marginal metrics. This is reported, not clipped.

The score is reported per metric as a profile; it is not collapsed into a single scalar
(§11).

---

## 7. Sample-size stability

The central theoretical claim is **non-divergence**; finite-sample flatness is an empirical
property, assessed not assumed.

Decompose each measured metric into its population value plus a finite-sample inflation
term, positive because finite empirical measures are on average farther apart than the laws
they estimate:

```
   m̂_model(N)  ≈  m*_model  +  c_model · N^{−α_model}
   a₀(N)        ≈  m*_ref    +  c_ref   · N^{−α_ref}      (m*_ref > 0 by admissibility, §5)
   a₁(N)        ≈  0         +  c_floor · N^{−α_floor}.
```

**Non-divergence (the theorem, in outline).** Provided the estimator is consistent and the
reference discrepancy is positive (guaranteed by admissibility),

```
   a₀(N) − a₁(N)  →  m*_ref > 0,
```

so the denominator is bounded away from zero and

```
   S_k(N)  →  1 − m*_model / m*_ref       as N → ∞,
```

a finite **population skill**. The framework estimates a well-defined intrinsic quantity,
not a test statistic. This holds without assuming the inflation rates or constants match.

**Finite-`N` behaviour (empirical, not guaranteed).** The residual inflation terms partly
*cancel*, being the shared sampling mode across model, reference, and floor evaluated at the
same `N`; a difference-over-difference of same-`N` estimates suppresses common modes.
However, the rates `α_·` and constants `c_·` need **not** match across distributions —
especially for Wasserstein distances, spectral errors, feature-space MMDs, diversity
deviations, or hand-crafted physical metrics, some of which have biased finite-sample
estimators or strong tail dependence, and which may behave differently under deterministic
baselines, Gaussian surrogates, and neural generators. We therefore make the honest claim:

> **Guarantee.** `S_k(N)` is **asymptotically non-divergent** (the denominator stabilizes at
> `m*_ref > 0`). Finite-sample stability is **assessed empirically** via the sample-size
> sweep — plot `S_k(N)` for the model and each baseline; flat curves are the evidence — and
> validated on a synthetic process with known structure (§13a). We do not claim guaranteed
> flatness.

---

## 8. Why noise-floor normalization fails

A natural-seeming alternative standardizes the model's departure by the **internal spread
of the real-vs-real null** — i.e. it studentizes:

```
                 m̂_model(N) − a₁(N)
   G(N)  =  ──────────────────────────────         (a₁ = floor median, w = floor spread)
                 w(N)
```

with `w(N) = Q_{1−ε}(M₀(N)) − median(M₀(N)) → 0`. The numerator is the skill numerator up
to anchor labelling; **only the denominator differs**. But the spread vanishes at rate
`N^{−α}`, so

```
   G(N)  →  m*_model / 0  =  ∞       as N → ∞.
```

`G` is a **studentized test statistic** — a departure divided by its own standard error —
monotone increasing in `N` for any fixed imperfect model. "`G > 1`" means *detectable from
real at this `N`*, not *far from real*; calibrating it so that an `ε` fraction of
real-vs-real replicates exceed `G = 1` simply confirms it is a hypothesis test at level
`ε`. With enough samples every model that is not *exactly* `P` has `G ≫ 1`. This is the
significance-as-effect-size failure of §2 made concrete.

The cure is a single substitution: **replace the vanishing denominator (null spread) with
the stable denominator `a₀ − a₁` (admissible-baseline-to-floor distance)**. The numerator is
essentially unchanged; the scale becomes an effect size. Stating this contrast — and
exhibiting it empirically (§13a) — is itself a contribution: it operationalizes the
significance/effect-size distinction the framework enforces.

---

## 9. The significance ribbon

The null spread is not discarded but demoted to its correct role. Mapping the resolution
band `[a₁(N), r(N)]` of §3 through the skill transform gives the **resolution ribbon** in
skill units, a thin interval just below `S = 1`:

```
   ribbon  =  [ (a₀ − r) / (a₀ − a₁) ,  1 ]            (lower-is-better)
```

with interpretation:

> A model whose skill lies **inside the ribbon** has a residual departure from real that is
> **not resolvable** at this evaluation sample size — it is statistically indistinguishable
> from a perfect generator given `N`. A model **below** the ribbon has a resolvable
> shortfall, and skill measures *how large* it is.

Significance thus answers only *"is the remaining gap even resolvable at this `N`?"* and is
drawn as a band around the perfect anchor, never as the axis. The ribbon **narrows** as `N`
grows — the visual counterpart of the very limit that makes `G` (§8) diverge. The same
construction also serves as the **baseline-admissibility filter** of §5: a baseline whose
interval overlaps the ribbon on metric `k` is "as good as real" there and is disqualified
as that metric's bad anchor. One object, two uses — model resolution and reference
admissibility — both at the single level `ε`.

**Recommended figure.** A schematic of one metric axis showing, in order: the perfect
anchor `a₁`, the ribbon, the model value, the admissible baselines, the *excluded*
(ribbon-overlapping) baselines, and the selected anchor `a₀`. This separates the three
concepts the prose must keep distinct — **metric value** (raw discrepancy), **significance**
(resolvability vs real), and **skill** (effect size vs an admissible baseline).

---

## 10. Metric orientation and admissibility into the suite

Every metric must be lower-is-better before entering the score. A higher-is-better quantity
`d` (e.g. an ensemble diversity measure) enters as an absolute deviation from its real
value, `m = |d_model − d_real|`, whose perfect anchor is the self-distance of that deviation
under real-vs-real resampling and whose admissible baselines are those whose deviation is
resolvably nonzero. More generally, a metric is admissible into the framework if it is
(i) a non-negative functional minimized at the real law and (ii) estimable from finite
samples by a consistent estimator.

---

## 11. Estimation, uncertainty, and aggregation

**Matched `N`.** Estimate `m̂_model`, every `m̂_b`, and the null `M₀` at the *same* `N`, so
the shared-mode cancellation of §7 holds. Report `N`.

**Dependence-aware bootstrap (a methodological requirement, not a detail).** For climate
data the resampling unit must equal the **scientific unit of exchangeability**. Brightness-
temperature images are not iid: they are correlated within storm, basin, season, satellite
platform, and intensity evolution. Resampling at the image level would make `M₀` too narrow,
the ribbon too tight, and the admissibility filter too aggressive (over-admitting baselines
and inflating skill). We therefore resample at the storm- or storm-season level. The same
unit is used for the real-vs-real null, the baselines, and the model.

**Small-`N` and rare classes.** Class-conditional evaluation creates small samples for rare
intensity classes. At small per-class `N`, the admissible set may shrink toward empty
(metrics become "not skill-scorable at this `N`", §5) and bootstrap intervals widen; this is
reported honestly per class rather than smoothed over. Classes with too few exchangeable
units to form a stable null are flagged and excluded from aggregation.

**Reference selection, uncertainty, and selection stability.** Determine the admissible set
and the argmin reference baseline for each metric from the *point estimate* (with a
pre-registered tie-break), then propagate uncertainty by bootstrapping `m̂_model`, `a₀` (for
the fixed selected baseline), and `a₁` under a common scheme, reporting a confidence
interval for `S_k`. Fixing the selected baseline avoids chasing the noise-smallest baseline
inside each replicate. Because this ignores uncertainty in *which* baseline is strongest, we
additionally report, as deliberate diagnostics: (i) the **selection frequency** of each
baseline across bootstrap replicates; (ii) the **skill under the top two or three** admissible
baselines (a sensitivity interval). A metric whose anchor flips between baselines of very
different strength across replicates is flagged as anchor-unstable.

**`ε`-sensitivity.** Recompute admissibility, the ribbon, and the skill profile under
`ε ∈ {0.01, 0.05, 0.10}` and report the spread, so the single global level cannot be
attacked as a tuned choice.

**Aggregation.** Present skill as a **profile** — one `S_k` per metric, with CI — on a common
`(−∞, 1]` axis with the ribbon marked, across classes and metrics. Do **not** report a
silently weighted scalar mean across incommensurable axes. If a single conservative summary
is required, use the **weakest axis** `min_k S_k` ("the model's least-realistic property"),
stated as such. Metrics with empty admissible sets are listed separately with their reason.

**Sample-size sweep.** Recompute the profile across a grid of `N` and plot `S_k(N)` for the
model and each baseline. Flat curves validate the §7 non-divergence guarantee on the
specific estimators and data at hand; this replaces any diagnostic based on the null's
internal shape, which does not bound `G` and is not the relevant stability check.

---

## 12. Baseline-panel design

The lower anchor is principled only if the panel is. A baseline is an admissible member of
the panel iff:

1. **It preserves one named low-order or structural property** of the real class-conditional
   distribution and is designed to **destroy the properties targeted by at least some
   metrics**. "Nothing beyond it" is not claimed — a surrogate inevitably retains incidental
   structure; what is claimed is an explicitly stated *preserved subset* and a *destroyed
   subset*. Its skill profile across the suite is then predictable from that statement,
   which is what makes it diagnostic of *what each metric measures*.
2. **It exposes no tunable knob aligned with the metrics being scored** — no free parameter
   that could be fit to move a particular metric. This is what makes skill = 0 an honest,
   non-gameable anchor.
3. **It is sensible, not trivial:** it retains real structure on the axes it preserves (so it
   is not pure noise, a benchmark that is trivial to beat) while *provably failing* on the
   axes it does not preserve (so it is a genuine bad anchor *there*).

Two baseline *kinds* are distinguished, because they answer different questions:

- **Property-preserving nulls** (e.g. moment- or spectrum-matched surrogates) — single-axis
  ablations of *unconditional* structure.
- **Conditional-label nulls** (e.g. real samples from the wrong condition) — these *preserve
  realism* but *violate conditioning*. They are not single-property ablations and are
  classified separately; they anchor the conditioning/label-fidelity axis of the taxonomy.

The panel must **span** the taxonomy of §4: for every metric, at least one baseline must
provably fail on that metric's property, so the admissible set is non-empty. Removing a
panel member that is the sole anchor of a taxonomy axis *will* change the scores on that
axis — this is expected and acceptable *because the panel is pre-registered against the
taxonomy*, not because the scores are robust to arbitrary panel edits. Robustness is
therefore reported in the right-sized form (anchor + runner-up per metric, §11), not as
invariance to deleting fundamental baselines. The transferable contribution is the *recipe*
— a taxonomy-spanning set of property-preserving and conditional nulls plus CI-admissible
per-metric assignment — not any particular list.

---

## 13. Assumptions, limitations, and objections

- *"The score is not an intrinsic effect size — it is relative to a chosen panel."* Correct,
  and stated as such: the **upper endpoint is dataset-defined; the lower endpoint is
  protocol-defined** through a pre-registered, taxonomy-spanning panel. The "strongest
  admissible baseline" rule removes cherry-picking *among* included baselines; the choice of
  *which* baselines are included is answered by the failure-mode taxonomy (§4) and
  pre-registration, not by an appeal to universality.
- *"The framework rewards matching metrics, not realism."* True of every metric suite, and
  not escaped here. The framework **does not define realism**; it normalizes the evidence
  supplied by a chosen suite of realism probes. A model could score well while failing in
  ways the suite does not probe. The skill score is therefore not a universal realism score,
  and is reported as conditional on the suite and panel.
- *"A baseline could preserve the very property a metric probes."* Then it is excluded as
  that metric's anchor by the CI-admissibility filter (§5), preventing both denominator
  collapse and the false-penalty failure. This is the structural fix distinguishing the
  framework from a naive minimum-over-all-baselines rule.
- *"A barely-worse-than-real baseline gives an unstable, inflated scale."* Prevented by
  CI-separation admissibility (§5): the selected anchor is separated from the floor by at
  least one `ε`-level of the metric's noise, bounding the denominator without an arbitrary
  margin parameter. Residual fragility surfaces as a wide skill CI, never a confident point
  estimate.
- *"Finite-`N` stability is overclaimed."* The theorem is **non-divergence**; flatness is
  empirical (§7) and demonstrated on synthetic ground truth (§13a).
- *"Why not a single proper scoring rule (energy distance, kernel score, CRPS)?"* Proper
  scoring rules answer a **different** question: incentive-compatible *absolute* scores for
  *probabilistic forecasts*. This framework provides an interpretable *relative* scale for a
  *heterogeneous* suite of metrics, **including non-proper physical diagnostics** that are
  not distances between laws. The relationship is **complementary, not competitive**; a
  proper score, where applicable, can be one metric wrapped by the framework. We also state
  scope explicitly: this is **sample-based (un)conditional generation evaluation**, not
  per-pixel forecast-distribution verification.
- *"Adoption cost."* The framework needs only real data, a small pre-registered baseline
  panel, a suite of lower-is-better metrics, and a bootstrap — no retraining, no new losses,
  no held-out benchmark.

### 13a. Validation on synthetic ground truth

Because the TC case study has no ground-truth "distance from realism," the framework is
first validated on a controlled synthetic process where the truth is known
(`scripts/toy_validation.py`). A real law is a spatial random field with prescribed
covariance, a non-Gaussian marginal, and an injected coherent structure; "models" are
constructed with *known* failure modes (matches marginal but not covariance; matches
covariance but Gaussian/structureless; close-but-imperfect); the panel is the
property-preserving nulls of §4. The experiment demonstrates, with ground truth available,
that: (i) noise-floor normalization `G` diverges with `N` while skill `S` converges;
(ii) the CI-admissibility filter excludes property-preserving baselines on the metrics they
preserve; (iii) the recovered skill profile matches the planted failure modes. This precedes
the TC study, which then serves as a real-world demonstration rather than the sole evidence.

---

## 14. Worked instantiation: tropical-cyclone brightness temperature

The framework is domain-agnostic; we instantiate it on class-conditional generation of
tropical-cyclone (TC) brightness-temperature (BT) fields, stratified by intensity class.

**Metric suite (all oriented lower-is-better).** BT-histogram Wasserstein-1 and
Jensen–Shannon distances (value distribution); radial-profile mean absolute error and
radial power-spectral-density error (radial structure and spectrum); a gradient-angle
("DAV") distance and an eye-contrast distance (coherent core structure); cold-cloud-fraction
distance (convective extent); a feature-space MMD and a feature-diversity deviation
(learned-feature distribution and spread). Higher-is-better diversity enters as `|·|` per
§10.

**Panel, mapped to the failure-mode taxonomy (§4).**

| Baseline       | Preserves                                   | Provably fails on | Taxonomy axis |
|----------------|---------------------------------------------|-------------------|---------------|
| `CLIM`         | per-class pixel mean only (deterministic)   | variance, spectrum, all structure, diversity | marginals / diversity |
| `GAUSS-PIX`    | per-pixel mean and variance                 | spatial dependence, coherent structure | second-order spatial |
| `GAUSS-FULL`   | full pixel mean and covariance              | non-Gaussian marginals, gradient-angle and per-image coherent structure | spectral / second-order |
| `PHASE-SHUFFLE`| per-image amplitude spectrum                | coherent spatial structure (eye, eyewall, asymmetry) | coherent structures |
| `CROSS-CLASS`  | real images of the *wrong* intensity class  | conditional/label fidelity | **conditioning (conditional-label null)** |

**Per-metric admissibility in action.** `GAUSS-FULL` reproduces the **expected second-order
spectral structure** by construction (matched covariance), so on the radial-PSD metric it
falls inside the ribbon and is **excluded** as that metric's anchor; the PSD axis is instead
anchored by the strongest remaining null that resolvably distorts the spectrum. Symmetrically
`PHASE-SHUFFLE` preserves the spectrum and is excluded as the anchor for spectral metrics but
is admissible — often the *strongest* admissible null — for coherent-structure metrics it
destroys. This routing is performed automatically by the CI-admissibility filter (§5).

**Implementation precision (pre-empting attack).**
- *GAUSS-FULL* matches the empirical mean and covariance, hence the *expected* second-order
  spectral structure; we do **not** claim exact preservation of the radial-PSD *metric*,
  whose value depends on vectorization, centering, per-image vs ensemble computation, and
  sampling. It is the admissibility filter, not the construction, that certifies whether
  GAUSS-FULL is at the floor for a given metric.
- *PHASE-SHUFFLE* preserves the amplitude spectrum and the mean and variance but
  **Gaussianizes the one-point value marginal** (sum of independent-phase modes → CLT);
  it should *not* be claimed to preserve the full BT histogram shape. Implementation must
  specify windowing/padding, radial centering, application to anomalies vs raw fields,
  treatment of missing values, and any post-hoc marginal remapping; admissibility is then
  decided per metric by the filter, not read off the construction.
- *CROSS-CLASS* is a conditional-label null (preserves realism, violates conditioning),
  classified separately from the single-property ablations.
- *CLIM* is deterministic: drawing `N` identical fields gives zero ensemble spread, so
  diversity and sample-based metrics treat it as a degenerate (zero-variance) case; we state
  that deterministic baselines are admitted and how each metric handles them.

---

## 15. One-paragraph statement of the contribution

> We evaluate generative climate models on a per-metric **skill scale with two interpretable
> endpoints**: a **dataset-defined** upper anchor — the irreducible finite-sample
> self-distance of the real data (skill 1, the behaviour of a perfect generator) — and a
> **protocol-defined** lower anchor — the error of the strongest member of a pre-registered,
> failure-mode-spanning panel of naive baselines that genuinely lacks the metric's target
> property (skill 0). Admissibility of a baseline as the bad anchor is decided per metric by
> resolvability against the real-vs-real null at a single significance level, which both
> prevents the scale from collapsing when a baseline reproduces a metric's property and
> bounds denominator fragility without an arbitrary margin parameter. The resulting score is a
> normalized **effect size** that is provably non-divergent in evaluation sample size — in
> contrast to noise-floor normalizations that are disguised significance tests and diverge —
> with finite-sample stability demonstrated empirically on a synthetic process with known
> failure modes. Statistical significance is retained only as a narrow resolution ribbon
> around the perfect anchor. The framework wraps any heterogeneous suite of lower-is-better
> metrics, makes its panel-dependence explicit, and requires only real data, a baseline panel,
> and a dependence-aware bootstrap, making it a low-cost, principled reference procedure for
> generative-model evaluation in climate science.

---

### References to cite

- Murphy, A. H. (1988). *Skill scores based on the mean square error and their relationships
  to the correlation coefficient.* Mon. Wea. Rev. 116.
- Nash, J. E. & Sutcliffe, J. V. (1970). *River flow forecasting through conceptual models,
  part I.* J. Hydrol. 10.
- Siebert (hydrological upper/lower-bound benchmark normalization) — the direct inspiration;
  cite the specific paper adapted.
- Theiler, J. et al. (1992). *Testing for nonlinearity in time series: the method of
  surrogate data.* Physica D 58 (phase-randomization null).
- Jolliffe, I. T. & Stephenson, D. B. (2012). *Forecast Verification*, 2nd ed.
- Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules, prediction, and
  estimation.* JASA 102 (proper-scoring-rule contrast, §13).
```
