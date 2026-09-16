# credit-confidence-gap

**When does an explanation lead a human to trust a model it should not?**

A study of the gap between a model's *confidence* and its *correctness*, built
on credit scoring. The object of interest is not how accurate the model is, but
how far its stated certainty can be believed — and what happens to that
question once an explanation is placed next to the number.

This is the applied continuation of my bachelor's thesis, *Transparency in
Human-AI Decision-Making: Effects on Reliance and Performance*.

> **Status: steps 1–2 of 3.** Data layer, two baselines, calibration
> measurement, cost-based thresholds, post-hoc recalibration and SHAP
> explanations with stability and faithfulness checks are implemented and
> tested. The demo interface is not yet built; the structure reserves a place
> for it (`src/credit_confidence_gap/app/`).


## The design commitment

Every choice below follows from one decision: **calibration matters more than
AUC, and honesty about limits matters more than good numbers.**

| Conventional choice | Choice here | Why |
| --- | --- | --- |
| Train/test split | **Train / calibration / test**, three disjoint sets | Recalibrating and evaluating on the same rows is the standard defect in projects like this. The third set is created in the data layer, so avoiding it is structural, not a matter of discipline. |
| Threshold 0.5 | Threshold from an **explicit cost matrix** | 0.5 minimises the error *count*, treating a missed default as equal to a rejected good customer. No lender believes that. |
| `class_weight="balanced"` | No class weighting | Reweighting the loss distorts exactly the probabilities under study. Asymmetry belongs at the decision, not in the likelihood. |
| Report point estimates | Bootstrap intervals **and** a 20-seed stability run | With ~60 defaults in the test set, a single AUC is one draw from a wide distribution. |
| Maximise accuracy | Report the confidence-correctness gap | See the guiding question. |

---

## Quickstart

Requires Python 3.10+ (developed on 3.11).

```bash
make reproduce
```

From an empty clone this creates the virtualenv, downloads the raw data,
verifies its SHA-256, fits both models, and regenerates every figure and metric
in `reports/`. Other targets:

```bash
make setup      # virtualenv + editable install
make data       # download and checksum-verify the raw file only
make test       # pytest
make lint       # ruff check + format check
make typecheck  # mypy
make all        # everything, in the order CI would run it
```

---

## Repository structure

```
src/credit_confidence_gap/
├── config.py             All seeds, split fractions, costs and hyperparameters
├── data/
│   ├── download.py       Fetch + SHA-256 verification of the raw archive
│   ├── codebook.py       The coding table, as executable code
│   ├── loader.py         Load, rename, validate against the codebook
│   ├── preprocess.py     Ordinal/nominal handling driven by the codebook
│   └── splits.py         Three-way stratified split with disjointness checks
├── models/
│   ├── base.py           The shared interface: predict_proba_bad()
│   ├── logistic.py       Logistic regression (the scorecard baseline)
│   ├── gradient_boosting.py
│   └── registry.py
├── evaluation/
│   ├── discrimination.py ROC-AUC, PR-AUC, bootstrap intervals
│   ├── calibration.py    Brier, ECE, reliability bins, trust gap + pooling
│   ├── null_model.py     What perfect calibration would score at this n
│   ├── costs.py          Cost matrix, threshold optimisation, sensitivity
│   └── report.py         One record per model, serialisable
├── recalibration/
│   ├── base.py           Calibrator interface + the model wrapper
│   ├── methods.py        Identity (control), Platt, isotonic
│   └── fitting.py        Cross-fitted fit, so the threshold stays honest
├── explanation/
│   ├── shap_values.py    Exact SHAP, aggregated to codebook features
│   ├── stability.py      Does the explanation survive resampling?
│   └── faithfulness.py   Does the model actually respond to it?
├── plots/figures.py      Matplotlib figures, fed by metric objects only
└── app/                  reserved for step 3 (demo interface)

scripts/reproduce_all.py  One command → every figure and number in reports/
tests/                    Codebook, splits, calibration, thresholds, models
notebooks/                Exploration only; never a source of results
```

UML models of all of the above — package, class, state, sequence and activity
diagrams — are in [`docs/architecture.md`](docs/architecture.md).

**Why this split.** The dependency direction is one-way: `data → models →
evaluation → plots`. Evaluation knows nothing about how a model was built,
which is what lets both models be judged on identical terms.

Step 2 tested that claim and it held. A recalibrated model is not a
`CreditModel` — it owns no preprocessing pipeline and cannot be fitted from raw
features — but it satisfies the narrower `ScoringModel` protocol, so every
metric, threshold routine and figure accepted it without a single change to the
evaluation layer. The explanation modules likewise attach through
`predict_proba_bad` and the fitted pipeline alone.

**Notebooks are exploration only.** No result in `reports/` or in this README
originates in a notebook. A notebook's output depends on the cell execution
order, which is neither recorded nor reproducible from the file. Anything whose
correctness matters lives in `src/`, is covered by a test, and is invoked from
`scripts/reproduce_all.py`.

---

## Method

**Data.** South German Credit in the corrected edition of Grömping (2019) —
1,000 cases, 20 predictors, binary target. Downloaded from UCI dataset 573 and
verified against a pinned SHA-256; the coding table lives in
`data/codebook.py` as a typed structure, not as a comment, and
`loader.py` refuses to proceed if any observed code is undeclared.

**Encoding.** The integer codes are the P2 scores of Häußler (1979), built so
that larger is better. Ordinal features (`status`, `savings`,
`employment_duration`, …) keep that order and are scaled; genuinely unordered
ones (`purpose`, `credit_history`, `personal_status_sex`, …) are one-hot
encoded. Categories come from the codebook rather than from the training rows,
so a level absent from one split cannot silently change the design matrix.

Two encoding caveats are recorded per feature rather than smoothed over. The
lowest code of `status`, `savings` and `property` means *absent or unknown*,
not the bottom of a scale — treating it as the bottom rung asserts that having
no checking account is worse than being overdrawn, which is a 1970s scorecard
assumption, not a measurement. And `credit_history` is treated as unordered
despite its monotone codes, because its levels mix payment behaviour with the
mere existence of a credit history.

**Positive class = bad credit.** The default event is `y = 1` throughout. It is
the event the cost matrix prices and the event the precision-recall curve
should be read against. The raw file codes it the other way round; the flip
happens in exactly one place.

**Split.** 60 / 20 / 20, stratified, seed 20240601 → 600 / 200 / 200 rows.
Disjointness is asserted at runtime, not only in tests.

**Threshold.** Derived from the cost matrix, dimensionless, defaulting to the
5:1 ratio of the original Statlog problem: a missed default costs five times a
rejected good customer. Under perfect calibration the optimum would be
`t* = C_FP / (C_FP + C_FN) = 0.167`. The empirical optimum is found by sweeping
the scores **on the calibration partition** and applied unchanged to the test
partition — the threshold is a fitted parameter, and fitting it on the
reporting data would be the same leakage as fitting a coefficient there.

Because 5:1 is an inherited convention rather than a measurement, the ratio is
swept from 1:1 to 20:1 and the resulting thresholds are reported alongside.

---

## Current results

Seed 20240601, regenerate with `make reproduce`. Full output in
`reports/metrics/results.json` and `reports/metrics/step2.json`.

### Every number against its own noise floor

The most consequential thing this project measures is not a model's ECE but
what an ECE *can* mean on 200 test rows. Ten quantile bins leave 20
observations each, so a perfectly calibrated model still scores an ECE around
0.07 through sampling alone. Every calibration figure below is therefore
reported against a parametric bootstrap under the null hypothesis of perfect
calibration (`evaluation/null_model.py`).

| | ROC-AUC (95% CI) | PR-AUC | Brier | ECE | null median | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.820 [0.750, 0.884] | 0.701 | 0.146 | 0.057 | 0.068 | indistinguishable |
| Gradient boosting | 0.835 [0.769, 0.896] | 0.720 | 0.142 | 0.064 | 0.066 | indistinguishable |

**Neither model is measurably miscalibrated.** Both ECE values sit *below* the
median a perfectly calibrated model would produce here. Quoting either as
evidence of miscalibration would be reading noise.

### The decision, and what it costs

| | threshold | defaults caught | good customers refused | cost/case | vs. 0.5 |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.339 | 45/60 = 75 % | 31/140 = 22 % | 0.530 | −36 % |
| Gradient boosting | 0.124 | 55/60 = 92 % | 78/140 = 56 % | 0.515 | −44 % |

The cost difference is 3 %. The operating points are not comparable at all:
one lender refuses every fifth good customer, the other more than every
second. The cost curve is flat across a wide band, so the matrix fixes far
less of the decision than it appears to — something outside these data has to
choose, and the sensitivity figure shows how much is at stake.

### Where the aggregate metric misses something

Under the 5:1 matrix a calibrated model should refuse at `t* = 0.167`. The
logistic model's empirical optimum is 0.339, a gap of **+0.172** against a
null 90 % range of [−0.065, +0.137] — outside the noise band, two-sided
p ≈ 0.035. Gradient boosting's −0.042 (p ≈ 0.37) is noise.

So for the logistic model the *aggregate* calibration metric shows nothing
while the *decision-relevant* one does. That is the project's thesis in one
line: a single summary number cannot answer whether a model's confidence is
safe to act on. One caveat, since this project is about unwarranted
confidence: this is one comparison among several reported here, and p ≈ 0.035
is not a large margin.

### Step 2a — recalibration changes nothing measurable

Identity, Platt and isotonic maps fitted on the calibration partition, with
the threshold chosen from **out-of-fold** probabilities so that it never sees
rows the calibrator was fitted on.

| model | method | ECE | its own null median | cost/case | surviving score levels |
| --- | --- | --- | --- | --- | --- |
| Logistic | identity | 0.057 | 0.068 | 0.530 | 200 |
| Logistic | Platt | 0.057 | 0.069 | 0.530 | 200 |
| Logistic | isotonic | 0.063 | 0.044 | 0.530 | **34** |
| Boosting | identity | 0.064 | 0.066 | 0.515 | 200 |
| Boosting | Platt | 0.066 | 0.068 | 0.530 | 200 |
| Boosting | isotonic | 0.060 | 0.058 | 0.545 | **40** |

Every variant sits inside its own noise band; no method achieved a measurable
improvement. That is the expected outcome when the starting point was not
measurably wrong, and it is why the identity control exists — without it,
"isotonic improved boosting's ECE from 0.064 to 0.060" would look like a
result.

The last column is a cost no calibration metric reports. Isotonic regression
is monotone but not injective: it collapses 200 distinct scores into 34, so
applicants who were ranked become tied and can no longer be told apart. Platt
scaling is strictly monotone and leaves ROC-AUC exactly unchanged; isotonic
does not.

### Step 2b — the explanations are faithful but unstable

| | top-5 overlap across refits | chance | same top feature | faithfulness ratio | direction |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.60 | 0.14 | 67 % | 1.82 | 1.00 |
| Gradient boosting | 0.56 | 0.14 | 54 % | 1.91 | 0.97 |

SHAP additivity error is at floating-point noise (≤ 4e-15) for both, so the
attributions are exact rather than approximate.

**Faithfulness passes.** Masking the features an explanation names moves the
model roughly 1.9 times as far as masking random ones, and the direction is
right in 97–100 % of cases. The explanation does describe what the model does.

**Stability does not.** Refit on a bootstrap resample of the same 600 training
rows and the single most influential characteristic changes for a third of
logistic cases and **nearly half** of boosting cases. Well above chance, far
from reproducible.

That combination is the finding. A caseworker shown "declined: checking
account status, credit duration" receives a statement that is *true of this
model* — and that would have named different characteristics had the bank
sampled a different 600 customers. The explanation earns trust in the
mechanism while concealing how arbitrary the mechanism is. This is the
condition under which an explanation produces confidence it has not earned,
which is the question the project set out to ask.

### The headline: the trust gap, pooled across splits

On a single split the top confidence band holds ~50 cases, so a five-point gap
is about two cases — a direction, not a finding. Pooling the twenty stability
splits raises each band to several hundred. The interval is a **cluster
bootstrap over whole splits**, not a binomial interval on the pooled count: the
splits overlap, so the cases are not independent and a binomial interval would
be too narrow.

| model | band | n | stated | realised | gap [95 % CI] | vs. null |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic | 0.9–1.0 | 1018 | 94.4 % | 92.4 % | **+0.020** [+0.003, +0.037] | p = 0.013 |
| Boosting | 0.8–0.9 | 910 | 85.2 % | 81.2 % | **+0.040** [+0.015, +0.064] | p = 0.001 |
| Boosting | 0.9–1.0 | 1137 | 94.6 % | 91.3 % | **+0.033** [+0.015, +0.050] | p < 0.001 |

Every lower band is indistinguishable from noise. Three of ten are not — and
they are the **high-confidence** bands, the ones with the most cases and the
ones where a human is least likely to check. Both models overstate their
certainty precisely where that overstatement matters most.

Two things this changes:

- **The question is now answerable.** Steps 1 and 2 could only say "we cannot
  measure it here". Pooling raised the resolution far enough that the answer
  is a number: when this model says it is 95 % sure, it is right about 92 % of
  the time, and that shortfall is larger than sampling noise.
- **Gradient boosting is the worse offender** (+0.033 versus +0.020 in the top
  band) despite having the marginally better aggregate ECE on the canonical
  split. A third instance of the same pattern: the summary metric ranks the
  models one way, the decision-relevant one ranks them the other.

Caveat kept in view: ten bands were tested against 90 % null intervals, so one
false positive would be unremarkable. Three, concentrated in the largest and
highest bands rather than scattered, is a pattern rather than an accident —
but it is a pattern in one dataset of a thousand 1970s loans.

Figure: `reports/figures/trust_gap_pooled.png`. Numbers:
`reports/metrics/trust_gap_pooled.json`.

### What it adds up to

1. On 1,000 rows, "which model is better" is not answerable: the intervals
   overlap and 20 seeds reverse the ranking (logistic 0.808, boosting 0.783).
2. Moving the threshold off 0.5 is the one robustly valuable intervention.
3. Aggregate calibration is not measurably broken, so recalibration cannot
   measurably fix it — and isotonic regression costs ranking resolution to try.
4. But pooled across splits, both models *are* measurably overconfident in
   their highest confidence bands. The aggregate metric missed what the
   decision-relevant and the band-level ones both catch.
5. Explanations describe the model correctly and describe an arbitrary model:
   faithful at ratio ~1.9, yet the top-named characteristic changes for a
   third to a half of cases under a refit.

Taken together: a caseworker is shown a confidence that is overstated where it
matters, next to a reason code that is true of a model another sample would
not have produced. That is the condition the project set out to characterise.

Figures in `reports/figures/`: `recalibration.png`, `explanation_checks.png`,
`reliability.png`, `trust_gap.png`, `cost_curve.png`,
`cost_ratio_sensitivity.png`, `stability.png`, `roc.png`,
`precision_recall.png`.
## Limits of this project

These are not disclaimers appended after the fact. They are the reason the
dataset is a usable *methodological* test bed and unusable as a picture of
lending today, and they constrain what any number above is allowed to mean.

**1. The widely used UCI edition is miscoded.** The circulated Statlog German
Credit file mislabels several variables; the best-known case inverts the
foreign-worker variable, so that the majority of a South German bank's 1970s
loan book appears to be foreign workers. This project uses only Grömping's
corrected edition, pins its checksum, and pins the corrected orientation in
`tests/test_codebook.py` so that no refactor can quietly reintroduce the legacy
coding.

**2. Bad credits are deliberately over-represented.** The 300 / 700 split is a
stratified sampling design, not an observed default rate. Every probability
this project produces is calibrated **against the sample distribution**.
Absolute default probabilities are therefore not claimed anywhere, and the
reliability diagram carries that caveat in its own caption. Any transfer to a
population with a different base rate would require a prior correction that is
out of scope here.

**3. Selection bias — the reject-inference problem.** The data contain only
applicants who actually received credit. The model has never seen a rejected
applicant and cannot say anything about one, yet a scoring system is deployed
precisely to decide about applicants before that filter. Nothing in this
project corrects for it; the point is that it is stated rather than assumed
away.

**4. The data are from 1973–1975.** Deutsche Mark amounts, a private telephone
line as a wealth signal, and a variable that fuses sex with marital status.
Two of the predictors (`personal_status_sex`, `foreign_worker`) encode
attributes whose use in a live scorecard would be unlawful under current EU and
German law. They are retained because removing them would hide the issue rather
than resolve it — and because a project about misplaced trust should not
quietly launder its own inputs.

**Not a scoring system.** This repository is a methodological study. It is not
a deployable credit-scoring system, it has not been validated against any
regulatory requirement, and no decision about a real person should be derived
from it. It contains no real customer or institutional data.

**Also deliberately absent.** No deep learning — with 1,000 rows that would be
a category error, not a shortcut. No optimisation for maximum accuracy. No
claim that either model is well specified for anything beyond this comparison.

---

## Roadmap

- **Step 1 (done).** Data layer with checksum verification and an executable
  codebook; three-way split; two baselines behind one interface; discrimination
  and calibration metrics; cost-derived thresholds; tests.
- **Step 2 (done).** A null distribution for every calibration metric, which
  turned out to change how step 1's own results must be read. Recalibration
  (identity control, Platt, isotonic) on the reserved partition, with
  cross-fitted threshold selection. Exact SHAP for both models, aggregated to
  codebook features, with stability and faithfulness checks.
- **Step 3.** A demo interface showing, per case, the score, the cost-derived
  decision, the explanation — and what that explanation has been *worth*: how
  often a characteristic like this one survives a refit, and how far the stated
  confidence has historically been from the truth. Step 2 produced exactly the
  numbers that panel needs.

**Done since:** the trust-gap bands are now pooled across the 20 stability
splits, which raised the top band from ~44 to ~1,000 cases and turned the
project's headline question from a direction into a measured answer. Step 3
can show a number rather than a hint.

## References

- Grömping, U. (2019). *South German Credit Data: Correcting a Widely Used Data
  Set.* Report 4/2019, Beuth University of Applied Sciences Berlin.
- Häußler, W. M. (1979). Empirische Ergebnisse zu Diskriminationsverfahren bei
  Kreditscoringsystemen. *Zeitschrift für Operations Research*, 23, B79–B94.
- Fahrmeir, L., & Hamerle, A. (1984). *Multivariate statistische Verfahren.*
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of
  Modern Neural Networks. *ICML.*
- Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. *IJCAI.*

## License

MIT — see [LICENSE](LICENSE).
