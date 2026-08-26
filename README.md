# credit-trust-lab

**When does an explanation lead a human to trust a model it should not?**

A study of the gap between a model's *confidence* and its *correctness*, built
on credit scoring. The object of interest is not how accurate the model is, but
how far its stated certainty can be believed — and what happens to that
question once an explanation is placed next to the number.

This is the applied continuation of my bachelor's thesis, *Transparency in
Human-AI Decision-Making: Effects on Reliance and Performance*.

> **Status: step 1 of 3 — foundation.** Data layer, two baseline models,
> evaluation and cost-based thresholding are implemented and tested.
> Recalibration, SHAP explanations and the demo interface are not yet built;
> the structure reserves a place for each (`src/credit_trust_lab/explanation/`,
> `src/credit_trust_lab/app/`).

---

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
src/credit_trust_lab/
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
│   ├── calibration.py    Brier, ECE, reliability bins, trust-gap table
│   ├── costs.py          Cost matrix, threshold optimisation, sensitivity
│   └── report.py         One record per model, serialisable
├── plots/figures.py      Matplotlib figures, fed by metric objects only
├── explanation/          reserved for step 2 (SHAP)
└── app/                  reserved for step 3 (demo interface)

scripts/reproduce_all.py  One command → every figure and number in reports/
tests/                    Codebook, splits, calibration, thresholds, models
notebooks/                Exploration only; never a source of results
```

UML models of all of the above — package, class, state, sequence and activity
diagrams — are in [`docs/architecture.md`](docs/architecture.md).

**Why this split.** The dependency direction is one-way: `data → models →
evaluation → plots`. Evaluation knows nothing about how a model was built,
which is what lets both models be judged on identical terms and what will let
the explanation layer attach without touching anything below it. The only hook
an explainer or an interface needs is `CreditModel.predict_proba_bad`, which is
already stable.

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
`reports/metrics/results.json`.

| | ROC-AUC (95% CI) | PR-AUC | Brier | ECE | threshold | gap to t\* | cost/case |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.820 [0.750, 0.884] | 0.701 | 0.146 | 0.057 | 0.339 | +0.172 | 0.530 |
| Gradient boosting | 0.835 [0.769, 0.896] | 0.720 | 0.142 | 0.064 | 0.124 | −0.042 | 0.515 |

Three observations, all of which are the point of the exercise rather than
incidental:

1. **The confidence intervals swallow the difference.** Gradient boosting leads
   by 0.015 AUC; each interval is roughly 0.13 wide. Across 20 random splits
   the medians reverse (logistic 0.808, boosting 0.783) with almost complete
   overlap. On 1,000 rows, "the better model" is mostly a statement about the
   split.
2. **The cost-optimal threshold is far from 0.5 — and not well identified.**
   Both models cost markedly less at their chosen threshold than at 0.5. But
   the logistic model's empirical optimum sits at 0.339 for every cost ratio
   from 1:1 to 5:1, because on 200 calibration rows the cost curve is a step
   function with wide flat regions. The threshold is a decision made on thin
   evidence, and the sensitivity figure shows it.
3. **A low ECE does not mean the confidence is safe to act on.** The logistic
   model's aggregate ECE is the better of the two, yet its threshold sits
   0.172 above the value its own probabilities imply — a disagreement in
   decision units that the aggregate number hides.

Figures in `reports/figures/`: `reliability.png`, `trust_gap.png`,
`cost_curve.png`, `cost_ratio_sensitivity.png`, `stability.png`, `roc.png`,
`precision_recall.png`.

---

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
- **Step 2.** Recalibration (Platt, isotonic) on the reserved calibration
  partition, measured with the apparatus already in place. SHAP values for both
  models, plus a stability measure for those values across bootstrap refits —
  an explanation that changes under resampling should not be presented as an
  account of the model's reasoning.
- **Step 3.** A demo interface showing, per case, the score, the cost-derived
  decision, the explanation, and what that level of confidence has historically
  been worth. The last of these is what the guiding question is about.

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
