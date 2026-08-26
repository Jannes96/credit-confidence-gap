# Architecture

UML models of `credit-trust-lab`. Diagrams are Mermaid, so they render on
GitHub and stay diffable — a PNG that silently drifts away from the code is
worse than no diagram.

Six views, each answering a different question:

| # | Diagram | Question it answers |
| --- | --- | --- |
| 1 | Package diagram | How are the layers allowed to depend on each other? |
| 2 | Class diagram — data | What is the coding table, structurally? |
| 3 | Class + state — models | What contract do both models satisfy? |
| 4 | Class diagram — evaluation | What exactly is reported per model? |
| 5 | Sequence diagram | What happens during `make reproduce`? |
| 6 | Activity diagram | Which partition is allowed to touch what? |

---

## 1. Package diagram — the one-way dependency rule

The load-bearing architectural claim: dependencies point in exactly one
direction. `evaluation` cannot import `models` for metric computation, so it
has no way of treating two models differently. It only ever sees two arrays,
`y_true` and `y_prob`.

```mermaid
flowchart TB
    SCR["«script»<br/>scripts/reproduce_all.py"]

    subgraph LIB["src/credit_trust_lab"]
        CFG["«package» config<br/><br/>SplitConfig · CostConfig<br/>CalibrationConfig · BootstrapConfig<br/>StabilityConfig · ModelConfig"]
        DATA["«package» data<br/><br/>download · codebook · loader<br/>splits · preprocess"]
        MOD["«package» models<br/><br/>base · logistic<br/>gradient_boosting · registry"]
        EVAL["«package» evaluation<br/><br/>discrimination · calibration<br/>costs · report"]
        PLT["«package» plots<br/><br/>figures"]
        EXP["«package» explanation<br/><br/>step 2 — reserved"]
        APP["«package» app<br/><br/>step 3 — reserved"]
    end

    DATA -.->|uses| CFG
    MOD -.->|uses| CFG
    MOD -.->|uses| DATA
    EVAL -.->|uses| CFG
    EVAL -.->|uses| DATA
    EVAL -.->|uses — report.py only| MOD
    PLT -.->|uses| EVAL

    SCR ==> DATA
    SCR ==> MOD
    SCR ==> EVAL
    SCR ==> PLT

    EXP -.->|planned| MOD
    APP -.->|planned| EVAL

    classDef reserved stroke-dasharray:5 5,opacity:0.55
    class EXP,APP reserved
```

Two details the diagram makes visible:

- **Only `evaluation/report.py` depends on `models`** — it needs `CreditModel`
  to call `fit` and `predict_proba_bad`. The three metric modules
  (`discrimination`, `calibration`, `costs`) import nothing from `models` at
  all. That is what makes identical treatment structural rather than intended.
- **The reserved packages depend on stable interfaces only.** An explainer
  needs `CreditModel.predict_proba_bad`; the interface layer needs
  `ModelEvaluation`. Neither requires a change below it.

---

## 2. Class diagram — the coding table as a structure

The codebook is the project's contract with the raw file. `FeatureKind` is not
documentation: it is the switch that `preprocess.py` reads to decide between
scaling and one-hot encoding.

```mermaid
classDiagram
    direction LR

    class FeatureKind {
        <<enumeration>>
        NUMERIC
        ORDINAL
        NOMINAL
        BINARY
    }

    class Feature {
        <<frozen dataclass>>
        +str raw_name
        +str name
        +FeatureKind kind
        +Mapping levels
        +str description
        +str note
        +codes() tuple
        +is_categorical() bool
    }

    class Target {
        <<frozen dataclass>>
        +str raw_name
        +str name
        +str positive_label
        +encode(raw_values) Series
    }

    class RawDataset {
        <<frozen dataclass>>
        +Path data_path
        +Path codetable_path
    }

    class DataSplits {
        <<frozen dataclass>>
        +DataFrame X_train
        +Series y_train
        +DataFrame X_calibration
        +Series y_calibration
        +DataFrame X_test
        +Series y_test
        +ndarray train_index
        +ndarray calibration_index
        +ndarray test_index
        +int random_state
        +sizes() dict
        +positive_rates() dict
    }

    class loader {
        <<module>>
        +load_dataset(raw) tuple
        +load_raw_frame(path) DataFrame
        +validate_codes(frame) void
        +data_card() str
    }

    class preprocess {
        <<module>>
        +build_preprocessor() ColumnTransformer
        +expected_feature_names() list
        +validate_frame(X) void
    }

    class splits {
        <<module>>
        +make_splits(X, y, config, seed) DataSplits
    }

    Feature --> FeatureKind : kind
    loader ..> Feature : validates against
    loader ..> Target : applies encode
    loader ..> RawDataset : reads
    splits ..> DataSplits : creates
    preprocess ..> Feature : reads kind and levels

    note for Target "encode flips the raw coding once, globally"
    note for Feature "levels is a MappingProxyType, so it cannot be mutated"
```

`Feature.note` carries the caveats that must survive into the report — for
example that level 1 of `savings` means *unknown or no account*, not a measured
balance. `loader.data_card()` renders those notes to
`reports/metrics/data_card.txt`, so they travel with the results.

---

## 3. Class and state diagram — the model contract

Both models are reduced to one operation: given a predictor frame, return
P(bad credit). Subclasses supply nothing but the estimator.

```mermaid
classDiagram
    direction TB

    class CreditModel {
        <<abstract>>
        +str name
        +ModelConfig config
        -Pipeline _pipeline
        +fit(X, y) Self
        +predict_proba_bad(X) ndarray
        +decide(X, threshold) ndarray
        +is_fitted() bool
        +pipeline() Pipeline
        #_build_estimator()*
    }

    class LogisticScorecard {
        +str name
        #_build_estimator() LogisticRegression
    }

    class GradientBoostingScorer {
        +str name
        #_build_estimator() GradientBoostingClassifier
    }

    class ModelConfig {
        <<frozen dataclass>>
        +int random_state
        +float logistic_c
        +int logistic_max_iter
        +int gb_n_estimators
        +float gb_learning_rate
        +int gb_max_depth
    }

    class registry {
        <<module>>
        +MODEL_REGISTRY
        +build_model(name, config) CreditModel
        +build_all_models(config) list
    }

    class NotFittedError {
        <<exception>>
    }

    class preprocess {
        <<module>>
        +build_preprocessor() ColumnTransformer
    }

    CreditModel <|-- LogisticScorecard
    CreditModel <|-- GradientBoostingScorer
    CreditModel o-- ModelConfig
    CreditModel ..> NotFittedError : raises
    CreditModel ..> preprocess : shared preparation
    registry ..> CreditModel : creates

    note for CreditModel "No predict method: decide requires an explicit threshold"
```

The lifecycle is two states, and the guard between them is enforced rather than
documented:

```mermaid
stateDiagram-v2
    direction LR

    [*] --> Unfitted : build_model
    Unfitted --> Fitted : fit on X_train
    Fitted --> Fitted : refit rebuilds the Pipeline from scratch
    Fitted --> [*]

    note right of Unfitted
        predict_proba_bad raises
        NotFittedError
    end note

    note right of Fitted
        predict_proba_bad looks the
        positive column up via
        estimator.classes_ == 1
    end note
```

`fit()` constructs a **new** `Pipeline` on every call, so no state survives a
refit. That is what lets the 20-seed stability run reuse the builder without
risk of contamination.

---

## 4. Class diagram — what is reported per model

Every evaluation collapses into one serialisable record. Note that
`ThresholdResult` carries *both* thresholds: the empirical optimum and the one
perfect calibration would imply. Their difference is the project's headline
diagnostic, expressed in decision units.

```mermaid
classDiagram
    direction LR

    class ModelEvaluation {
        <<frozen dataclass>>
        +str model_name
        +float test_cost_at_chosen_threshold
        +float test_cost_per_case
        +dict test_confusion
        +as_dict() dict
    }

    class PartitionEvaluation {
        <<frozen dataclass>>
        +str partition
        +ndarray y_true
        +ndarray y_prob
    }

    class DiscriminationMetrics {
        <<frozen dataclass>>
        +float roc_auc
        +float pr_auc
        +float prevalence
        +int n_samples
        +int n_positive
    }

    class CalibrationMetrics {
        <<frozen dataclass>>
        +float brier_score
        +float expected_calibration_error
        +float maximum_calibration_error
        +float signed_calibration_error
        +float mean_predicted
        +float observed_frequency
        +int n_bins
        +str binning
    }

    class ReliabilityBin {
        <<frozen dataclass>>
        +float lower
        +float upper
        +int count
        +float mean_predicted
        +float observed_frequency
        +gap() float
    }

    class TrustGapRow {
        <<frozen dataclass>>
        +float lower
        +float upper
        +int count
        +float share
        +float mean_confidence
        +float accuracy
        +float gap
    }

    class ThresholdResult {
        <<frozen dataclass>>
        +float threshold
        +float theoretical_threshold
        +float threshold_gap
        +float total_cost
        +float cost_per_case
        +float cost_at_half
        +float cost_at_theoretical
        +float cost_ratio
    }

    class ConfusionCounts {
        <<frozen dataclass>>
        +int true_negative
        +int false_positive
        +int false_negative
        +int true_positive
        +n() int
    }

    class Interval {
        <<frozen dataclass>>
        +float point
        +float lower
        +float upper
        +float confidence_level
    }

    ModelEvaluation *-- "2" PartitionEvaluation : calibration and test
    ModelEvaluation *-- "1" ThresholdResult : chosen
    ModelEvaluation *-- "8" ThresholdResult : cost-ratio sweep
    ModelEvaluation *-- "2" Interval : bootstrap CIs
    PartitionEvaluation *-- DiscriminationMetrics
    PartitionEvaluation *-- CalibrationMetrics
    PartitionEvaluation *-- "n" TrustGapRow
    CalibrationMetrics *-- "n" ReliabilityBin
    ThresholdResult *-- ConfusionCounts

    note for ThresholdResult "threshold_gap = threshold minus theoretical"
    note for CalibrationMetrics "signed error keeps the sign; ECE alone cannot"
```

---

## 5. Sequence diagram — one run of `make reproduce`

The runtime flow. The single most important thing it encodes: the threshold is
optimised on the **calibration** partition and only then applied to test.

```mermaid
sequenceDiagram
    autonumber
    actor U as make reproduce
    participant S as reproduce_all.py
    participant D as data.download
    participant L as data.loader
    participant SP as data.splits
    participant R as models.registry
    participant M as CreditModel
    participant EV as evaluation.report
    participant C as evaluation.costs
    participant K as evaluation.calibration
    participant F as plots.figures

    U->>S: main

    rect rgb(236, 243, 250)
    Note over S,L: Step 1 — data, verified rather than trusted
    S->>D: download_raw
    D->>D: SHA-256 of archive, then of each member
    D-->>S: RawDataset
    S->>L: load_dataset
    L->>L: rename, TARGET.encode, validate_codes
    L-->>S: X 1000x20, y with 300 bad credits
    end

    rect rgb(238, 247, 238)
    Note over S,SP: Step 2 — three-way split
    S->>SP: make_splits
    SP->>SP: stratified cut 1, test 20 percent
    SP->>SP: stratified cut 2, 25 percent of the remainder
    SP->>SP: assert disjoint and exhaustive
    SP-->>S: DataSplits 600 / 200 / 200
    end

    rect rgb(252, 245, 234)
    Note over S,K: Step 3 — per model
    S->>R: build_all_models
    R-->>S: LogisticScorecard, GradientBoostingScorer

    loop for each registered model
        S->>EV: evaluate_model
        EV->>M: fit on X_train, y_train
        M-->>EV: fitted Pipeline

        EV->>M: predict_proba_bad on X_calibration
        M-->>EV: p_cal
        EV->>C: optimise_threshold
        C->>C: sweep every observed score
        C-->>EV: ThresholdResult, t and t_star 0.167 and gap

        Note over EV,M: only now is the test partition touched
        EV->>M: predict_proba_bad on X_test
        M-->>EV: p_test
        EV->>K: calibration_metrics and trust_gap_table
        K-->>EV: ECE, Brier, bins, trust-gap rows
        EV->>C: expected_cost at the fixed t
        C-->>EV: realised test cost
        EV-->>S: ModelEvaluation
    end
    end

    rect rgb(245, 241, 250)
    Note over S,F: Steps 4 to 6 — figures, stability, artefacts
    S->>F: reliability, roc, pr, cost curve, sensitivity, trust gap
    S->>S: run_stability over 20 seeds
    S->>F: stability box plots
    S->>S: write results.json and data_card.txt
    end

    S-->>U: exit 0
```

---

## 6. Activity diagram — which partition may touch what

The rule the whole three-way split exists to enforce. Arrows only ever move
downward: no quantity fitted on a later partition flows back to an earlier one,
and nothing is fitted on test at all.

```mermaid
flowchart TB
    START(["load_dataset<br/>1000 rows, 300 bad"]) --> SPLIT{{"make_splits<br/>stratified, seed 20240601"}}

    SPLIT --> TR
    SPLIT --> CA
    SPLIT --> TE

    subgraph TR["Train — 600 rows"]
        direction TB
        T1["fit StandardScaler and OneHotEncoder"] --> T2["estimate coefficients / trees"]
    end

    subgraph CA["Calibration — 200 rows"]
        direction TB
        K1["sweep the cost curve"] --> K2["choose threshold t"] --> K3["cost-ratio sensitivity 1:1 to 20:1"]
        K4["step 2: Platt / isotonic recalibration"]
    end

    subgraph TE["Test — 200 rows"]
        direction TB
        E1["ROC-AUC, PR-AUC, stratified bootstrap CI"]
        E2["Brier, ECE, MCE, signed error"]
        E3["trust-gap table by confidence band"]
        E4["realised cost at the fixed t"]
    end

    TR -->|fitted model| CA
    CA -->|threshold t, frozen| TE
    TE --> OUT(["results.json<br/>7 figures"])

    NOGO["never: choose t on test,<br/>refit on test,<br/>recalibrate on test"]
    TE -.->|excluded by construction| NOGO

    classDef planned stroke-dasharray:5 5,opacity:0.55
    classDef forbidden stroke:#b00020,color:#b00020,stroke-dasharray:3 3
    class K4 planned
    class NOGO forbidden
```

Each partition consumes only what an earlier one produced. The threshold `t` is
a fitted parameter like any coefficient; fitting it on the reporting data would
be the same leak as fitting a coefficient there, only harder to notice because
it looks like evaluation rather than training.

---

## Where the invariants are enforced

Diagrams describe intent; these tests make the intent binding.

| Modelled property | Enforced by |
| --- | --- |
| Partitions are pairwise disjoint and exhaustive | `splits._assert_disjoint_and_exhaustive()` at runtime, plus `tests/test_splits.py` |
| Threshold fitted on calibration, not test | `test_threshold_is_fitted_on_calibration_and_applied_to_test` |
| No `predict()` reinstating 0.5 | `test_decide_requires_an_explicit_threshold` |
| `predict_proba_bad` really returns P(bad) | `test_probabilities_are_oriented_towards_bad_credit` |
| Codebook matches the raw file | `loader.validate_codes()`, plus the integration tests in `tests/test_codebook.py` |
| Corrected data edition, not the legacy UCI one | `test_foreign_worker_minority_is_the_foreign_one` |
| Design matrix width independent of the split | `test_declared_categories_survive_absence_from_the_training_rows` |
| Threshold arithmetic | `test_optimal_threshold_matches_theory_on_calibrated_scores` |
