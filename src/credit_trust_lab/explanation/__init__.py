"""Explanation layer: SHAP attributions and the two checks that qualify them.

Step 2 of the project. The order matters and is not arbitrary: an explanation
is worth showing to a person only if it is *reproducible* (stability) and
*describes what the model actually does* (faithfulness). Both are measured here
before any explanation is put in front of a human in step 3.
"""

from credit_trust_lab.explanation.faithfulness import (
    FaithfulnessReport,
    assess_faithfulness,
    baseline_row,
)
from credit_trust_lab.explanation.shap_values import (
    Attributions,
    additivity_error,
    explain,
    model_margin,
)
from credit_trust_lab.explanation.stability import StabilityReport, assess_stability

__all__ = [
    "Attributions",
    "FaithfulnessReport",
    "StabilityReport",
    "additivity_error",
    "assess_faithfulness",
    "assess_stability",
    "baseline_row",
    "explain",
    "model_margin",
]
