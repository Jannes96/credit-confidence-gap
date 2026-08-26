"""Explanation layer -- reserved for step 2, intentionally empty.

Planned contents: SHAP values for both models, a stability measure for those
values across bootstrap refits, and a faithfulness check comparing the
explanation's implied ranking against the model's own behaviour under
perturbation.

The interface this layer will consume already exists:
``CreditModel.predict_proba_bad`` is the only hook an explainer needs, so
nothing here requires changes to the model or data layers.
"""
