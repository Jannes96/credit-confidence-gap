"""Demo interface: the single-case view (step 3).

Renders what step 1 and step 2 measured into the form a person would actually
be shown — a score, a decision, a reason — with each assertion paired against
the evidence about what that kind of assertion is worth.

Static HTML by design. See :mod:`credit_confidence_gap.app.case_view` for why a
server is not merely unnecessary here but would add the one capability the
project must not offer.
"""

from credit_confidence_gap.app.case_view import (
    CaseView,
    band_for,
    choose_case,
    render_case_page,
    write_case_page,
)

__all__ = [
    "CaseView",
    "band_for",
    "choose_case",
    "render_case_page",
    "write_case_page",
]
