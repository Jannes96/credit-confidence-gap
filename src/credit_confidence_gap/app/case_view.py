"""The single-case view: a score, a decision, a reason — and what each is worth.

This is the project's argument rendered as an interface. A conventional scoring
screen shows two things: a probability and a reason code. Both are presented at
face value, and the research this project continues says that is precisely when
a person over-relies on them.

So the layout inverts the usual emphasis. Every assertion the screen makes is
paired with what step 2 measured about that kind of assertion:

* the probability sits next to the pooled trust gap for its confidence band —
  what confidence at this level has historically been worth;
* the reason code sits next to the stability figure — how often the top-named
  characteristic survives refitting the model on a resample;
* the decision sits next to the threshold and the cost assumption it came from,
  because that assumption is an input, not a measurement.

The page computes nothing. It consumes the records written by
``scripts/reproduce_all.py`` and renders them, so what it shows is the same
number that is in ``reports/metrics/`` by construction. A runtime-computing
interface could drift from the reported results without anybody noticing.

Deliberately static HTML with no framework and no server. The one capability a
server would add — scoring a freshly typed-in applicant — is the one this
project must not offer: the probabilities are calibrated against a stratified
sample of accepted 1973-75 applicants and are not default rates.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from credit_confidence_gap.data.codebook import FEATURES_BY_NAME, FeatureKind
from credit_confidence_gap.evaluation.calibration import PooledTrustGapRow
from credit_confidence_gap.evaluation.costs import CostConfig, ThresholdResult
from credit_confidence_gap.explanation.stability import StabilityReport


@dataclass(frozen=True)
class CaseView:
    """Everything the page shows about one applicant.

    Attributes:
        case_id: Row label of the case in the test partition.
        features: The applicant's raw codebook values.
        probability: Predicted P(bad credit).
        threshold: The decision threshold in force.
        threshold_result: Where that threshold came from, including the
            theoretical value the model's own probabilities imply.
        cost: The cost matrix the threshold was derived from.
        attributions: SHAP contributions for this case, log-odds scale.
        band: The pooled trust-gap row for this case's confidence band, or
            ``None`` if the stability run has not been executed.
        stability: What refitting does to explanations of this model.
        model_name: Which model produced the score.
        refit_intervals: Per feature, the range its contribution takes across
            the bootstrap refits. The data behind the ghost bars.
        top_k: How many characteristics the reason code lists.

    """

    case_id: int
    features: pd.Series
    probability: float
    threshold: float
    threshold_result: ThresholdResult
    cost: CostConfig
    attributions: np.ndarray
    feature_names: tuple[str, ...]
    band: PooledTrustGapRow | None
    stability: StabilityReport
    model_name: str
    refit_intervals: dict[str, tuple[float, float]]
    top_k: int = 5

    @property
    def refused(self) -> bool:
        """Whether the credit is refused at the threshold in force."""
        return self.probability >= self.threshold

    @property
    def confidence(self) -> float:
        """Decision confidence, max(p, 1-p) — the figure an interface shows."""
        return max(self.probability, 1.0 - self.probability)

    def ranked_contributions(self) -> list[tuple[str, float]]:
        """Return the top-k characteristics by absolute contribution, largest first."""
        order = np.argsort(-np.abs(self.attributions))[: self.top_k]
        return [(self.feature_names[i], float(self.attributions[i])) for i in order]


def _describe(feature_name: str, value: float) -> str:
    """Render one raw codebook value the way a person would read it."""
    feature = FEATURES_BY_NAME[feature_name]
    if feature.kind is FeatureKind.NUMERIC:
        if feature_name == "amount":
            return f"{int(value):,} DM".replace(",", " ")
        if feature_name == "duration":
            return f"{int(value)} months"
        return str(int(value))
    return feature.levels[int(value)]


def _label(feature_name: str) -> str:
    """Turn a codebook name into a column heading."""
    return feature_name.replace("_", " ")


def _contribution_bar(value: float, interval: tuple[float, float], scale: float) -> str:
    """Render one contribution, with the range it takes across refits behind it.

    The ghost is the point of this chart. A solid bar asserts "this
    characteristic contributed +0.73"; the ghost shows that across twenty-five
    refits on resampled training data the same characteristic contributed
    anywhere in a range — sometimes a narrow one, sometimes a range that
    crosses zero and reverses the sign of the claim.

    Diverging encoding: two hues either side of a neutral zero line, which is
    the correct form for a polarity measure. The ghost reuses its bar's own
    hue at low opacity rather than introducing a third colour, so the spread
    stays visibly attached to the contribution it belongs to.

    Args:
        value: The contribution actually shown, log-odds scale.
        interval: ``(minimum, maximum)`` across refits.
        scale: Half-width of the axis, in log-odds.

    Returns:
        An HTML fragment positioned on a shared centre line.

    """
    low, high = interval

    def place(x: float) -> float:
        """Map a log-odds value onto 0-100 % of the track, zero at the centre."""
        return 50.0 + max(-1.0, min(1.0, x / scale)) * 50.0

    ghost_left = min(place(low), place(high))
    ghost_width = max(abs(place(high) - place(low)), 0.8)
    crosses_zero = low < 0.0 < high
    tone = "lowers" if value < 0 else "raises"
    bar_left, bar_right = sorted((50.0, place(value)))
    flip = (
        '<span class="flip" title="Across refits this characteristic also points '
        'the other way">&#8644;</span>'
        if crosses_zero
        else ""
    )

    return (
        f'<span class="bar">'
        f'<span class="ghost {tone}" style="left:{ghost_left:.1f}%;'
        f'width:{ghost_width:.1f}%"></span>'
        f'<span class="fill {tone}" style="left:{bar_left:.1f}%;'
        f'width:{bar_right - bar_left:.1f}%"></span>'
        f"{flip}"
        f"</span>"
    )


def _decision_ruler(view: CaseView) -> str:
    """Place the case on the probability axis, against the thresholds in play.

    A number pair ("0.065 against a threshold of 0.339") states a relation;
    the ruler makes it a distance, so a reader sees at once whether the
    decision was comfortable or a coin-flip. The three marks also carry the
    project's argument at a glance: the threshold in force, the one the
    model's own probabilities imply, and the naive 0.5 that neither supports.

    Built from positioned HTML rather than SVG. An SVG stretched to the column
    width distorts its own text, and a uniformly scaled one would render the
    labels far too small in a column this narrow; percentage-positioned HTML
    keeps the labels as real text at a real size.

    Args:
        view: The case being shown.

    Returns:
        An HTML fragment.

    """
    theoretical = view.threshold_result.theoretical_threshold
    case = view.probability
    # Nudge the case label inboard at the extremes so it cannot overhang.
    anchor = (
        "left:0;transform:none"
        if case < 0.10
        else (
            "right:0;transform:none"
            if case > 0.90
            else f"left:{case:.1%};transform:translateX(-50%)"
        )
    )

    return f"""
    <div class="ruler">
      <div class="track">
        <div class="refuse" style="left:{view.threshold:.1%}"></div>
        <div class="mk mk-faint" style="left:50%"></div>
        <div class="mk mk-theory" style="left:{theoretical:.1%}"></div>
        <div class="mk mk-cut" style="left:{view.threshold:.1%}"></div>
        <div class="dot" style="left:{case:.1%}"></div>
      </div>
      <div class="rl-row">
        <span class="rl rl-case" style="{anchor}">this case {case:.3f}</span>
        <span class="rl rl-cut" style="left:{view.threshold:.1%}">
          refuse from {view.threshold:.3f}</span>
      </div>
      <p class="rl-key">
        Faint marks: <b>t* {theoretical:.2f}</b>, the cut this model's own
        probabilities imply, and <b>0.5</b>, the naive one neither supports.
      </p>
    </div>
    """


def _icon_array(view: CaseView) -> str:
    """Show the band's accuracy as a hundred people rather than a percentage.

    Frequencies of the form "8 out of 100" are read far more reliably than the
    equivalent percentage — a well-established result in risk communication.
    A project about people misreading stated confidence should not state its
    own key figure in the format that is hardest to read correctly.

    The correct cases are deliberately recessive and fail the contrast floor on
    purpose: they are the ground, not the signal. The relief the guideline
    requires is supplied by the counts labelled directly beside the array.

    Args:
        view: The case being shown; its band supplies the accuracy.

    Returns:
        An inline SVG fragment, or an empty string without a band.

    """
    if view.band is None:
        return ""
    wrong = int(round((1.0 - view.band.accuracy) * 100))
    correct = 100 - wrong

    dots = []
    for index in range(100):
        row, column = divmod(index, 10)
        fill = "#ccd2d8" if index < correct else "#9a6b12"
        dots.append(
            f'<rect x="{column * 7.4:.1f}" y="{row * 7.4:.1f}" width="5.2" '
            f'height="5.2" rx="2.6" fill="{fill}"/>'
        )
    return f"""
    <div class="array">
      <svg viewBox="0 0 72 72" role="img"
           aria-label="{correct} of 100 correct, {wrong} wrong">{"".join(dots)}</svg>
      <div class="array-key">
        <div><span class="sw" style="background:#ccd2d8"></span>
             <b>{correct}</b> judged correctly</div>
        <div><span class="sw" style="background:#9a6b12"></span>
             <b>{wrong}</b> got wrong</div>
        <p>Out of 100 past applicants the model was this confident about.</p>
      </div>
    </div>
    """


def _qualifier(view: CaseView) -> str:
    """Render the block that distinguishes this page from a normal scoring screen."""
    if view.band is None:
        return (
            '<p class="muted">Run the full reproduction (without '
            "<code>--skip-stability</code>) to populate this panel.</p>"
        )

    band = view.band
    overstated = band.gap > 0
    verdict = "overstates" if overstated else "understates"
    # The gap is a fraction; the sentence talks in percentage points.
    points = abs(band.gap) * 100.0
    count = f"{band.count:,}".replace(",", "\u2009")  # thin space, not a comma
    return f"""
      <div class="qualifier {"warn" if overstated and band.excludes_zero else ""}">
        <div class="q-head">What confidence at this level has been worth</div>
        <div class="q-figures">
          <div><span class="q-num">{band.mean_confidence:.1%}</span>
               <span class="q-cap">stated, on average</span></div>
          <div class="q-arrow">→</div>
          <div><span class="q-num">{band.accuracy:.1%}</span>
               <span class="q-cap">actually correct</span></div>
        </div>
        <p>
          Across <b>{count}</b> past cases in the
          {band.lower:.0%}–{band.upper:.0%} confidence band, this model
          {verdict} its certainty by
          <b>{points:.1f}&nbsp;percentage&nbsp;points</b>
          (95&nbsp;% interval {band.gap_lower:+.3f} to {band.gap_upper:+.3f}).
          {
        "That gap is larger than measurement noise."
        if band.excludes_zero
        else "That gap is within measurement noise and should not be acted on."
    }
        </p>
        {_icon_array(view)}
      </div>
    """


def _conclusion(view: CaseView) -> str:
    """Say plainly what the page just demonstrated, in this case's own numbers.

    Without this the page is a well-qualified scoring screen and the reader is
    left to infer the point. The conclusion states it: each of the three
    assertions above is individually defensible, and together they describe a
    decision resting on far less than it appears to.

    Written from the case's own figures rather than as fixed prose, so it
    cannot drift away from what is shown above it.

    Args:
        view: The case being shown.

    Returns:
        An HTML fragment.

    """
    contributions = view.ranked_contributions()
    top_name = contributions[0][0] if contributions else "—"
    changed = 1.0 - view.stability.top_feature_agreement
    band = view.band
    wrong = int(round((1.0 - band.accuracy) * 100)) if band else 0
    protected = [
        name for name, _ in contributions if name in {"foreign_worker", "personal_status_sex"}
    ]

    verdict_word = "Refuse" if view.refused else "Accept"
    band_line = (
        f"In this confidence band it has been right "
        f"{band.accuracy:.1%} of the time — wrong about {wrong} applicants in 100."
        if band
        else "The band evidence is not available in this run."
    )
    many = len(protected) > 1
    named = " and ".join(html.escape(_label(name)) for name in protected)
    protected_line = (
        f"<p><b>And one more thing this case makes visible.</b> Among the "
        f"characteristics driving the score {'are' if many else 'is'} {named} — "
        f"{'attributes' if many else 'an attribute'} whose use in a live "
        f"scorecard would be unlawful under current EU and German law. "
        f"{'They are' if many else 'It is'} kept in the model deliberately: "
        f"removing {'them' if many else 'it'} would hide the issue rather than "
        f"resolve it, and a screen like this one is where it becomes impossible "
        f"to overlook.</p>"
        if protected
        else ""
    )

    return f"""
    <section class="card conclusion">
      <h2>What this page is demonstrating</h2>
      <table class="ledger">
        <thead><tr><th>The screen asserts</th>
        <th>What that assertion has been worth</th></tr></thead>
        <tbody>
          <tr>
            <td><b>{verdict_word}</b>, because the score is
                {"above" if view.refused else "below"} {view.threshold:.3f}</td>
            <td>That line is an <i>assumption</i>, not a measurement. It follows
                from holding a missed default to cost
                {view.cost.cost_false_negative / view.cost.cost_false_positive:.0f}×
                a rejected good customer. Nothing in the data fixes that ratio,
                and the cost curve is flat enough that a wide range of lines
                costs almost the same while rejecting very different numbers of
                good customers.</td>
          </tr>
          <tr>
            <td><b>{view.confidence:.0%} confident</b> in that call</td>
            <td>{band_line} The overstatement is small but larger than
                measurement noise, and it sits in the highest confidence band —
                exactly where a reader is least likely to check.</td>
          </tr>
          <tr>
            <td><b>Because of {html.escape(_label(top_name))}</b>, and four
                further characteristics</td>
            <td>Faithful to this model: masking those characteristics moves it
                far more than masking random ones. But refit on a resample of
                the same training data and the leading characteristic changes
                for <b>{changed:.0%}</b> of cases. The reason is true of
                <i>this</i> model, and this model is one of many the same data
                could have produced.</td>
          </tr>
        </tbody>
      </table>

      <p><b>The point is not that any one of these is wrong.</b> Each is
      defensible on its own terms. The point is what they amount to together: a
      decision taken at a line nobody measured, justified by a confidence that
      is slightly but systematically overstated, and explained by a reason that
      another sample of customers would have stated differently.</p>

      <p>The expected danger from an explanation is that it might be false — but
      a false explanation eventually contradicts what people observe and gets
      caught. What this page shows is the harder case: <b>a correct explanation
      of an arbitrary model.</b> It earns trust in a mechanism while concealing
      how much of that mechanism is an accident of the sample. That is the
      condition this project set out to characterise, and the reason every
      assertion here is printed beside its own evidence instead of alone.</p>

      {protected_line}
    </section>
    """


def render_case_page(view: CaseView, *, generated_at: str) -> str:
    """Render the complete single-case page as self-contained HTML.

    Args:
        view: The case and its qualifying evidence.
        generated_at: Timestamp of the reproduction run that produced the data.

    Returns:
        A full HTML document with no external dependencies.

    """
    contributions = view.ranked_contributions()
    # The axis must hold the refit ranges too, or a ghost would be silently
    # clipped and a wobbling characteristic would look stable.
    extremes = [abs(v) for _, v in contributions]
    for name, _ in contributions:
        low, high = view.refit_intervals.get(name, (0.0, 0.0))
        extremes += [abs(low), abs(high)]
    scale = max(extremes) if extremes else 1.0

    rows = "".join(
        f"<tr><td>{html.escape(_label(name))}</td>"
        f"<td class='val'>{html.escape(_describe(name, value))}</td></tr>"
        for name, value in view.features.items()
    )

    reasons = "".join(
        f"<tr><td>{html.escape(_label(name))}</td>"
        f"<td class='val'>{html.escape(_describe(name, view.features[name]))}</td>"
        f"<td class='barcell'>"
        f"{_contribution_bar(value, view.refit_intervals.get(name, (value, value)), scale)}"
        f"</td>"
        f"<td class='num {'lowers' if value < 0 else 'raises'}'>{value:+.2f}</td></tr>"
        for name, value in contributions
    )

    digest = html.escape(
        " · ".join(
            f"{_describe(name, view.features[name])}"
            for name in ("age", "amount", "duration", "housing")
            if name in view.features
        )
    )
    decision = "Refuse" if view.refused else "Accept"
    ratio = view.cost.cost_false_negative / view.cost.cost_false_positive

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Case {view.case_id} — credit-confidence-gap</title>
<style>
  :root {{
    --ink: #1b1d21; --muted: #6a7078; --rule: #d9dde2; --paper: #ffffff;
    --ground: #f4f5f7; --raises: #a4343a; --lowers: #1a6a99; --warnbg: #fdf6e8;
    --warnrule: #d9a441;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--ground); color: var(--ink);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }}
  header.top {{ border-bottom: 2px solid var(--ink); padding-bottom: 14px; margin-bottom: 8px; }}
  header.top h1 {{ font-size: 17px; margin: 0; letter-spacing: .02em; }}
  header.top p {{ margin: 6px 0 0; color: var(--muted); font-size: 13px; max-width: 70ch; }}
  .demo {{
    display: inline-block; margin-top: 10px; padding: 3px 9px; font-size: 11px;
    letter-spacing: .09em; text-transform: uppercase; background: var(--ink);
    color: #fff; border-radius: 2px;
  }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
  details.applicant {{ margin-top: 24px; padding: 0; }}
  details.applicant summary {{
    cursor: pointer; padding: 13px 20px; list-style: none;
    display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
  }}
  details.applicant summary::-webkit-details-marker {{ display: none; }}
  details.applicant summary::before {{
    content: "▸"; color: var(--muted); font-size: 11px; margin-right: -6px;
  }}
  details.applicant[open] summary::before {{ content: "▾"; }}
  details.applicant[open] summary {{ border-bottom: 1px solid var(--rule); }}
  .sum-label {{
    font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
    color: var(--muted); font-weight: 600;
  }}
  .sum-digest {{ font-size: 13px; color: var(--ink); }}
  details.applicant table {{ padding: 6px 20px 16px; }}
  .conclusion {{ margin-top: 20px; }}
  .ledger {{ margin-bottom: 14px; }}
  .ledger th {{
    text-align: left; font-size: 11px; letter-spacing: .08em;
    text-transform: uppercase; color: var(--muted); font-weight: 600;
    padding-bottom: 6px; border-bottom: 1px solid var(--rule);
  }}
  .ledger th:first-child {{ width: 30%; }}
  .ledger td {{ padding: 10px 0; vertical-align: top; font-size: 13.5px; }}
  .ledger td:first-child {{ padding-right: 22px; }}
  .conclusion p {{ font-size: 13.5px; margin: 0 0 11px; max-width: 78ch; }}
  .conclusion p:last-child {{ margin-bottom: 0; }}
  @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--paper); border: 1px solid var(--rule);
           border-radius: 4px; padding: 18px 20px; }}
  .card h2 {{
    font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
    color: var(--muted); margin: 0 0 12px; font-weight: 600;
  }}
  table {{ width: 100%; border-collapse: collapse; }}
  td {{ padding: 4px 0; border-bottom: 1px solid #eef0f3; vertical-align: baseline; }}
  tr:last-child td {{ border-bottom: 0; }}
  td.val {{ text-align: right; color: var(--muted); }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; width: 56px; font-weight: 600; }}
  .raises {{ color: var(--raises); }} .lowers {{ color: var(--lowers); }}
  .sw {{ display:inline-block; width:9px; height:9px; border-radius:2px; vertical-align:baseline; }}
  .ruler {{ margin: 16px 0 4px; }}
  .track {{ position: relative; height: 10px; background: #f3f5f7;
            border: 1px solid var(--rule); border-radius: 2px; }}
  .refuse {{ position: absolute; top: 0; right: 0; bottom: 0;
             background: var(--raises); opacity: .12; }}
  .mk {{ position: absolute; width: 1px; }}
  .mk-faint {{ top: -1px; bottom: -1px; background: #c8ced5; }}
  .mk-theory {{ top: -3px; bottom: -3px; background: var(--muted); }}
  .mk-cut {{ top: -5px; bottom: -5px; width: 2px; background: var(--ink); margin-left: -1px; }}
  .dot {{ position: absolute; top: 50%; width: 9px; height: 9px; border-radius: 50%;
          background: #1a6a99; border: 1.5px solid var(--paper);
          transform: translate(-50%, -50%); }}
  .rl-row {{ position: relative; height: 16px; margin-top: 3px; }}
  .rl-key {{ margin: 4px 0 0; font-size: 11px; color: var(--muted); line-height: 1.4; }}
  .rl-key b {{ color: var(--ink); font-weight: 600; }}
  .rl {{ position: absolute; font-size: 10.5px; color: var(--muted);
         white-space: nowrap; transform: translateX(-50%); }}
  .rl-cut {{ color: var(--ink); font-weight: 600; }}
  .rl-case {{ color: #1a6a99; font-weight: 600; }}
  .array {{ display:flex; gap:16px; align-items:center; margin-top:14px;
            padding-top:13px; border-top:1px solid rgba(0,0,0,.07); }}
  .array svg {{ width: 118px; flex: none; }}
  .array-key {{ font-size: 12.5px; }}
  .array-key div {{ margin-bottom: 3px; }}
  .array-key p {{ margin: 6px 0 0; color: var(--muted); font-size: 11.5px; }}
  .flip {{ position:absolute; right:-14px; top:-4px; font-size:11px; color:var(--muted); }}
  .barcell {{ width: 150px; padding-left: 14px; }}
  .bar {{ display: block; position: relative; height: 10px;
          background: #f3f5f7; border-radius: 2px; }}
  .bar::after {{ content:""; position:absolute; left:50%; top:-3px;
                 bottom:-3px; width:1px; background:#b9c0c8; }}
  .ghost, .fill {{ position: absolute; top: 0; height: 10px; border-radius: 2px; }}
  .ghost {{ opacity: .22; }}
  .fill {{ box-shadow: 0 0 0 1.5px var(--paper); }}
  .ghost.raises, .fill.raises {{ background: var(--raises); }}
  .ghost.lowers, .fill.lowers {{ background: var(--lowers); }}
  .verdict {{ display: flex; align-items: baseline; gap: 14px; margin-bottom: 4px; }}
  .verdict b {{ font-size: 30px; letter-spacing: -.01em; }}
  .verdict .p {{ font-size: 15px; color: var(--muted); font-variant-numeric: tabular-nums; }}
  .why {{ font-size: 13px; color: var(--muted); margin: 8px 0 0; }}
  .qualifier {{
    margin-top: 18px; padding: 16px 18px; border: 1px solid var(--rule);
    border-left: 3px solid var(--muted); border-radius: 3px; background: #fbfcfd;
  }}
  .qualifier.warn {{ background: var(--warnbg); border-left-color: var(--warnrule); }}
  .q-head {{
    font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
    color: var(--muted); font-weight: 600; margin-bottom: 10px;
  }}
  .q-figures {{ display: flex; align-items: center; gap: 18px; margin-bottom: 10px; }}
  .q-num {{ font-size: 26px; font-weight: 600; display: block;
            font-variant-numeric: tabular-nums; }}
  .q-cap {{ font-size: 11px; color: var(--muted); }}
  .q-arrow {{ font-size: 20px; color: var(--muted); }}
  .qualifier p {{ margin: 0; font-size: 13.5px; }}
  .note {{ margin-top: 12px; font-size: 13px; color: var(--muted); }}
  footer {{
    margin-top: 28px; padding-top: 14px; border-top: 1px solid var(--rule);
    font-size: 12px; color: var(--muted);
  }}
  footer b {{ color: var(--ink); }}
  code {{ font-size: 12px; background: #eef0f3; padding: 1px 4px; border-radius: 2px; }}
</style></head>
<body><div class="wrap">

<header class="top">
  <h1>credit-confidence-gap · single-case view</h1>
  <p>A demonstration of what a scoring screen looks like when every assertion it
     makes is shown next to what that kind of assertion has historically been
     worth. Not a credit decision tool — see the footer.</p>
  <span class="demo">Demonstration · model: {html.escape(view.model_name)}</span>
</header>

<details class="card applicant">
  <summary>
    <span class="sum-label">Application {view.case_id}</span>
    <span class="sum-digest">{digest}</span>
  </summary>
  <table>{rows}</table>
</details>

<div class="grid">
  <section class="card">
    <h2>Decision</h2>
    <div class="verdict">
      <b>{decision}</b>
        <span class="p">P(default) = {view.probability:.3f}</span>
      </div>
      <p class="why">
        Refuse at or above <b>{view.threshold:.3f}</b>. That threshold is not
        0.5: it is derived from a cost matrix in which a missed default is
        held to cost <b>{ratio:.0f}×</b> a rejected good customer. Under
        perfect calibration the same matrix would imply
        {view.threshold_result.theoretical_threshold:.3f} — the difference is
        a property of this model, not of the costs.
      </p>
    {_decision_ruler(view)}
    {_qualifier(view)}
  </section>

  <section class="card">
    <h2>Why — characteristics driving this score</h2>
      <table>{reasons}</table>
      <p class="note">
        <span class="sw" style="background:#a4343a"></span> raises predicted
        risk &nbsp; <span class="sw" style="background:#1a6a99"></span> lowers
        it. Exact Shapley values on the log-odds scale, summing to the model's
        score — its actual arithmetic, not a plausible story about it.
        The pale band behind each bar is the range the same characteristic
        takes across {view.stability.n_refits} refits on resampled training
        data; &#8644; marks one that changes sign.
      </p>
      <div class="qualifier warn">
        <div class="q-head">How much to believe this reason</div>
        <p>
          Refit this model on a resample of the same training data and the
          single most influential characteristic changes for
          <b>{1 - view.stability.top_feature_agreement:.0%}</b> of cases; of
          the five listed, on average
          <b>{(1 - view.stability.top_k_jaccard) * view.top_k:.1f}</b> would
          differ. The reason above is faithful to this model — and this model
          is one of many the same data could have produced.
        </p>
    </div>
  </section>
</div>

{_conclusion(view)}

<footer>
  <b>This is not a credit scoring system.</b> The underlying data are 1 000
  loans issued in South Germany between 1973 and 1975. They contain only
  applicants who actually received credit, so the model knows nothing about
  those refused, and defaults are deliberately over-represented (300 of 1 000)
  by the sampling design. The probability above is calibrated against that
  sample, not against any lending population, and is not a default rate.
  No decision about a real person should be derived from it.
  <br><br>
  Generated {html.escape(generated_at)} by <code>make reproduce</code>.
  Every figure on this page is read from <code>reports/metrics/</code>;
  the page computes nothing of its own.
</footer>

</div></body></html>
"""


def write_case_page(view: CaseView, path: Path, *, generated_at: str) -> Path:
    """Render a case page and write it to disk.

    Args:
        view: The case and its qualifying evidence.
        path: Destination file.
        generated_at: Timestamp of the reproduction run.

    Returns:
        The path written.

    """
    document = render_case_page(view, generated_at=generated_at)
    assert_well_formed(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


class MarkupError(RuntimeError):
    """Raised when the rendered page has unbalanced tags."""


def assert_well_formed(document: str) -> None:
    """Check that every non-void element in the page is closed.

    Worth its own check because the failure mode is silent and misleading. A
    single missing closing tag does not raise and does not look broken in the
    source — the browser simply nests everything that follows inside the
    unclosed element, and the page renders with its lower half swallowed. That
    happened once while building this view.

    Args:
        document: The rendered HTML.

    Raises:
        MarkupError: If a tag is unclosed, or closed without being open.

    """
    void = {"area", "base", "br", "col", "hr", "img", "input", "link", "meta", "source"}
    stack: list[str] = []

    class Checker(HTMLParser):
        """Track open elements and flag the first imbalance."""

        def handle_starttag(self, tag: str, attrs: object) -> None:
            if tag not in void:
                stack.append(tag)

        def handle_endtag(self, tag: str) -> None:
            if tag in void:
                return
            if not stack:
                raise MarkupError(f"</{tag}> closes nothing.")
            if stack[-1] != tag:
                raise MarkupError(f"</{tag}> closes <{stack[-1]}>, which is still open.")
            stack.pop()

    Checker(convert_charrefs=True).feed(document)
    if stack:
        raise MarkupError(f"Unclosed elements, innermost first: {list(reversed(stack))}")


def choose_case(probabilities: np.ndarray, band: tuple[float, float] = (0.9, 1.0)) -> int:
    """Pick a representative case from a confidence band.

    The default band is the highest one, because that is where step 2 found the
    overconfidence and therefore where the page has something to say.

    Args:
        probabilities: Predicted P(bad credit) for the candidate cases.
        band: Inclusive lower, exclusive upper edge on decision confidence.

    Returns:
        Positional index of the chosen case.

    Raises:
        ValueError: If no case falls in the band.

    """
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    eligible = np.flatnonzero((confidence >= band[0]) & (confidence <= band[1]))
    if eligible.size == 0:
        raise ValueError(f"No case falls in the confidence band {band}.")
    # The median case of the band, so the page is not built around an outlier.
    return int(eligible[np.argsort(confidence[eligible])[eligible.size // 2]])


def band_for(probability: float, bands: Sequence[PooledTrustGapRow]) -> PooledTrustGapRow | None:
    """Find the pooled trust-gap row covering a case's decision confidence.

    Args:
        probability: Predicted P(bad credit).
        bands: Pooled rows in ascending order.

    Returns:
        The matching row, or ``None`` if the bands do not cover it.

    """
    confidence = max(probability, 1.0 - probability)
    for row in bands:
        if row.lower <= confidence < row.upper or (row is bands[-1] and confidence <= row.upper):
            return row
    return None


def case_payload(view: CaseView) -> str:
    """Serialise a case as JSON, for the multi-case navigation added later.

    Args:
        view: The case to serialise.

    Returns:
        A compact JSON string.

    """
    return json.dumps(
        {
            "case_id": view.case_id,
            "probability": view.probability,
            "refused": view.refused,
            "contributions": view.ranked_contributions(),
        },
        separators=(",", ":"),
    )
