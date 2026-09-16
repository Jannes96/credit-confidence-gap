"""The coding table of the South German Credit data, expressed as code.

This module is the single source of truth for what every integer in the raw
file means. It is code and not a comment because the level assignment is the
one place where this dataset is routinely got wrong, and a wrong assignment
produces a model that trains, scores well, and means something different from
what its author believes.

Provenance
----------
Grömping, U. (2019): *South German Credit Data: Correcting a Widely Used Data
Set*. Report 4/2019, Beuth University of Applied Sciences Berlin. The data
originate from Fahrmeir/Hamerle and were collected 1973-1975; the integer level
codes are the P2 scores of Häußler (1979), constructed so that larger is
better.

Why the corrected edition
-------------------------
The widely circulated UCI *Statlog (German Credit Data)* edition mislabels
several variables. The best-known case is ``gastarb`` / ``foreign_worker``,
whose level labels are inverted there: the UCI documentation reads the majority
level as "foreign worker = yes", which would make ~96% of a South German bank's
1970s loan book foreign workers. In the corrected edition used here, ``1`` is
*yes* and ``2`` is *no*, and the minority is the foreign one.
``tests/test_codebook.py`` pins this orientation so that a future refactor
cannot quietly reintroduce the legacy coding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class FeatureKind(str, Enum):
    """How a column is to be treated by the preprocessing layer.

    Attributes:
        NUMERIC: A genuine quantitative measurement.
        ORDINAL: Integer-coded levels with a defensible monotone order; kept as
            a single scaled column.
        NOMINAL: Integer-coded levels without a defensible order; one-hot
            encoded.
        BINARY: Exactly two levels; one-hot encoded to a single indicator.

    """

    NUMERIC = "numeric"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"
    BINARY = "binary"


@dataclass(frozen=True)
class Feature:
    """One column of the raw file, with its meaning attached.

    Attributes:
        raw_name: Column name in ``SouthGermanCredit.asc`` (German, from the
            Fahrmeir/Hamerle book).
        name: Speaking English name used everywhere downstream.
        kind: Treatment by the preprocessing layer.
        levels: Integer code to label. Empty for :attr:`FeatureKind.NUMERIC`.
        description: What the variable measures.
        note: Caveat that a reader must know to interpret the column honestly.
            Surfaced in the data card; ``None`` when there is nothing to warn
            about.

    """

    raw_name: str
    name: str
    kind: FeatureKind
    levels: Mapping[int, str]
    description: str
    note: str | None = None

    @property
    def codes(self) -> tuple[int, ...]:
        """Declared level codes in ascending order (empty for numeric columns)."""
        return tuple(sorted(self.levels))

    @property
    def is_categorical(self) -> bool:
        """Whether the column is one-hot encoded downstream."""
        return self.kind in (FeatureKind.NOMINAL, FeatureKind.BINARY)


def _levels(pairs: dict[int, str]) -> Mapping[int, str]:
    """Return an immutable level mapping, so the codebook cannot be mutated."""
    return MappingProxyType(dict(pairs))


# --------------------------------------------------------------------------
# Ordinal assignment: what "larger is better" is and is not allowed to mean
# --------------------------------------------------------------------------
# The P2 codes are monotone by construction, so an ordinal reading is the
# design intent of the data. Two qualifications are recorded per feature in
# ``note`` and are not cosmetic:
#
#   1. For ``status``, ``savings`` and ``property``, the lowest code is not the
#      bottom of a scale but an *absent or unknown* state ("no checking
#      account"). Treating it as the bottom rung asserts that having no
#      checking account is worse than being overdrawn. That is an assumption
#      of the 1970s scorecard, not a measurement.
#   2. ``credit_history`` is classified NOMINAL despite carrying P2 codes,
#      because its levels mix two different dimensions (payment behaviour and
#      the mere existence of a credit history: level 2 is "no credits taken").
#      One-hot costs four columns and buys an assumption we do not have to make.
# --------------------------------------------------------------------------

FEATURES: tuple[Feature, ...] = (
    Feature(
        raw_name="laufkont",
        name="status",
        kind=FeatureKind.ORDINAL,
        levels=_levels(
            {
                1: "no checking account",
                2: "< 0 DM",
                3: "0 <= ... < 200 DM",
                4: ">= 200 DM / salary assignment for at least 1 year",
            }
        ),
        description="Status of the applicant's checking account with the bank.",
        note=(
            "Level 1 is an absence of an account, not the low end of a balance "
            "scale; the ordinal treatment places it below an overdrawn account."
        ),
    ),
    Feature(
        raw_name="laufzeit",
        name="duration",
        kind=FeatureKind.NUMERIC,
        levels=_levels({}),
        description="Credit duration in months.",
    ),
    Feature(
        raw_name="moral",
        name="credit_history",
        kind=FeatureKind.NOMINAL,
        levels=_levels(
            {
                0: "delay in paying off in the past",
                1: "critical account / other credits elsewhere",
                2: "no credits taken / all credits paid back duly",
                3: "existing credits paid back duly till now",
                4: "all credits at this bank paid back duly",
            }
        ),
        description="Payment history of the applicant's previous credits.",
        note=(
            "Carries monotone P2 codes but conflates payment behaviour with the "
            "existence of a credit history (level 2 = no credits taken), so it is "
            "treated as unordered."
        ),
    ),
    Feature(
        raw_name="verw",
        name="purpose",
        kind=FeatureKind.NOMINAL,
        levels=_levels(
            {
                0: "others",
                1: "car (new)",
                2: "car (used)",
                3: "furniture/equipment",
                4: "radio/television",
                5: "domestic appliances",
                6: "repairs",
                7: "education",
                8: "vacation",
                9: "retraining",
                10: "business",
            }
        ),
        description="Declared purpose of the credit.",
        note=(
            "Level 7 (education) is declared in the coding table but unobserved: "
            "it has zero rows. Its one-hot column is therefore all zeros, which "
            "is why the encoder takes its categories from this codebook rather "
            "than from whichever levels a given split happens to contain."
        ),
    ),
    Feature(
        raw_name="hoehe",
        name="amount",
        kind=FeatureKind.NUMERIC,
        levels=_levels({}),
        description="Credit amount in Deutsche Mark (1973-1975 prices).",
    ),
    Feature(
        raw_name="sparkont",
        name="savings",
        kind=FeatureKind.ORDINAL,
        levels=_levels(
            {
                1: "unknown / no savings account",
                2: "< 100 DM",
                3: "100 <= ... < 500 DM",
                4: "500 <= ... < 1000 DM",
                5: ">= 1000 DM",
            }
        ),
        description="Balance of the applicant's savings account.",
        note="Level 1 mixes 'unknown' with 'no account'; it is not a measured balance.",
    ),
    Feature(
        raw_name="beszeit",
        name="employment_duration",
        kind=FeatureKind.ORDINAL,
        levels=_levels(
            {
                1: "unemployed",
                2: "< 1 yr",
                3: "1 <= ... < 4 yrs",
                4: "4 <= ... < 7 yrs",
                5: ">= 7 yrs",
            }
        ),
        description="Duration of the applicant's current employment.",
    ),
    Feature(
        raw_name="rate",
        name="installment_rate",
        kind=FeatureKind.ORDINAL,
        levels=_levels(
            {
                1: ">= 35",
                2: "25 <= ... < 35",
                3: "20 <= ... < 25",
                4: "< 20",
            }
        ),
        description="Installment as a percentage of disposable income, banded.",
        note="Codes run opposite to the percentage: a higher code means a lower burden.",
    ),
    Feature(
        raw_name="famges",
        name="personal_status_sex",
        kind=FeatureKind.NOMINAL,
        levels=_levels(
            {
                1: "male : divorced/separated",
                2: "female : non-single or male : single",
                3: "male : married/widowed",
                4: "female : single",
            }
        ),
        description="Combined marital status and sex of the applicant.",
        note=(
            "Encodes a protected attribute and confounds it with marital status, "
            "so the levels are not separable. Retained because removing it would "
            "hide the issue rather than resolve it; using it in a live scorecard "
            "would be unlawful under current EU and German law."
        ),
    ),
    Feature(
        raw_name="buerge",
        name="other_debtors",
        kind=FeatureKind.NOMINAL,
        levels=_levels({1: "none", 2: "co-applicant", 3: "guarantor"}),
        description="Other debtors or guarantors on the credit.",
    ),
    Feature(
        raw_name="wohnzeit",
        name="present_residence",
        kind=FeatureKind.ORDINAL,
        levels=_levels(
            {
                1: "< 1 yr",
                2: "1 <= ... < 4 yrs",
                3: "4 <= ... < 7 yrs",
                4: ">= 7 yrs",
            }
        ),
        description="Time at the applicant's present address, banded.",
    ),
    Feature(
        raw_name="verm",
        name="property",
        kind=FeatureKind.ORDINAL,
        levels=_levels(
            {
                1: "unknown / no property",
                2: "car or other",
                3: "building society savings agreement / life insurance",
                4: "real estate",
            }
        ),
        description="Most valuable available property of the applicant.",
        note="Level 1 mixes 'unknown' with 'none'.",
    ),
    Feature(
        raw_name="alter",
        name="age",
        kind=FeatureKind.NUMERIC,
        levels=_levels({}),
        description="Age of the applicant in years.",
    ),
    Feature(
        raw_name="weitkred",
        name="other_installment_plans",
        kind=FeatureKind.NOMINAL,
        levels=_levels({1: "bank", 2: "stores", 3: "none"}),
        description="Other installment plans held outside this bank.",
    ),
    Feature(
        raw_name="wohn",
        name="housing",
        kind=FeatureKind.NOMINAL,
        levels=_levels({1: "for free", 2: "rent", 3: "own"}),
        description="Housing situation of the applicant.",
    ),
    Feature(
        raw_name="bishkred",
        name="number_credits",
        kind=FeatureKind.ORDINAL,
        levels=_levels({1: "1", 2: "2-3", 3: "4-5", 4: ">= 6"}),
        description="Number of credits held at this bank, including the current one.",
        note=(
            "Quantitative in origin but banded in the data; more credits is coded "
            "as a higher level, which is not obviously 'better'."
        ),
    ),
    Feature(
        raw_name="beruf",
        name="job",
        kind=FeatureKind.ORDINAL,
        levels=_levels(
            {
                1: "unemployed / unskilled - non-resident",
                2: "unskilled - resident",
                3: "skilled employee / official",
                4: "manager / self-employed / highly qualified employee",
            }
        ),
        description="Qualification level of the applicant's occupation.",
        note="Level 1 confounds qualification with residency status.",
    ),
    Feature(
        raw_name="pers",
        name="people_liable",
        kind=FeatureKind.BINARY,
        levels=_levels({1: "3 or more", 2: "0 to 2"}),
        description="Number of persons the applicant is liable to provide maintenance for.",
        note=(
            "Corrected edition: the code order is opposite to the legacy UCI file. "
            "Higher code = fewer dependants."
        ),
    ),
    Feature(
        raw_name="telef",
        name="telephone",
        kind=FeatureKind.BINARY,
        levels=_levels({1: "no", 2: "yes (under customer name)"}),
        description="Whether a telephone is registered under the applicant's name.",
        note="In 1973-1975 a private telephone line was a wealth proxy, not a given.",
    ),
    Feature(
        raw_name="gastarb",
        name="foreign_worker",
        kind=FeatureKind.BINARY,
        levels=_levels({1: "yes", 2: "no"}),
        description="Whether the applicant is a foreign worker ('Gastarbeiter').",
        note=(
            "The legacy UCI edition inverts these labels. This orientation "
            "(1 = yes, 2 = no) is the corrected one and is pinned by a test. "
            "Also a protected-attribute proxy under current law."
        ),
    ),
)


@dataclass(frozen=True)
class Target:
    """The response variable, in the orientation used by this project.

    The raw column ``kredit`` codes 1 = good and 0 = bad. Everything downstream
    models the *default event* as the positive class (y=1 means bad credit),
    because that is the event the cost matrix prices and the event a
    precision-recall curve should be read against. :func:`encode` performs that
    flip in exactly one place.
    """

    raw_name: str = "kredit"
    name: str = "credit_risk"
    raw_levels: Mapping[int, str] = field(default_factory=lambda: _levels({0: "bad", 1: "good"}))
    positive_label: str = "bad"

    @staticmethod
    def encode(raw_values: object) -> object:
        """Flip the raw good/bad coding into the modelling orientation.

        Args:
            raw_values: Raw ``kredit`` values (1 = good, 0 = bad); anything
                supporting ``1 - x``, in practice a numpy array or pandas
                Series.

        Returns:
            Values in which 1 denotes bad credit (the default event).

        """
        return 1 - raw_values  # type: ignore[operator]


TARGET = Target()

RAW_COLUMN_ORDER: tuple[str, ...] = tuple(f.raw_name for f in FEATURES) + (TARGET.raw_name,)
"""Column order of ``SouthGermanCredit.asc``, used to validate the raw file."""

FEATURES_BY_NAME: Mapping[str, Feature] = MappingProxyType({f.name: f for f in FEATURES})

RENAME_MAP: Mapping[str, str] = MappingProxyType(
    {f.raw_name: f.name for f in FEATURES} | {TARGET.raw_name: TARGET.name}
)


def feature_names(kind: FeatureKind | None = None) -> list[str]:
    """List feature names, optionally restricted to one kind.

    Args:
        kind: If given, return only features of this kind.

    Returns:
        Feature names in codebook order.

    """
    return [f.name for f in FEATURES if kind is None or f.kind is kind]


def describe_level(feature_name: str, code: int) -> str:
    """Translate an integer code into its human-readable label.

    Args:
        feature_name: Speaking feature name, e.g. ``"savings"``.
        code: Integer code as found in the raw file.

    Returns:
        The label for that code.

    Raises:
        KeyError: If the feature is unknown or the code is not declared for it.

    """
    feature = FEATURES_BY_NAME[feature_name]
    if feature.kind is FeatureKind.NUMERIC:
        raise KeyError(f"{feature_name!r} is numeric and has no level labels.")
    if code not in feature.levels:
        raise KeyError(f"Code {code!r} is not declared for feature {feature_name!r}.")
    return feature.levels[code]
