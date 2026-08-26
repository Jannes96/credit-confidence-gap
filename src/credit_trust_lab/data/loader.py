"""Load the raw file into a validated, speaking-named DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_trust_lab.data.codebook import (
    FEATURES,
    RAW_COLUMN_ORDER,
    RENAME_MAP,
    TARGET,
    FeatureKind,
)
from credit_trust_lab.data.download import RawDataset, download_raw

EXPECTED_N_ROWS = 1000
EXPECTED_N_BAD = 300


class DataValidationError(RuntimeError):
    """Raised when the raw file does not match the codebook."""


def load_raw_frame(path: Path) -> pd.DataFrame:
    """Read ``SouthGermanCredit.asc`` without interpreting it.

    Args:
        path: Path to the whitespace-separated raw file.

    Returns:
        DataFrame with the original German column names and integer codes.

    Raises:
        DataValidationError: If the column set or order differs from the
            codebook's expectation.

    """
    frame = pd.read_csv(path, sep=r"\s+", header=0)
    if tuple(frame.columns) != RAW_COLUMN_ORDER:
        raise DataValidationError(
            "Raw file columns do not match the codebook.\n"
            f"  expected: {RAW_COLUMN_ORDER}\n  found:    {tuple(frame.columns)}"
        )
    return frame


def validate_codes(frame: pd.DataFrame) -> None:
    """Check every categorical column against its declared levels.

    This is the guard that turns the codebook from documentation into a
    contract: an undeclared code means the file is not the edition this project
    was written against.

    Args:
        frame: Frame with speaking column names.

    Raises:
        DataValidationError: On undeclared codes, unexpected row counts, or an
            unexpected class balance.

    """
    problems: list[str] = []

    if len(frame) != EXPECTED_N_ROWS:
        problems.append(f"Expected {EXPECTED_N_ROWS} rows, found {len(frame)}.")

    for feature in FEATURES:
        column = frame[feature.name]
        if feature.kind is FeatureKind.NUMERIC:
            if column.isna().any():
                problems.append(f"{feature.name}: contains missing values.")
            if (column < 0).any():
                problems.append(f"{feature.name}: contains negative values.")
            continue
        observed = set(column.unique().tolist())
        undeclared = sorted(observed - set(feature.codes))
        if undeclared:
            problems.append(
                f"{feature.name}: codes {undeclared} are not declared "
                f"(declared: {list(feature.codes)})."
            )

    target = frame[TARGET.name]
    undeclared_target = sorted(set(target.unique().tolist()) - {0, 1})
    if undeclared_target:
        problems.append(f"{TARGET.name}: unexpected values {undeclared_target}.")
    else:
        n_bad = int((target == 1).sum())
        if n_bad != EXPECTED_N_BAD:
            problems.append(
                f"Expected {EXPECTED_N_BAD} bad credits (the stratified sample "
                f"design), found {n_bad}."
            )

    if problems:
        raise DataValidationError(
            "Raw data failed codebook validation:\n- " + "\n- ".join(problems)
        )


def load_dataset(
    raw: RawDataset | None = None, *, validate: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset in modelling orientation.

    Args:
        raw: Verified raw files. Downloaded on demand when omitted.
        validate: Whether to enforce the codebook contract.

    Returns:
        ``(X, y)`` where ``X`` holds the 20 predictors under speaking names and
        integer codes, and ``y`` is 1 for *bad credit* (the default event).

    """
    raw = raw or download_raw()
    frame = load_raw_frame(raw.data_path).rename(columns=RENAME_MAP)
    frame[TARGET.name] = TARGET.encode(frame[TARGET.name])

    if validate:
        validate_codes(frame)

    features = frame[[f.name for f in FEATURES]].copy()
    target = frame[TARGET.name].astype(int).rename(TARGET.name)
    return features, target


def data_card() -> str:
    """Render the codebook and its caveats as human-readable text.

    Written to ``reports/metrics/data_card.txt`` by the reproduction script, so
    that the notes attached to individual features travel with the results
    instead of living only in source code.

    Returns:
        A plain-text data card.

    """
    lines = [
        "South German Credit (Groemping 2019, corrected edition)",
        "=" * 72,
        f"Rows: {EXPECTED_N_ROWS}  |  bad credits: {EXPECTED_N_BAD} (stratified by design)",
        "Positive class (y=1): bad credit / default event",
        "Collection period: 1973-1975",
        "",
        "Features",
        "-" * 72,
    ]
    for feature in FEATURES:
        lines.append(f"{feature.name}  [{feature.kind.value}]  (raw: {feature.raw_name})")
        lines.append(f"    {feature.description}")
        for code in feature.codes:
            lines.append(f"      {code:>2} = {feature.levels[code]}")
        if feature.note:
            lines.append(f"    NOTE: {feature.note}")
        lines.append("")
    return "\n".join(lines)
