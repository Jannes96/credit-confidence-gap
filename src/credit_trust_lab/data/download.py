"""Reproducible acquisition of the raw dataset.

One command from an empty clone to a verified raw file. The checksums below
were computed from the archive published by the UCI Machine Learning Repository
(dataset 573, *South German Credit - UPDATE*) and are pinned so that a silent
upstream change, a truncated download, or a substituted file fails loudly
instead of propagating into results.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from credit_trust_lab.config import RAW_DIR

DATASET_URL = "https://archive.ics.uci.edu/static/public/573/south+german+credit+update.zip"

ARCHIVE_SHA256 = "0b40d40eb7321693d559e247a556f88a6cc8df8489c3cb2ae084db7592584551"

DATA_MEMBER = "SouthGermanCredit.asc"
DATA_SHA256 = "5f363343f356ca38a0236baab849e472846399b2176ccc5bd686483dd8a7562f"

CODETABLE_MEMBER = "codetable.txt"
CODETABLE_SHA256 = "0040e7a173a06d1167447b2bbc4315ee1380fa3c25db321b9fd7d97ada0721bf"

_DOWNLOAD_TIMEOUT_SECONDS = 60


class ChecksumError(RuntimeError):
    """Raised when a downloaded or cached file does not match its pinned digest."""


@dataclass(frozen=True)
class RawDataset:
    """Locations of the verified raw files on disk.

    Attributes:
        data_path: Whitespace-separated raw data file with a header row.
        codetable_path: Upstream coding table, kept for provenance so the
            codebook in :mod:`credit_trust_lab.data.codebook` can be audited
            against its source.

    """

    data_path: Path
    codetable_path: Path


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.

    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    """Raise :class:`ChecksumError` unless ``path`` hashes to ``expected``."""
    actual = sha256_of(path)
    if actual != expected:
        raise ChecksumError(
            f"Checksum mismatch for {path}.\n  expected: {expected}\n  actual:   {actual}\n"
            "Delete the file and re-run to refetch; if it persists, the upstream "
            "file has changed and the pinned digest must be reviewed deliberately."
        )


def download_raw(
    destination: Path | None = None,
    *,
    force: bool = False,
    url: str = DATASET_URL,
) -> RawDataset:
    """Fetch, verify and extract the corrected South German Credit dataset.

    Idempotent: an already-present file that matches its pinned digest is
    reused, so repeated runs do not hit the network.

    Args:
        destination: Directory for the extracted files. Defaults to ``data/raw``.
        force: Refetch even if verified local copies exist.
        url: Archive URL. Overridable so tests can serve a local fixture.

    Returns:
        Paths to the verified raw files.

    Raises:
        ChecksumError: If the archive or any extracted member fails verification.

    """
    destination = destination or RAW_DIR
    destination.mkdir(parents=True, exist_ok=True)

    data_path = destination / DATA_MEMBER
    codetable_path = destination / CODETABLE_MEMBER

    if not force and data_path.exists() and codetable_path.exists():
        try:
            _verify(data_path, DATA_SHA256)
            _verify(codetable_path, CODETABLE_SHA256)
        except ChecksumError:
            pass  # Cached copy is corrupt; fall through and refetch.
        else:
            return RawDataset(data_path=data_path, codetable_path=codetable_path)

    with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
        payload = response.read()

    actual_archive_digest = hashlib.sha256(payload).hexdigest()
    if actual_archive_digest != ARCHIVE_SHA256:
        raise ChecksumError(
            f"Checksum mismatch for the archive at {url}.\n"
            f"  expected: {ARCHIVE_SHA256}\n  actual:   {actual_archive_digest}"
        )

    # The archive uses bzip2 compression, which some system unzip builds cannot
    # read; Python's zipfile handles it, which is why extraction happens here.
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member, target in ((DATA_MEMBER, data_path), (CODETABLE_MEMBER, codetable_path)):
            target.write_bytes(archive.read(member))

    _verify(data_path, DATA_SHA256)
    _verify(codetable_path, CODETABLE_SHA256)

    return RawDataset(data_path=data_path, codetable_path=codetable_path)


def main() -> None:
    """Entry point for ``python -m credit_trust_lab.data.download``."""
    dataset = download_raw()
    print(f"Verified raw data: {dataset.data_path}")
    print(f"Verified codetable: {dataset.codetable_path}")


if __name__ == "__main__":
    main()
