"""Download the Kaggle release into this project's ignored data cache."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
CACHE = PROJECT / "data" / "kagglehub_cache"
os.environ["KAGGLEHUB_CACHE"] = str(CACHE)

import kagglehub  # noqa: E402  # cache must be configured before import


def main() -> None:
    path = Path(kagglehub.competition_download("2026-ieee-big-data-traffic-flow-bench"))
    candidates = (path, path / "kaggle_public")
    release = next((p for p in candidates if (p / "corridors").is_dir()
                    and (p / "submission_key.csv").is_file()), None)
    if release is None:
        raise RuntimeError(f"Download returned {path}, but the expected release root was not found")
    location = release.resolve()
    (PROJECT / "data" / "release_root.txt").write_text(str(location), encoding="utf-8")
    print(f"Competition files: {path}")
    print(f"Use --release-root {location}")


if __name__ == "__main__":
    main()
