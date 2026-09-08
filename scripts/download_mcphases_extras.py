#!/usr/bin/env python3
"""Download mid-size mcPHASES Fitbit tables into data/raw/.

Requires PhysioNet credentials (DUA already signed for the project):

  export PHYSIONET_USER=skykauf
  export PHYSIONET_PASSWORD='...'
  python scripts/download_mcphases_extras.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

TABLES = (
    "sleep.csv",
    "sleep_score.csv",
    "heart_rate_variability_details.csv",
    "respiratory_rate_summary.csv",
    "stress_score.csv",
    "wrist_temperature.csv",
)

BASE = "https://physionet.org/files/mcphases/1.0.0/"


def main() -> int:
    user = os.environ.get("PHYSIONET_USER")
    password = os.environ.get("PHYSIONET_PASSWORD")
    if not user or not password:
        print(
            "Set PHYSIONET_USER and PHYSIONET_PASSWORD, then re-run.",
            file=sys.stderr,
        )
        return 1

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, "https://physionet.org/", user, password)
    opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(password_mgr))

    for name in TABLES:
        url = BASE + name
        dest = out_dir / name
        print(f"GET {url} -> {dest}")
        try:
            with opener.open(url) as resp, dest.open("wb") as fh:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
            print(f"  ok ({dest.stat().st_size:,} bytes)")
        except Exception as exc:  # noqa: BLE001 — surface download errors clearly
            print(f"  FAILED: {exc}", file=sys.stderr)
            return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
