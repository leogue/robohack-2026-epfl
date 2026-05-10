"""Shared ESP sensor state helpers.

Only one process should open COM7. Use esp_serial_broker.py for that, then
other processes can read the latest sensor sample from this JSON file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


STATE_PATH = Path(__file__).resolve().parent / "outputs" / "esp_latest.json"


def parse_esp_line(line: str) -> dict | None:
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None

    try:
        light = int(parts[0])
        temp = float(parts[1])
        moisture = int(parts[2])
    except ValueError:
        return None

    now = time.time()
    return {
        "light": light,
        "temp": temp,
        "moisture": moisture,
        "temp_f": temp * 9 / 5 + 32,
        "raw": line.strip(),
        "timestamp": now,
        "timestamp_s": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
    }


def write_latest_sample(sample: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(sample)
    for attempt in range(20):
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.025)


def read_latest_sample(path: Path = STATE_PATH, max_age_s: float | None = None) -> dict | None:
    sample = None
    for _ in range(5):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                sample = json.load(fh)
            break
        except PermissionError:
            time.sleep(0.01)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    if sample is None:
        return None

    if max_age_s is not None:
        timestamp = float(sample.get("timestamp", 0))
        if time.time() - timestamp > max_age_s:
            return None

    return sample
