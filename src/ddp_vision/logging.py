from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

class LocalLogger:
    """Append-only metrics logger for human-readable local training history."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = log_dir / "metrics.jsonl"
        self.csv_path = log_dir / "metrics.csv"
    
    def log(self, metrics: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dump(metrics, sort_keys=True) + "\n")
        
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)