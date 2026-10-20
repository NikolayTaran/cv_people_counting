#!/usr/bin/env python3
"""Copy the completed second-approach runs into portable report artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "second_approach"
REPORTS = ROOT / "reports"
SOURCES = {
    "baseline_development": "second_approach_baseline_development.json",
    "development": "second_approach_ensemble_development.json",
    "baseline_comparison": "second_approach_baseline_comparison.json",
    "comparison": "second_approach_ensemble_comparison.json",
}


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    for run_name, report_name in SOURCES.items():
        source = RUNS / run_name / "results.json"
        result = json.loads(source.read_text(encoding="utf-8"))
        detector = result["detector"]
        detector["model"] = Path(detector["model"]).name
        if detector["model"] == "people_yolo11s_mot.pt":
            detector["model"] = "models/people_yolo11s_mot.pt"
        if "ensemble_with" in detector:
            detector["ensemble_with"] = Path(detector["ensemble_with"]).name
        target = REPORTS / report_name
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
