#!/usr/bin/env python3
"""
read.py

Usage:
    python read.py quality.jsonl combined_views.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

###############################################################################
# 1. Helpers
###############################################################################

def build_view_mapping(view_json_path: Path) -> dict[str, str]:
    """
    Load combined_views.json and return a dict that maps the *PNG* filename
    string used in the JSONL to its canonical view label.
    """
    with open(view_json_path, "r", encoding="utf-8") as fp:
        raw_map = json.load(fp)

    view_map: dict[str, str] = {}
    for dcmpath, view_list in raw_map.items():
        # Canonical key transformation: '/' → '_' and '.dcm' → '.png'
        key = dcmpath.replace("/", "_").replace(".dcm", ".png")
        # combined_views.json stores the view name(s) in a list -> take first
        view_map[key] = view_list[0] if view_list else "Unknown"

    return view_map

###############################################################################
# 2. Main analysis
###############################################################################

def analyze(jsonl_path: Path, view_json_path: Path):
    # ------------------------------------------------------------------ setup
    view_map = build_view_mapping(view_json_path)

    total_by_view     = Counter()
    correct_by_view   = Counter()
    uncertain_by_view = Counter()

    global_total     = 0
    global_correct   = 0
    global_uncertain = 0

    # ----------------------------------------------------------- stream parse
    with open(jsonl_path, "r", encoding="utf-8") as infile:
        for lineno, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"⚠️  [line {lineno}] skipped malformed JSON: {exc}", file=sys.stderr)
                continue

            fname = record.get("filename", "")
            view  = view_map.get(fname, "Unknown")

            total_by_view[view] += 1
            global_total        += 1

            labels = set(record.get("label", []))  # faster membership tests
            if "Correct" in labels:
                correct_by_view[view] += 1
                global_correct        += 1
            if "Uncertain" in labels:
                uncertain_by_view[view] += 1
                global_uncertain       += 1

    # ---------------------------------------------------------------- report
    overall_acc = global_correct / (global_total-global_uncertain) if global_total else 0.0

    print("\n===== Overall =====")
    print(f"Total samples : {global_total:,}")
    print(f"Correct       : {global_correct:,}")
    print(f"Uncertain     : {global_uncertain:,}")
    print(f"Accuracy      : {overall_acc:.4%}")

    # Table header
    print("\n===== By view =====")
    header = f"{'View':>20} | {'Total':>7} | {'Correct':>8} | {'Uncertain':>9} | {'Accuracy':>9}"
    print(header)
    print("-" * len(header))

    # Sort alphabetically by the *human* view name
    for view in sorted(total_by_view):
        tot   = total_by_view[view]
        corr  = correct_by_view[view]
        uncrt = uncertain_by_view[view]
        acc   = corr / (tot-uncrt) if tot else 0.0
        print(f"{view:>20} | {tot:7d} | {corr:8d} | {uncrt:9d} | {acc:9.4%}")


###############################################################################
# 3. Entrypoint
###############################################################################

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluation.py <results.jsonl> <combined_views.json>", file=sys.stderr)
        sys.exit(1)

    analyze(Path(sys.argv[1]), Path(sys.argv[2]))
