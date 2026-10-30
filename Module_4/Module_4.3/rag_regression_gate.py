#!/usr/bin/env python3
"""Compare reviewed binary support labels for two versions of the same eval set."""
import argparse
import json
import math
from pathlib import Path
import sys


def load_run(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("run must be an object")
    for key in ("dataset_version", "rubric_version"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"missing {key}")
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("results must be a nonempty list")
    labels = {}
    for row in results:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError("each result needs a nonempty string id")
        if row["id"] in labels:
            raise ValueError(f"duplicate case ID: {row['id']}")
        if type(row.get("support_pass")) is not bool:
            raise ValueError(f"missing or invalid support_pass for {row['id']}")
        labels[row["id"]] = row["support_pass"]
    return data, labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--max-drop", type=float, default=0.05)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_drop) or not 0 <= args.max_drop <= 1:
        parser.error("--max-drop must be between 0 and 1")
    try:
        baseline, old = load_run(args.baseline)
        candidate, new = load_run(args.candidate)
        for field in ("dataset_version", "rubric_version"):
            if baseline[field] != candidate[field]:
                raise ValueError(f"{field} mismatch")
        if old.keys() != new.keys():
            raise ValueError(f"case IDs differ; missing={sorted(old.keys()-new.keys())}, extra={sorted(new.keys()-old.keys())}")
        old_rate = sum(old.values()) / len(old)
        new_rate = sum(new.values()) / len(new)
        worse = sorted(k for k in old if old[k] and not new[k])
        better = sorted(k for k in old if not old[k] and new[k])
        print(f"Cases: {len(old)} | baseline: {old_rate:.3f} | candidate: {new_rate:.3f} | allowed drop: {args.max_drop:.3f}")
        print(f"Regressed IDs: {worse}; improved IDs: {better}")
        if old_rate - new_rate > args.max_drop + 1e-12:
            print("FAIL: support pass rate dropped beyond the allowed amount")
            return 1
        print("PASS: support pass rate is within the allowed drop")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: invalid or incompatible runs: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
