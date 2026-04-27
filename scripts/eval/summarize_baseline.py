#!/usr/bin/env python3
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    parse_ok = sum(1 for row in rows if row["result"]["parse_ok"])
    decisions = {}
    for row in rows:
        decision = row["result"]["decision"]
        decisions[decision] = decisions.get(decision, 0) + 1

    print(json.dumps({
        "num_samples": len(rows),
        "parse_ok": parse_ok,
        "parse_rate": parse_ok / len(rows) if rows else 0.0,
        "decision_counts": decisions,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
