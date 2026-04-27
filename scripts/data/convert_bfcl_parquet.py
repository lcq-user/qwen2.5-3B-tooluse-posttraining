#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd


STRICT_BFCL_SUBSETS = {
    "irrelevance",
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "java",
    "javascript",
}


def _safe_json_loads(text: Any, default: Any) -> Any:
    if text is None:
        return default
    if isinstance(text, (dict, list)):
        return text
    if isinstance(text, str):
        text = text.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    return default


def _normalize_tools(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = _safe_json_loads(row.get("tools"), [])
    if isinstance(tools, list) and tools:
        normalized = []
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool and "name" not in tool:
                normalized.append(tool["function"])
            else:
                normalized.append(tool)
        return normalized
    functions = _safe_json_loads(row.get("functions"), [])
    return functions if isinstance(functions, list) else []


def _normalize_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    turns = _safe_json_loads(row.get("turns"), [])
    if isinstance(turns, list) and turns:
        first_turn = turns[0]
        if isinstance(first_turn, list):
            return first_turn
        if isinstance(first_turn, dict):
            return turns
    raise ValueError(f"Could not normalize turns for row id={row.get('id')}")


def _normalize_ground_truth(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = _safe_json_loads(row.get("ground_truth"), {})
    if isinstance(payload, dict):
        if payload:
            return payload
        subset = row.get("subset") or row.get("test_category")
        if subset in {"irrelevance", "chatable", "live_irrelevance"}:
            return {"decision": "answer_directly", "tool_name": None, "arguments": {}}
        return {}
    if isinstance(payload, list):
        tool_calls = []
        for item in payload:
            if not isinstance(item, dict) or len(item) != 1:
                continue
            tool_name, arguments = next(iter(item.items()))
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(
                {
                    "name": tool_name,
                    "arguments": arguments,
                }
            )
        if tool_calls:
            return {
                "decision": "tool_call",
                "tool_calls": tool_calls,
            }
        return {}
    return {}


def _normalize_involved_classes(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="data/raw/bfcl/train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/bfcl/bfcl_v3_all.json",
    )
    parser.add_argument(
        "--strict-output-path",
        default="data/processed/bfcl/bfcl_v3_strict.json",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input_path)
    rows = df.to_dict(orient="records")
    samples = []
    for row in rows:
        samples.append(
            {
                "id": row["id"],
                "messages": _normalize_messages(row),
                "tools": _normalize_tools(row),
                "ground_truth": _normalize_ground_truth(row),
                "metadata": {
                    "multi_turn": row.get("multi_turn"),
                    "language": row.get("language"),
                    "test_category": row.get("test_category"),
                    "subset": row.get("subset"),
                    "strict_eval_supported": (row.get("subset") in STRICT_BFCL_SUBSETS),
                    "missed_functions": _safe_json_loads(row.get("missed_functions"), {}),
                    "initial_config": _safe_json_loads(row.get("initial_config"), {}),
                    "involved_classes": _normalize_involved_classes(row.get("involved_classes")),
                },
            }
        )

    strict_samples = [
        sample
        for sample in samples
        if sample["metadata"].get("strict_eval_supported")
    ]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    os.makedirs(os.path.dirname(args.strict_output_path), exist_ok=True)
    with open(args.strict_output_path, "w", encoding="utf-8") as f:
        json.dump(strict_samples, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(samples)} rows to {args.output_path}")
    print(f"Saved {len(strict_samples)} strict rows to {args.strict_output_path}")


if __name__ == "__main__":
    main()
