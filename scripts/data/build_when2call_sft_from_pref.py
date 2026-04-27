#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def _safe_json_loads(value: str) -> Optional[Any]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _normalize_tool(tool: Any) -> Dict[str, Any]:
    if isinstance(tool, str):
        parsed = _safe_json_loads(tool)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"Could not parse tool JSON: {tool[:120]}")
    if isinstance(tool, dict) and "function" in tool and "name" not in tool:
        function = tool["function"]
        if isinstance(function, dict):
            return function
    if isinstance(tool, dict):
        return tool
    raise ValueError(f"Unsupported tool payload type: {type(tool)}")


def _extract_json_block(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    cleaned = re.sub(r"</?TOOLCALL>", "", text, flags=re.IGNORECASE).strip()
    if _safe_json_loads(cleaned) is not None:
        return cleaned
    for pattern in [r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"]:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        candidate = match.group(1)
        if _safe_json_loads(candidate) is not None:
            return candidate
    return None


def _classify_text(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    if "<toolcall>" in lowered:
        return "tool_call"
    if stripped.endswith("?"):
        return "ask_user"
    ask_markers = [
        "could you please",
        "can you please",
        "please provide",
        "which ",
        "what is ",
        "what's ",
        "i need",
        "to assist you better",
        "to provide",
        "to proceed",
        "just to confirm",
    ]
    if any(marker in lowered for marker in ask_markers):
        return "ask_user"
    return "refuse"


def _normalize_target(chosen_text: str) -> Tuple[Dict[str, Any], str]:
    label = _classify_text(chosen_text)
    if label == "tool_call":
        json_block = _extract_json_block(chosen_text)
        if json_block is None:
            raise ValueError(f"Could not extract tool call JSON from chosen response: {chosen_text[:200]}")
        payload = _safe_json_loads(json_block)
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
            raise ValueError(f"Unexpected tool-call payload: {json_block[:200]}")
        first = payload[0]
        tool_name = first.get("name") or first.get("tool_name")
        arguments = first.get("arguments", {}) or {}
        if tool_name is None:
            raise ValueError(f"Missing tool name in payload: {json_block[:200]}")
        target = {
            "decision": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments,
        }
        return target, "tool_call"

    target = {
        "decision": label,
        "tool_name": None,
        "arguments": {},
    }
    return target, label


def load_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="data/raw/when2call/train/when2call_train_pref.jsonl",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/when2call/when2call_train_sft_fixed.jsonl",
    )
    args = parser.parse_args()

    rows = load_rows(args.input_path)
    label_counts = Counter()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as out:
        for idx, row in enumerate(rows):
            tools = [_normalize_tool(tool) for tool in row.get("tools", [])]
            messages = row.get("messages", [])
            if not messages or messages[-1].get("role") != "user":
                raise ValueError(f"Expected a single user message in row {idx}")

            chosen_text = row["chosen_response"]["content"]
            target, label = _normalize_target(chosen_text)
            label_counts[label] += 1

            record = {
                "id": f"when2call_pref_{idx}",
                "dataset": "when2call",
                "task": "when2call_decision",
                "valid_decisions": ["tool_call", "ask_user", "refuse"],
                "tools": tools,
                "messages": messages,
                "target": target,
                "target_text": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
                "metadata": {
                    "source_format": "when2call_train_pref",
                    "chosen_response": chosen_text,
                    "rejected_response": row.get("rejected_response", {}).get("content"),
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {sum(label_counts.values())} rows to {args.output_path}")
    print(json.dumps(dict(sorted(label_counts.items())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
