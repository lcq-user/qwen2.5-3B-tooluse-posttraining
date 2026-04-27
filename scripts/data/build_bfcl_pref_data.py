#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_json_loads(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _normalize_tool(tool: Any) -> Optional[Dict[str, Any]]:
    payload = _safe_json_loads(tool, tool)
    if isinstance(payload, dict) and "function" in payload and "name" not in payload:
        function = payload.get("function")
        return function if isinstance(function, dict) else None
    return payload if isinstance(payload, dict) else None


def _normalize_tools(value: Any) -> List[Dict[str, Any]]:
    payload = _safe_json_loads(value, [])
    if not isinstance(payload, list):
        return []
    normalized = []
    for tool in payload:
        item = _normalize_tool(tool)
        if item is not None:
            normalized.append(item)
    return normalized


def _tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    names = []
    for tool in tools:
        name = tool.get("name") if isinstance(tool, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    return names


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


def _classify_when2call_text(text: str) -> str:
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


def _normalize_when2call_response_text(text: str) -> str:
    label = _classify_when2call_text(text)
    if label == "tool_call":
        json_block = _extract_json_block(text)
        if json_block is None:
            raise ValueError(f"Could not extract tool call JSON from response: {text[:200]}")
        payload = _safe_json_loads(json_block)
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
            raise ValueError(f"Unexpected tool-call payload: {json_block[:200]}")
        first = payload[0]
        target = {
            "decision": "tool_call",
            "tool_name": first.get("name") or first.get("tool_name"),
            "arguments": first.get("arguments", {}) or {},
        }
    else:
        target = {
            "decision": label,
            "tool_name": None,
            "arguments": {},
        }
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def _load_when2call_pref_generic(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            chosen_text = _normalize_when2call_response_text(raw["chosen_response"]["content"])
            rejected_text = _normalize_when2call_response_text(raw["rejected_response"]["content"])
            if chosen_text == rejected_text:
                continue
            rows.append(
                {
                    "id": f"when2call_pref_{idx}",
                    "dataset": "when2call_pref",
                    "task": "when2call_decision",
                    "valid_decisions": ["tool_call", "ask_user", "refuse"],
                    "messages": raw["messages"],
                    "tools": [_normalize_tool(tool) for tool in raw.get("tools", []) if _normalize_tool(tool) is not None],
                    "chosen_text": chosen_text,
                    "rejected_text": rejected_text,
                    "metadata": {
                        "source_format": "when2call_pref_raw",
                    },
                }
            )
    return rows


def _convert_xlam_single(row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    answers = _safe_json_loads(row.get("answers"), [])
    if not isinstance(answers, list) or len(answers) != 1:
        return None
    answer = answers[0]
    if not isinstance(answer, dict):
        return None
    tool_name = answer.get("name") or answer.get("tool_name")
    arguments = answer.get("arguments", {}) or {}
    if not isinstance(tool_name, str) or not isinstance(arguments, dict):
        return None
    target = {
        "decision": "tool_call",
        "tool_name": tool_name,
        "arguments": arguments,
    }
    return {
        "id": str(row.get("id", f"xlam_single_{idx}")),
        "dataset": "xlam_fc_60k",
        "task": "function_calling",
        "valid_decisions": ["tool_call", "answer_directly"],
        "messages": [{"role": "user", "content": row["query"]}],
        "tools": _normalize_tools(row.get("tools")),
        "target": target,
        "target_text": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        "metadata": {
            "num_tool_calls": 1,
            "source_format": "xlam_fc_60k_raw",
        },
    }


def _convert_xlam_multi(row: Dict[str, Any], idx: int, min_calls: int, max_calls: int) -> Optional[Dict[str, Any]]:
    answers = _safe_json_loads(row.get("answers"), [])
    if not isinstance(answers, list):
        return None
    if len(answers) < min_calls or len(answers) > max_calls:
        return None
    tool_calls = []
    for answer in answers:
        if not isinstance(answer, dict):
            return None
        tool_name = answer.get("name") or answer.get("tool_name")
        arguments = answer.get("arguments", {}) or {}
        if not isinstance(tool_name, str) or not isinstance(arguments, dict):
            return None
        tool_calls.append({"name": tool_name, "arguments": arguments})
    return {
        "id": str(row.get("id", f"xlam_multi_{idx}")),
        "dataset": "xlam_fc_60k_multi",
        "task": "function_calling_multi",
        "valid_decisions": ["tool_call"],
        "messages": [{"role": "user", "content": row["query"]}],
        "tools": _normalize_tools(row.get("tools")),
        "target": {"tool_calls": tool_calls},
        "target_text": json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":")),
        "metadata": {
            "num_tool_calls": len(tool_calls),
            "source_format": "xlam_fc_60k_raw",
        },
    }


def _build_irrelevance_rows(single_rows: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    donor_pool = []
    for row in single_rows:
        tools = row.get("tools", [])
        names = set(_tool_names(tools))
        if tools and names:
            donor_pool.append((row["id"], tools, names))
    rows = []
    for row in single_rows:
        target = row["target"]
        original_tool_name = target.get("tool_name")
        if not isinstance(original_tool_name, str):
            continue
        donor_tools = None
        for _ in range(24):
            donor_id, candidate_tools, donor_names = donor_pool[rng.randrange(len(donor_pool))]
            if donor_id == row["id"]:
                continue
            if original_tool_name in donor_names:
                continue
            donor_tools = candidate_tools
            break
        if donor_tools is None:
            continue
        answer_directly = {
            "decision": "answer_directly",
            "tool_name": None,
            "arguments": {},
        }
        rows.append(
            {
                "id": f"{row['id']}_irrelevance",
                "dataset": "xlam_fc_60k_irrelevance",
                "task": "function_calling",
                "valid_decisions": ["tool_call", "answer_directly"],
                "messages": row["messages"],
                "tools": donor_tools,
                "target": answer_directly,
                "target_text": json.dumps(answer_directly, ensure_ascii=False, separators=(",", ":")),
                "metadata": {
                    "source_format": "xlam_fc_60k_irrelevance_synthetic",
                    "irrelevance_type": "tool_mismatch",
                },
            }
        )
    return rows


def _wrong_tool_rejected(row: Dict[str, Any], rng: random.Random) -> Optional[str]:
    target = row["target"]
    correct_tool = target.get("tool_name")
    if not isinstance(correct_tool, str):
        return None
    candidate_names = [name for name in _tool_names(row.get("tools", [])) if name != correct_tool]
    if not candidate_names:
        return None
    rejected = {
        "decision": "tool_call",
        "tool_name": candidate_names[rng.randrange(len(candidate_names))],
        "arguments": target.get("arguments", {}) or {},
    }
    return json.dumps(rejected, ensure_ascii=False, separators=(",", ":"))


def _drop_argument_rejected(row: Dict[str, Any], rng: random.Random) -> Optional[str]:
    target = row["target"]
    arguments = dict(target.get("arguments", {}) or {})
    if not arguments:
        return None
    keys = list(arguments.keys())
    drop_key = keys[rng.randrange(len(keys))]
    arguments.pop(drop_key, None)
    rejected = {
        "decision": "tool_call",
        "tool_name": target.get("tool_name"),
        "arguments": arguments,
    }
    return json.dumps(rejected, ensure_ascii=False, separators=(",", ":"))


def _answer_directly_rejected() -> str:
    return json.dumps(
        {
            "decision": "answer_directly",
            "tool_name": None,
            "arguments": {},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _build_single_pref_rows(rows: List[Dict[str, Any]], rng: random.Random, limit: int) -> List[Dict[str, Any]]:
    rng.shuffle(rows)
    output = []
    for row in rows:
        chosen_text = row["target_text"]
        rejected_text = (
            _wrong_tool_rejected(row, rng)
            or _drop_argument_rejected(row, rng)
            or _answer_directly_rejected()
        )
        output.append(
            {
                "id": f"{row['id']}_pref",
                "dataset": "bfcl_style_single_pref",
                "task": row["task"],
                "valid_decisions": row["valid_decisions"],
                "messages": row["messages"],
                "tools": row["tools"],
                "chosen_text": chosen_text,
                "rejected_text": rejected_text,
                "metadata": {
                    **row.get("metadata", {}),
                    "pref_type": "single_call",
                },
            }
        )
        if len(output) >= limit:
            break
    return output


def _build_multi_pref_rows(rows: List[Dict[str, Any]], rng: random.Random, limit: int) -> List[Dict[str, Any]]:
    rng.shuffle(rows)
    output = []
    for row in rows:
        tool_calls = row["target"]["tool_calls"]
        if len(tool_calls) < 2:
            continue
        if len(tool_calls) == 2:
            rejected_calls = tool_calls[:1]
        else:
            rejected_calls = list(tool_calls)
            rejected_calls[0], rejected_calls[1] = rejected_calls[1], rejected_calls[0]
            if rejected_calls == tool_calls:
                rejected_calls = tool_calls[:-1]
        rejected_text = json.dumps(rejected_calls, ensure_ascii=False, separators=(",", ":"))
        if rejected_text == row["target_text"]:
            continue
        output.append(
            {
                "id": f"{row['id']}_pref",
                "dataset": "bfcl_style_multi_pref",
                "task": row["task"],
                "valid_decisions": row["valid_decisions"],
                "messages": row["messages"],
                "tools": row["tools"],
                "chosen_text": row["target_text"],
                "rejected_text": rejected_text,
                "metadata": {
                    **row.get("metadata", {}),
                    "pref_type": "multi_call",
                },
            }
        )
        if len(output) >= limit:
            break
    return output


def _build_irrelevance_pref_rows(rows: List[Dict[str, Any]], rng: random.Random, limit: int) -> List[Dict[str, Any]]:
    rng.shuffle(rows)
    output = []
    for row in rows:
        tool_names = _tool_names(row.get("tools", []))
        if not tool_names:
            continue
        hallucinated = {
            "decision": "tool_call",
            "tool_name": tool_names[0],
            "arguments": {},
        }
        output.append(
            {
                "id": f"{row['id']}_pref",
                "dataset": "bfcl_style_irrelevance_pref",
                "task": row["task"],
                "valid_decisions": row["valid_decisions"],
                "messages": row["messages"],
                "tools": row["tools"],
                "chosen_text": row["target_text"],
                "rejected_text": json.dumps(hallucinated, ensure_ascii=False, separators=(",", ":")),
                "metadata": {
                    **row.get("metadata", {}),
                    "pref_type": "irrelevance",
                },
            }
        )
        if len(output) >= limit:
            break
    return output


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--when2call-pref-path", default="data/raw/when2call/train/when2call_train_pref.jsonl")
    parser.add_argument("--xlam-path", default="data/raw/xlam_fc_60k/xlam_fc_60k.jsonl")
    parser.add_argument("--output-bfcl-pref-path", default="data/processed/dpo/bfcl_style_pref_v1.jsonl")
    parser.add_argument("--output-mixed-pref-path", default="data/processed/dpo/when2call_bfcl_mixed_pref_v1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-single-pref", type=int, default=4000)
    parser.add_argument("--max-multi-pref", type=int, default=2500)
    parser.add_argument("--max-irrelevance-pref", type=int, default=2500)
    parser.add_argument("--max-when2call-pref", type=int, default=6000)
    parser.add_argument("--xlam-multi-min-calls", type=int, default=2)
    parser.add_argument("--xlam-multi-max-calls", type=int, default=4)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    raw_xlam = _load_jsonl(args.xlam_path)
    xlam_single = []
    xlam_multi = []
    for idx, row in enumerate(raw_xlam):
        single = _convert_xlam_single(row, idx)
        if single is not None:
            xlam_single.append(single)
        multi = _convert_xlam_multi(row, idx, args.xlam_multi_min_calls, args.xlam_multi_max_calls)
        if multi is not None:
            xlam_multi.append(multi)
    xlam_irrelevance = _build_irrelevance_rows(xlam_single, rng)

    bfcl_single_pref = _build_single_pref_rows(xlam_single, rng, args.max_single_pref)
    bfcl_multi_pref = _build_multi_pref_rows(xlam_multi, rng, args.max_multi_pref)
    bfcl_irrelevance_pref = _build_irrelevance_pref_rows(xlam_irrelevance, rng, args.max_irrelevance_pref)
    bfcl_pref_rows = bfcl_single_pref + bfcl_multi_pref + bfcl_irrelevance_pref
    rng.shuffle(bfcl_pref_rows)

    when2call_rows = _load_when2call_pref_generic(args.when2call_pref_path)
    rng.shuffle(when2call_rows)
    when2call_rows = when2call_rows[: args.max_when2call_pref]

    mixed_rows = list(when2call_rows) + list(bfcl_pref_rows)
    rng.shuffle(mixed_rows)

    _write_jsonl(args.output_bfcl_pref_path, bfcl_pref_rows)
    _write_jsonl(args.output_mixed_pref_path, mixed_rows)

    print(
        json.dumps(
            {
                "num_bfcl_style_pref": len(bfcl_pref_rows),
                "num_mixed_pref": len(mixed_rows),
                "bfcl_pref_breakdown": dict(Counter(row["dataset"] for row in bfcl_pref_rows)),
                "mixed_breakdown": dict(Counter(row["dataset"] for row in mixed_rows)),
                "output_bfcl_pref_path": args.output_bfcl_pref_path,
                "output_mixed_pref_path": args.output_mixed_pref_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
