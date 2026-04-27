#!/usr/bin/env python3
import argparse
import json
import os
import random
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
    tools = []
    for tool in payload:
        normalized = _normalize_tool(tool)
        if normalized is not None:
            tools.append(normalized)
    return tools


def _tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    names = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _convert_xlam_row(row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    answers = _safe_json_loads(row.get("answers"), [])
    if not isinstance(answers, list) or not answers:
        return None
    if len(answers) != 1:
        return None

    first = answers[0]
    if not isinstance(first, dict):
        return None
    tool_name = first.get("name") or first.get("tool_name")
    arguments = first.get("arguments", {}) or {}
    if tool_name is None or not isinstance(arguments, dict):
        return None

    target = {
        "decision": "tool_call",
        "tool_name": tool_name,
        "arguments": arguments,
    }
    return {
        "id": str(row.get("id", f"xlam_fc_60k_{idx}")),
        "dataset": "xlam_fc_60k",
        "task": "function_calling",
        "valid_decisions": ["tool_call", "answer_directly"],
        "tools": _normalize_tools(row.get("tools")),
        "messages": [{"role": "user", "content": row["query"]}],
        "target": target,
        "target_text": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        "metadata": {
            "source_format": "xlam_fc_60k_raw",
            "num_tool_calls": len(answers),
        },
    }


def _normalize_tool_call(answer: Any) -> Optional[Dict[str, Any]]:
    payload = _safe_json_loads(answer, answer)
    if not isinstance(payload, dict):
        return None
    tool_name = payload.get("name") or payload.get("tool_name")
    arguments = payload.get("arguments", {}) or {}
    if tool_name is None or not isinstance(arguments, dict):
        return None
    return {
        "name": tool_name,
        "arguments": arguments,
    }


def _convert_xlam_multi_row(
    row: Dict[str, Any],
    idx: int,
    min_calls: int,
    max_calls: Optional[int],
) -> Optional[Dict[str, Any]]:
    answers = _safe_json_loads(row.get("answers"), [])
    if not isinstance(answers, list) or len(answers) < min_calls:
        return None
    if max_calls is not None and len(answers) > max_calls:
        return None

    tool_calls = []
    for answer in answers:
        normalized = _normalize_tool_call(answer)
        if normalized is None:
            return None
        tool_calls.append(normalized)

    target_text = json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": str(row.get("id", f"xlam_fc_60k_multi_{idx}")),
        "dataset": "xlam_fc_60k_multi",
        "task": "function_calling_multi",
        "valid_decisions": ["tool_call"],
        "tools": _normalize_tools(row.get("tools")),
        "messages": [{"role": "user", "content": row["query"]}],
        "target": {"tool_calls": tool_calls},
        "target_text": target_text,
        "metadata": {
            "source_format": "xlam_fc_60k_raw",
            "num_tool_calls": len(tool_calls),
        },
    }


def _load_xlam_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    raw_rows = _load_jsonl(path)
    for idx, row in enumerate(raw_rows):
        converted = _convert_xlam_row(row, idx=idx)
        if converted is not None:
            rows.append(converted)
    return rows


def _load_xlam_multi_rows(path: str, min_calls: int, max_calls: Optional[int]) -> List[Dict[str, Any]]:
    rows = []
    raw_rows = _load_jsonl(path)
    for idx, row in enumerate(raw_rows):
        converted = _convert_xlam_multi_row(row, idx=idx, min_calls=min_calls, max_calls=max_calls)
        if converted is not None:
            rows.append(converted)
    return rows


def _build_xlam_irrelevance_rows(single_rows: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    if len(single_rows) < 2:
        return []

    donor_pool = []
    for row in single_rows:
        tools = row.get("tools", [])
        tool_names = set(_tool_names(tools))
        if tools and tool_names:
            donor_pool.append((tools, tool_names, row.get("id")))

    irrelevance_rows = []
    for idx, row in enumerate(single_rows):
        target = row.get("target", {})
        original_tool_name = target.get("tool_name") if isinstance(target, dict) else None
        if not isinstance(original_tool_name, str) or not original_tool_name:
            continue

        donor = None
        for _ in range(16):
            donor_tools, donor_names, donor_id = donor_pool[rng.randrange(len(donor_pool))]
            if donor_id == row.get("id"):
                continue
            if original_tool_name in donor_names:
                continue
            donor = donor_tools
            break
        if donor is None:
            continue

        target_payload = {
            "decision": "answer_directly",
            "tool_name": None,
            "arguments": {},
        }
        irrelevance_rows.append(
            {
                "id": f"{row.get('id', f'xlam_fc_60k_{idx}')}_irrelevance",
                "dataset": "xlam_fc_60k_irrelevance",
                "task": "function_calling",
                "valid_decisions": ["tool_call", "answer_directly"],
                "tools": donor,
                "messages": row.get("messages", []),
                "target": target_payload,
                "target_text": json.dumps(target_payload, ensure_ascii=False, separators=(",", ":")),
                "metadata": {
                    "source_format": "xlam_fc_60k_irrelevance_synthetic",
                    "original_tool_name": original_tool_name,
                    "irrelevance_type": "tool_mismatch",
                },
            }
        )
    return irrelevance_rows


def _count_label(row: Dict[str, Any]) -> Tuple[str, str]:
    dataset = row.get("dataset", "unknown")
    target = row.get("target", {})
    if isinstance(target, dict) and "decision" in target:
        return dataset, str(target["decision"])
    if isinstance(target, dict) and "tool_calls" in target:
        return dataset, "tool_call_multi"
    return dataset, "unknown"


def _decision_of(row: Dict[str, Any]) -> Optional[str]:
    target = row.get("target", {})
    if isinstance(target, dict):
        decision = target.get("decision")
        if isinstance(decision, str):
            return decision
    return None


def _sample_when2call_balanced(
    rows: List[Dict[str, Any]],
    rng: random.Random,
    max_total: int,
    ask_user_target: Optional[int],
    refuse_target: Optional[int],
    tool_call_target: Optional[int],
    oversample: bool,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "ask_user": [],
        "refuse": [],
        "tool_call": [],
    }
    remainder: List[Dict[str, Any]] = []
    for row in rows:
        decision = _decision_of(row)
        if decision in buckets:
            buckets[decision].append(row)
        else:
            remainder.append(row)

    for bucket in buckets.values():
        rng.shuffle(bucket)
    rng.shuffle(remainder)

    explicit_targets = {
        "ask_user": ask_user_target,
        "refuse": refuse_target,
        "tool_call": tool_call_target,
    }
    selected: List[Dict[str, Any]] = []
    used = 0
    for decision, target in explicit_targets.items():
        if target is None:
            continue
        bucket = buckets[decision]
        if not bucket:
            continue
        if oversample and target > len(bucket):
            selected.extend(bucket)
            selected.extend(rng.choices(bucket, k=target - len(bucket)))
            buckets[decision] = []
            used += target
            continue
        take = min(target, len(bucket))
        selected.extend(bucket[:take])
        buckets[decision] = bucket[take:]
        used += take

    remaining_budget = max(0, max_total - used)
    if remaining_budget == 0:
        rng.shuffle(selected)
        return selected[:max_total]

    unset_decisions = [decision for decision, target in explicit_targets.items() if target is None]
    if unset_decisions:
        per_bucket = remaining_budget // len(unset_decisions)
        extra = remaining_budget % len(unset_decisions)
        for idx, decision in enumerate(unset_decisions):
            budget = per_bucket + (1 if idx < extra else 0)
            take = min(budget, len(buckets[decision]))
            selected.extend(buckets[decision][:take])
            buckets[decision] = buckets[decision][take:]

    if len(selected) < max_total:
        leftovers: List[Dict[str, Any]] = []
        for bucket in buckets.values():
            leftovers.extend(bucket)
        leftovers.extend(remainder)
        rng.shuffle(leftovers)
        need = max_total - len(selected)
        selected.extend(leftovers[:need])

    rng.shuffle(selected)
    return selected[:max_total]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--when2call-path", default="data/processed/when2call/when2call_train_sft_fixed.jsonl")
    parser.add_argument("--xlam-path", default="data/raw/xlam_fc_60k/xlam_fc_60k.jsonl")
    parser.add_argument("--output-path", default="data/processed/mixed_sft/when2call_xlam_bfcl_sft_v7.jsonl")
    parser.add_argument("--max-when2call", type=int, default=4500)
    parser.add_argument("--when2call-ask-user", type=int, default=1500)
    parser.add_argument("--when2call-refuse", type=int, default=1500)
    parser.add_argument("--when2call-tool-call", type=int, default=1500)
    parser.add_argument("--oversample-when2call", action="store_true")
    parser.add_argument("--max-xlam-single", type=int, default=7000)
    parser.add_argument("--max-xlam-multi", type=int, default=4500)
    parser.add_argument("--max-xlam-irrelevance", type=int, default=3000)
    parser.add_argument("--xlam-multi-min-calls", type=int, default=2)
    parser.add_argument("--xlam-multi-max-calls", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    when2call_rows = _load_jsonl(args.when2call_path)
    xlam_rows = _load_xlam_rows(args.xlam_path)
    xlam_multi_rows = _load_xlam_multi_rows(
        args.xlam_path,
        min_calls=args.xlam_multi_min_calls,
        max_calls=args.xlam_multi_max_calls,
    )
    xlam_irrelevance_rows = _build_xlam_irrelevance_rows(xlam_rows, rng)

    rng.shuffle(xlam_rows)
    rng.shuffle(xlam_multi_rows)
    rng.shuffle(xlam_irrelevance_rows)

    selected_when2call = _sample_when2call_balanced(
        when2call_rows,
        rng=rng,
        max_total=args.max_when2call,
        ask_user_target=args.when2call_ask_user,
        refuse_target=args.when2call_refuse,
        tool_call_target=args.when2call_tool_call,
        oversample=args.oversample_when2call,
    )
    selected_xlam_single = xlam_rows[: args.max_xlam_single]
    selected_xlam_multi = xlam_multi_rows[: args.max_xlam_multi]
    selected_xlam_irrelevance = xlam_irrelevance_rows[: args.max_xlam_irrelevance]
    mixed = selected_when2call + selected_xlam_single + selected_xlam_multi + selected_xlam_irrelevance
    rng.shuffle(mixed)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    counts = Counter()
    with open(args.output_path, "w", encoding="utf-8") as out:
        for row in mixed:
            counts[_count_label(row)] += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_when2call": len(selected_when2call),
        "num_xlam_single": len(selected_xlam_single),
        "num_xlam_multi": len(selected_xlam_multi),
        "num_xlam_irrelevance": len(selected_xlam_irrelevance),
        "num_total": len(mixed),
        "when2call_targets": {
            "ask_user": args.when2call_ask_user,
            "refuse": args.when2call_refuse,
            "tool_call": args.when2call_tool_call,
        },
        "oversample_when2call": args.oversample_when2call,
        "xlam_multi_constraints": {
            "min_calls": args.xlam_multi_min_calls,
            "max_calls": args.xlam_multi_max_calls,
        },
        "counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
