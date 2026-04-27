#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


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


def load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_decision(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    mapping = {
        "direct": "answer_directly",
        "answer_directly": "answer_directly",
        "tool_call": "tool_call",
        "request_for_info": "ask_user",
        "ask_user": "ask_user",
        "cannot_answer": "refuse",
        "refuse": "refuse",
    }
    return mapping.get(label, label)


def score_when2call(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    correct = 0
    parse_ok = 0
    confusion = Counter()
    labels = ["tool_call", "ask_user", "refuse"]
    per_label_tp = Counter()
    per_label_fp = Counter()
    per_label_fn = Counter()
    non_tool_total = 0
    hallucinated_tool_calls = 0
    answer_directly_predictions = 0
    for row in rows:
        expected = normalize_decision(row.get("ground_truth", {}).get("decision"))
        predicted = normalize_decision(row["result"]["decision"])
        if expected is None:
            continue
        total += 1
        parse_ok += int(bool(row["result"]["parse_ok"]))
        confusion[f"{expected} -> {predicted}"] += 1
        correct += int(expected == predicted)
        if expected != "tool_call":
            non_tool_total += 1
            hallucinated_tool_calls += int(predicted == "tool_call")
        answer_directly_predictions += int(predicted == "answer_directly")

        for label in labels:
            if predicted == label and expected == label:
                per_label_tp[label] += 1
            elif predicted == label and expected != label:
                per_label_fp[label] += 1
            elif predicted != label and expected == label:
                per_label_fn[label] += 1

    per_label_f1 = {}
    for label in labels:
        tp = per_label_tp[label]
        fp = per_label_fp[label]
        fn = per_label_fn[label]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label_f1[label] = f1

    return {
        "dataset": "when2call",
        "num_scored": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(per_label_f1.values()) / len(labels) if labels else 0.0,
        "parse_rate": parse_ok / total if total else 0.0,
        "tool_hallucination_rate": hallucinated_tool_calls / non_tool_total if non_tool_total else 0.0,
        "answer_directly_predictions": answer_directly_predictions,
        "per_label_f1": per_label_f1,
        "confusion": dict(sorted(confusion.items())),
    }


def _norm_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _norm_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _norm_obj(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_norm_obj(v) for v in value]
    return _norm_scalar(value)


def _normalize_tool_name(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str):
        return name
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return normalized.strip("_")


def _subset_of(row: Dict[str, Any]) -> Optional[str]:
    metadata = row.get("metadata", {}) or {}
    if metadata.get("subset"):
        return metadata.get("subset")
    raw_row = metadata.get("raw_row", {}) or {}
    if raw_row.get("subset") or raw_row.get("test_category"):
        return raw_row.get("subset") or raw_row.get("test_category")
    nested_metadata = raw_row.get("metadata", {}) or {}
    return nested_metadata.get("subset") or nested_metadata.get("test_category")


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _legacy_bfcl_ground_truth_from_raw_row(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_row = ((row.get("metadata", {}) or {}).get("raw_row", {}) or {})
    raw_ground_truth = raw_row.get("ground_truth")
    payload = _safe_json_loads(raw_ground_truth)
    subset = _subset_of(row)
    if isinstance(payload, dict):
        if payload:
            return payload
        if subset in {"irrelevance", "chatable", "live_irrelevance"}:
            return {"decision": "answer_directly", "tool_calls": []}
        return {}
    if isinstance(payload, list):
        tool_calls = []
        for item in payload:
            if not isinstance(item, dict) or len(item) != 1:
                continue
            tool_name, arguments = next(iter(item.items()))
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append({"name": tool_name, "arguments": arguments})
        if tool_calls:
            return {"decision": "tool_call", "tool_calls": tool_calls}
    return {}


def _extract_expected_tool_calls(row: Dict[str, Any]) -> Dict[str, Any]:
    ground_truth = row.get("ground_truth", {}) or {}
    tool_calls = ground_truth.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        normalized = []
        for call in tool_calls:
            if isinstance(call, dict):
                normalized.append(
                    {
                        "name": call.get("name") or call.get("tool_name"),
                        "arguments": call.get("arguments", {}) or {},
                    }
                )
        if normalized:
            return {
                "decision": "tool_call",
                "tool_calls": normalized,
            }
    decision = normalize_decision(ground_truth.get("decision"))
    tool_name = ground_truth.get("tool_name") or ground_truth.get("target_tool")
    arguments = ground_truth.get("arguments", {}) or {}
    if tool_name:
        return {
            "decision": "tool_call",
            "tool_calls": [{"name": tool_name, "arguments": arguments}],
        }
    legacy = _legacy_bfcl_ground_truth_from_raw_row(row)
    if legacy:
        return legacy
    return {
        "decision": decision,
        "tool_calls": [],
    }


def _extract_predicted_tool_calls(row: Dict[str, Any]) -> Dict[str, Any]:
    result = row.get("result", {}) or {}
    tool_calls = result.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        normalized = []
        for call in tool_calls:
            if isinstance(call, dict):
                normalized.append(
                    {
                        "name": call.get("name") or call.get("tool_name"),
                        "arguments": call.get("arguments", {}) or {},
                    }
                )
        if normalized:
            return {
                "decision": "tool_call",
                "tool_calls": normalized,
            }
    tool_name = result.get("tool_name")
    if tool_name:
        return {
            "decision": "tool_call",
            "tool_calls": [{"name": tool_name, "arguments": result.get("arguments", {}) or {}}],
        }
    return {
        "decision": normalize_decision(result.get("decision")),
        "tool_calls": [],
    }


def _match_expected_value(pred_value: Any, expected_value: Any) -> bool:
    if _norm_obj(pred_value) == _norm_obj(expected_value):
        return True
    if isinstance(expected_value, dict):
        if not isinstance(pred_value, dict):
            return False
        for key, nested_expected in expected_value.items():
            if key not in pred_value:
                return False
            if not _match_expected_value(pred_value[key], nested_expected):
                return False
        return True
    if isinstance(expected_value, list):
        if not expected_value:
            return pred_value == expected_value
        if any(isinstance(item, (dict, list)) for item in expected_value):
            return any(_match_expected_value(pred_value, item) for item in expected_value)
        normalized_expected = [_norm_obj(item) for item in expected_value]
        return _norm_obj(pred_value) in normalized_expected
    return _norm_obj(pred_value) == _norm_obj(expected_value)


def _match_arguments(predicted_args: Dict[str, Any], expected_args: Dict[str, Any]) -> bool:
    if not isinstance(predicted_args, dict) or not isinstance(expected_args, dict):
        return False
    for key, expected_value in expected_args.items():
        if key not in predicted_args:
            return False
        if not _match_expected_value(predicted_args[key], expected_value):
            return False
    return True


def _tool_name_equal(left: Optional[str], right: Optional[str]) -> bool:
    return _normalize_tool_name(left) == _normalize_tool_name(right)


def _match_tool_calls(
    predicted_calls: List[Dict[str, Any]], expected_calls: List[Dict[str, Any]]
) -> Tuple[bool, bool]:
    if len(predicted_calls) != len(expected_calls):
        return False, False
    used = set()
    matched_names = True
    matched_args = True
    for expected_call in expected_calls:
        matched_idx = None
        for idx, predicted_call in enumerate(predicted_calls):
            if idx in used:
                continue
            if not _tool_name_equal(predicted_call.get("name"), expected_call.get("name")):
                continue
            matched_idx = idx
            if not _match_arguments(
                predicted_call.get("arguments", {}) or {},
                expected_call.get("arguments", {}) or {},
            ):
                matched_args = False
            break
        if matched_idx is None:
            matched_names = False
            matched_args = False
            break
        used.add(matched_idx)
    return matched_names, matched_args if matched_names else False


def score_bfcl(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    parse_ok = 0
    decision_match = 0
    tool_name_match = 0
    args_match = 0
    exact_match = 0
    skipped = 0
    skip_reasons = Counter()
    scored_subsets = Counter()

    for row in rows:
        subset = _subset_of(row)
        if subset not in STRICT_BFCL_SUBSETS:
            skipped += 1
            skip_reasons[f"unsupported_subset:{subset or 'unknown'}"] += 1
            continue

        expected = _extract_expected_tool_calls(row)
        exp_decision = normalize_decision(expected.get("decision"))
        if exp_decision is None:
            skipped += 1
            skip_reasons["missing_ground_truth"] += 1
            continue

        predicted = row.get("result", {}) or {}
        predicted_struct = _extract_predicted_tool_calls(row)
        pred_decision = normalize_decision(predicted_struct.get("decision"))

        total += 1
        scored_subsets[subset] += 1
        parse_ok += int(bool(predicted.get("parse_ok")))
        decision_match += int(pred_decision == exp_decision)

        if exp_decision != "tool_call":
            tool_name_ok = int(pred_decision == exp_decision)
            args_ok = int(pred_decision == exp_decision)
        else:
            tool_name_ok, args_ok = _match_tool_calls(
                predicted_struct.get("tool_calls", []),
                expected.get("tool_calls", []),
            )
            tool_name_ok = int(tool_name_ok)
            args_ok = int(args_ok)

        tool_name_match += tool_name_ok
        args_match += args_ok
        exact_match += int(pred_decision == exp_decision and tool_name_ok and args_ok)

    return {
        "dataset": "bfcl",
        "num_scored": total,
        "num_skipped": skipped,
        "parse_rate": parse_ok / total if total else 0.0,
        "decision_accuracy": decision_match / total if total else 0.0,
        "tool_name_accuracy": tool_name_match / total if total else 0.0,
        "arguments_accuracy": args_match / total if total else 0.0,
        "exact_match": exact_match / total if total else 0.0,
        "scored_subsets": dict(sorted(scored_subsets.items())),
        "skip_reasons": dict(sorted(skip_reasons.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--dataset", choices=["when2call", "bfcl"], required=True)
    args = parser.parse_args()

    rows = load_rows(args.predictions_path)
    if args.dataset == "when2call":
        result = score_when2call(rows)
    else:
        result = score_bfcl(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
