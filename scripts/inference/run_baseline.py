#!/usr/bin/env python3
import argparse
import copy
import json
import os
import re
import urllib.request
from urllib.error import HTTPError, URLError
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant for function calling evaluation. "
    "Decide whether to call a tool, ask the user for clarification, refuse, "
    "or answer directly. "
    "Return either a tool call or a single JSON object with keys "
    "decision, tool_name, arguments."
)

WHEN2CALL_SYSTEM_PROMPT = (
    "You are an assistant for When2Call evaluation. "
    "Your final action must be exactly one of: tool_call, ask_user, refuse. "
    "Never answer the user's task directly. "
    "If the provided tools can fully satisfy the request, return a tool call. "
    "If required information is missing, ask a clarification question. "
    "If the request cannot be completed with the provided tools, refuse. "
    "Return either a tool call or a single JSON object with keys "
    "decision, tool_name, arguments."
)

VALID_DECISIONS = {"tool_call", "ask_user", "refuse", "answer_directly"}


SMOKE_SAMPLES = [
    {
        "id": "tool_call_weather",
        "messages": [
            {"role": "user", "content": "What's the weather like in London in celsius?"}
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ],
    },
    {
        "id": "ask_user_missing_arg",
        "messages": [
            {"role": "user", "content": "Book me a flight for tomorrow."}
        ],
        "tools": [
            {
                "name": "book_flight",
                "description": "Book a flight ticket",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_city": {"type": "string"},
                        "to_city": {"type": "string"},
                        "date": {"type": "string"},
                    },
                    "required": ["from_city", "to_city", "date"],
                },
            }
        ],
    },
]


@dataclass
class BaselineResult:
    decision: str
    tool_name: Optional[str]
    arguments: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    raw_text: str
    parse_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "tool_calls": self.tool_calls,
            "raw_text": self.raw_text,
            "parse_ok": self.parse_ok,
        }


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported dataset payload at {path}: expected list or jsonl")


def normalize_tool(tool: Any) -> Dict[str, Any]:
    if isinstance(tool, str):
        return json.loads(tool)
    if isinstance(tool, dict) and "function" in tool and "name" not in tool:
        function = copy.deepcopy(tool["function"])
        return function
    return copy.deepcopy(tool)


def load_when2call_samples(dataset_path: str) -> List[Dict[str, Any]]:
    rows = read_json_or_jsonl(dataset_path)
    answer_map = {
        "direct": "answer_directly",
        "tool_call": "tool_call",
        "request_for_info": "ask_user",
        "cannot_answer": "refuse",
    }
    samples = []
    for row in rows:
        tools = [normalize_tool(tool) for tool in row.get("tools", [])]
        samples.append(
            {
                "id": row["uuid"],
                "messages": [{"role": "user", "content": row["question"]}],
                "tools": tools,
                "ground_truth": {
                    "decision": answer_map[row["correct_answer"]],
                    "raw_label": row["correct_answer"],
                    "target_tool": row.get("target_tool"),
                },
                "metadata": {
                    "source": row.get("source"),
                    "source_id": row.get("source_id"),
                    "answers": row.get("answers"),
                },
            }
        )
    return samples


def _pick_first(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _coerce_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    question = _pick_first(row, ["question", "prompt", "user_query", "query", "input"])
    if question is None:
        raise ValueError(f"Could not locate question-like field in row: {list(row.keys())}")
    return [{"role": "user", "content": question}]


def _coerce_tools(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_payload = _pick_first(row, ["tools", "functions", "tool_list", "available_tools"])
    if tool_payload is None:
        return []
    return [normalize_tool(tool) for tool in tool_payload]


def _coerce_ground_truth(row: Dict[str, Any]) -> Dict[str, Any]:
    for key in ["correct_answer", "expected_decision"]:
        if key in row:
            return {"decision": row[key]}
    for key in ["ground_truth", "ground_truths", "expected", "answers", "tool_calls"]:
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict) and "name" in first:
                return {"decision": "tool_call", "tool_calls": value}
        if isinstance(value, dict):
            if "decision" in value:
                return value
            if "name" in value:
                return {"decision": "tool_call", "tool_calls": [value]}
        if isinstance(value, str):
            parsed = _safe_json_loads(value)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return {"decision": "tool_call", "tool_calls": parsed}
            if isinstance(parsed, dict) and "decision" in parsed:
                return parsed
    subset = _pick_first(row, ["subset", "test_category"])
    if subset in {"irrelevance", "chatable"}:
        return {"decision": "answer_directly", "tool_name": None, "arguments": {}}
    return {}


def load_bfcl_samples(dataset_path: str) -> List[Dict[str, Any]]:
    rows = read_json_or_jsonl(dataset_path)
    samples = []
    for idx, row in enumerate(rows):
        samples.append(
            {
                "id": str(_pick_first(row, ["id", "uuid", "question_id", "source_id"]) or idx),
                "messages": _coerce_messages(row),
                "tools": _coerce_tools(row),
                "ground_truth": _coerce_ground_truth(row),
                "metadata": {
                    "raw_row": row,
                },
            }
        )
    return samples


def load_samples(dataset_name: str, dataset_path: Optional[str]) -> List[Dict[str, Any]]:
    if dataset_name == "smoke":
        return [
            {
                "id": sample["id"],
                "messages": sample["messages"],
                "tools": sample["tools"],
                "ground_truth": {},
                "metadata": {},
            }
            for sample in SMOKE_SAMPLES
        ]
    if dataset_path is None:
        raise ValueError(f"--dataset-path is required for dataset={dataset_name}")
    if dataset_name == "when2call":
        return load_when2call_samples(dataset_path)
    if dataset_name == "bfcl":
        return load_bfcl_samples(dataset_path)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_model(model_path: str, dtype: str):
    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def is_xlam_model(model_path: str) -> bool:
    return "xlam" in os.path.basename(model_path).lower()


def build_messages(messages: List[Dict[str, str]], system_prompt: str) -> List[Dict[str, str]]:
    if messages and messages[0]["role"] == "system":
        return messages
    return [{"role": "system", "content": system_prompt}] + messages


def build_qwen_prompt(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    valid_decisions: List[str],
    dataset_name: str,
) -> str:
    instruction = {
        "task": "Decide the next action for the assistant.",
        "valid_decisions": valid_decisions,
        "output_schema": {
            "decision": " | ".join(valid_decisions),
            "tool_name": "string_or_null",
            "arguments": "object",
        },
        "tools": tools,
    }
    if dataset_name == "when2call":
        extra_rules = (
            "Important rules:\n"
            "- Your final action must be exactly one of tool_call, ask_user, refuse.\n"
            "- Do not answer the user's request directly.\n"
            "- If no provided tool can complete the request, choose refuse.\n"
            "- If some required argument is missing, choose ask_user.\n"
            "- If the request is fully supported by a provided tool, choose tool_call.\n"
            "- Output JSON only. No explanation before or after JSON."
        )
    else:
        extra_rules = "Output JSON only. Do not add extra prose before JSON."
    user_suffix = (
        "\n\nAvailable tools and output requirements:\n"
        f"{json.dumps(instruction, ensure_ascii=False)}\n"
        f"{extra_rules}\n"
        "If a tool is needed, output either a JSON array like "
        '[{"name":"tool_name","arguments":{...}}] '
        "or the JSON object schema above."
    )
    updated = list(messages)
    updated[-1] = {
        "role": updated[-1]["role"],
        "content": updated[-1]["content"] + user_suffix,
    }
    return updated


def generate_text(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    max_new_tokens: int,
    model_path: str,
    valid_decisions: List[str],
    dataset_name: str,
) -> str:
    xlam = is_xlam_model(model_path)
    templated_messages = (
        messages if xlam else build_qwen_prompt(messages, tools, valid_decisions, dataset_name)
    )
    inputs = tokenizer.apply_chat_template(
        templated_messages,
        tools=tools,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids_len = inputs["input_ids"].shape[-1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    generated_tokens = outputs[:, input_ids_len:]
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()


def build_vllm_payload(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    max_new_tokens: int,
    model_name: str,
) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0,
    }
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": tool,
            }
            for tool in tools
        ]
    return payload


def generate_text_vllm(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    max_new_tokens: int,
    model_path: str,
    api_base: str,
    served_model_name: Optional[str],
) -> str:
    model_name = served_model_name or os.path.basename(model_path.rstrip("/"))
    payload = build_vllm_payload(
        messages=messages,
        tools=tools,
        max_new_tokens=max_new_tokens,
        model_name=model_name,
    )
    req = urllib.request.Request(
        url=f"{api_base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer empty",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"vLLM request failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to vLLM server at {api_base}: {exc}") from exc

    choice = body["choices"][0]["message"]
    tool_calls = choice.get("tool_calls") or []
    if tool_calls:
        serialized = []
        for call in tool_calls:
            function = call.get("function", {})
            arguments = function.get("arguments", "{}")
            parsed_arguments = _safe_json_loads(arguments)
            serialized.append(
                {
                    "name": function.get("name"),
                    "arguments": parsed_arguments if isinstance(parsed_arguments, dict) else {},
                }
            )
        return json.dumps(serialized, ensure_ascii=False)
    return (choice.get("content") or "").strip()


def _safe_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_json_block(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    text = re.sub(r"</?tool_call>", "", text, flags=re.IGNORECASE).strip()
    direct = _safe_json_loads(text)
    if direct is not None:
        return text

    for pattern in [r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"]:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1)
            if _safe_json_loads(candidate) is not None:
                return candidate
    return None


def _normalize_tool_call(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(call, dict):
        return None
    tool_name = call.get("name") or call.get("tool_name")
    arguments = call.get("arguments", {})
    if tool_name is None and "function" in call:
        function = call.get("function")
        if isinstance(function, dict):
            tool_name = function.get("name") or function.get("function")
            arguments = function.get("arguments", arguments)
        elif isinstance(function, str):
            tool_name = function
    if tool_name is None:
        return None
    if not isinstance(arguments, dict):
        arguments = {}
    return {
        "name": tool_name,
        "arguments": arguments,
    }


def _tool_result(raw_text: str, tool_calls: List[Dict[str, Any]], parse_ok: bool) -> BaselineResult:
    first = tool_calls[0] if tool_calls else {}
    return BaselineResult(
        decision="tool_call",
        tool_name=first.get("name"),
        arguments=first.get("arguments", {}) or {},
        tool_calls=tool_calls,
        raw_text=raw_text,
        parse_ok=parse_ok,
    )


def parse_result(raw_text: str, dataset_name: str = "smoke") -> BaselineResult:
    json_block = _extract_json_block(raw_text)
    if json_block is not None:
        payload = _safe_json_loads(json_block)
        if isinstance(payload, list) and not payload:
            return BaselineResult(
                decision="answer_directly",
                tool_name=None,
                arguments={},
                tool_calls=[],
                raw_text=raw_text,
                parse_ok=True,
            )
        if isinstance(payload, list) and payload:
            tool_calls = []
            for item in payload:
                normalized = _normalize_tool_call(item)
                if normalized is not None:
                    tool_calls.append(normalized)
            if tool_calls:
                return _tool_result(raw_text=raw_text, tool_calls=tool_calls, parse_ok=True)
        if isinstance(payload, dict):
            payload_decision = payload.get("decision")
            if (
                isinstance(payload_decision, str)
                and payload_decision not in VALID_DECISIONS
                and ("parameters" in payload or "arguments" in payload)
            ):
                return _tool_result(
                    raw_text=raw_text,
                    tool_calls=[
                        {
                            "name": payload_decision,
                            "arguments": payload.get("arguments", payload.get("parameters", {})) or {},
                        }
                    ],
                    parse_ok=True,
                )
            normalized = _normalize_tool_call(payload)
            if normalized is not None:
                return _tool_result(raw_text=raw_text, tool_calls=[normalized], parse_ok=True)
            return BaselineResult(
                decision=payload_decision if payload_decision in VALID_DECISIONS else "answer_directly",
                tool_name=payload.get("tool_name"),
                arguments=payload.get("arguments", {}) or {},
                tool_calls=[],
                raw_text=raw_text,
                parse_ok=True,
            )

    lowered = raw_text.lower()
    ask_user_markers = [
        "clarif",
        "need more",
        "could you provide",
        "can you provide",
        "please provide",
        "what is your",
        "what's your",
        "where are you",
        "which city",
        "which date",
    ]
    if any(marker in lowered for marker in ask_user_markers) or raw_text.strip().endswith("?"):
        decision = "ask_user"
    elif "cannot" in lowered or "can't" in lowered or "refuse" in lowered:
        decision = "refuse"
    elif dataset_name == "when2call":
        refuse_markers = [
            "unable to",
            "not able to",
            "do not have access",
            "don't have access",
            "no provided tool",
            "none of the available tools",
            "unrelated to the available tools",
            "cannot complete",
            "can't complete",
            "cannot perform",
            "can't perform",
            "cannot help with",
            "can't help with",
            "tool is unrelated",
            "tools are unrelated",
            "not supported by the provided tools",
        ]
        if any(marker in lowered for marker in refuse_markers):
            decision = "refuse"
        else:
            decision = "refuse"
    else:
        decision = "answer_directly"
    return BaselineResult(
        decision=decision,
        tool_name=None,
        arguments={},
        tool_calls=[],
        raw_text=raw_text,
        parse_ok=bool(raw_text.strip()),
    )


def run_samples(
    model: Any,
    tokenizer: Any,
    model_path: str,
    output_path: str,
    max_new_tokens: int,
    system_prompt: str,
    samples: List[Dict[str, Any]],
    backend: str,
    api_base: str,
    served_model_name: Optional[str],
    valid_decisions: List[str],
    dataset_name: str,
) -> None:
    results = []
    total = len(samples)
    iterator = samples
    if tqdm is not None:
        iterator = tqdm(samples, desc="Evaluating", total=total)

    for idx, sample in enumerate(iterator, start=1):
        messages = build_messages(sample["messages"], system_prompt)
        if backend == "vllm":
            xlam = is_xlam_model(model_path)
            templated_messages = (
                messages
                if xlam
                else build_qwen_prompt(messages, sample["tools"], valid_decisions, dataset_name)
            )
            raw_text = generate_text_vllm(
                messages=templated_messages,
                tools=sample["tools"] if xlam else [],
                max_new_tokens=max_new_tokens,
                model_path=model_path,
                api_base=api_base,
                served_model_name=served_model_name,
            )
        else:
            raw_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                tools=sample["tools"],
                max_new_tokens=max_new_tokens,
                model_path=model_path,
                valid_decisions=valid_decisions,
                dataset_name=dataset_name,
            )
        parsed = parse_result(raw_text, dataset_name=dataset_name)
        results.append(
            {
                "id": sample["id"],
                "messages": messages,
                "tools": sample["tools"],
                "ground_truth": sample.get("ground_truth", {}),
                "metadata": sample.get("metadata", {}),
                "result": parsed.to_dict(),
            }
        )
        if tqdm is None and (idx == 1 or idx % 50 == 0 or idx == total):
            print(f"Evaluating {idx}/{total}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dataset", choices=["smoke", "when2call", "bfcl"], default="smoke")
    parser.add_argument("--dataset-path")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model-name")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    samples = load_samples(args.dataset, args.dataset_path)
    valid_decisions = (
        ["tool_call", "ask_user", "refuse"]
        if args.dataset == "when2call"
        else ["tool_call", "ask_user", "refuse", "answer_directly"]
    )
    system_prompt = (
        args.system_prompt
        if args.system_prompt != DEFAULT_SYSTEM_PROMPT
        else (WHEN2CALL_SYSTEM_PROMPT if args.dataset == "when2call" else DEFAULT_SYSTEM_PROMPT)
    )
    model, tokenizer = (None, None)
    if args.backend == "transformers":
        model, tokenizer = load_model(args.model_path, args.dtype)
    run_samples(
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
        output_path=args.output_path,
        max_new_tokens=args.max_new_tokens,
        system_prompt=system_prompt,
        samples=samples,
        backend=args.backend,
        api_base=args.api_base,
        served_model_name=args.served_model_name,
        valid_decisions=valid_decisions,
        dataset_name=args.dataset,
    )
    print(f"Saved baseline results to {args.output_path}")


if __name__ == "__main__":
    main()
