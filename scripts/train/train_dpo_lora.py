#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


DEFAULT_OUTPUT_SCHEMA = {
    "decision": "string",
    "tool_name": "string_or_null",
    "arguments": "object",
}

WHEN2CALL_SYSTEM_PROMPT = (
    "You are an assistant for When2Call evaluation. "
    "Your final action must be exactly one of: tool_call, ask_user, refuse. "
    "Never answer the user's task directly. "
    "If the provided tools can fully satisfy the request, return a tool call. "
    "If required information is missing, ask a clarification question. "
    "If the request cannot be completed with the provided tools, refuse. "
    "Return a single JSON object with keys decision, tool_name, arguments."
)


def _safe_json_loads(value: Any) -> Optional[Any]:
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


def _expects_multi_tool_output(row: Dict[str, Any]) -> bool:
    task = row.get("task", "generic")
    if task == "function_calling_multi":
        return True
    metadata = row.get("metadata", {})
    return isinstance(metadata, dict) and int(metadata.get("num_tool_calls", 1)) > 1


def _build_system_prompt(row: Dict[str, Any]) -> str:
    task = row.get("task", "generic")
    valid_decisions = row.get("valid_decisions", ["tool_call"])
    joined = ", ".join(valid_decisions)
    output_shape = (
        "Return a JSON array of tool calls. Each item must contain keys name and arguments."
        if _expects_multi_tool_output(row)
        else "Return a single JSON object with keys decision, tool_name, arguments."
    )
    if task == "when2call_decision":
        return WHEN2CALL_SYSTEM_PROMPT
    if task == "function_calling":
        return (
            "You are an assistant for function calling fine-tuning. "
            f"Your final action must be exactly one of: {joined}. "
            "Prefer tool_call when a provided tool can satisfy the request. "
            f"{output_shape} "
            "Do not add explanations outside the JSON."
        )
    if task == "function_calling_multi":
        return (
            "You are an assistant for multi-step function calling fine-tuning. "
            "The request requires multiple tool calls. "
            "Return a JSON array of tool calls. "
            "Each item must contain keys name and arguments. "
            "Do not add explanations outside the JSON."
        )
    return (
        "You are an assistant for structured tool-use fine-tuning. "
        f"Your final action must be exactly one of: {joined}. "
        f"{output_shape}"
    )


def _build_prompt_messages_generic(row: Dict[str, Any], messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    valid_decisions = row.get("valid_decisions", ["tool_call"])
    task = row.get("task", "generic")
    if _expects_multi_tool_output(row):
        instruction = {
            "task": task,
            "valid_decisions": valid_decisions,
            "output_schema": [
                {
                    "name": "tool_name",
                    "arguments": DEFAULT_OUTPUT_SCHEMA["arguments"],
                }
            ],
            "tools": tools,
        }
    else:
        instruction = {
            "task": task,
            "valid_decisions": valid_decisions,
            "output_schema": {
                "decision": " | ".join(valid_decisions),
                "tool_name": DEFAULT_OUTPUT_SCHEMA["tool_name"],
                "arguments": DEFAULT_OUTPUT_SCHEMA["arguments"],
            },
            "tools": tools,
        }

    if task == "when2call_decision":
        extra_rules = (
            "Important rules:\n"
            f"- Your final action must be exactly one of {', '.join(valid_decisions)}.\n"
            "- Do not answer the user's request directly.\n"
            "- If no provided tool can complete the request, choose refuse.\n"
            "- If some required argument is missing, choose ask_user.\n"
            "- If the request is fully supported by a provided tool, choose tool_call.\n"
            "- Output JSON only. No explanation before or after JSON."
        )
    elif task == "function_calling":
        fallback_decision = "answer_directly" if "answer_directly" in valid_decisions else valid_decisions[-1]
        extra_rules = (
            "Important rules:\n"
            f"- Your final action must be exactly one of {', '.join(valid_decisions)}.\n"
            "- If a provided tool can satisfy the request, choose tool_call.\n"
            f"- If the request should not use tools, choose {fallback_decision}.\n"
            "- For tool_call, fill tool_name and arguments precisely.\n"
            "- Output JSON only. No explanation before or after JSON."
        )
    elif task == "function_calling_multi":
        extra_rules = (
            "Important rules:\n"
            "- Return a JSON array only. No explanation before or after JSON.\n"
            "- Each array item must contain name and arguments.\n"
            "- Preserve the intended tool-call sequence from the request.\n"
            "- Fill every arguments object precisely."
        )
    else:
        extra_rules = (
            "Important rules:\n"
            f"- Your final action must be exactly one of {', '.join(valid_decisions)}.\n"
            "- Output JSON only. No explanation before or after JSON."
        )
    user_suffix = (
        "\n\nAvailable tools and output requirements:\n"
        f"{json.dumps(instruction, ensure_ascii=False)}\n"
        f"{extra_rules}\n"
        + (
            "Return a JSON array as the final answer."
            if _expects_multi_tool_output(row)
            else "Return a single JSON object as the final answer."
        )
    )
    updated = copy.deepcopy(messages)
    updated[-1] = {
        "role": updated[-1]["role"],
        "content": updated[-1]["content"] + user_suffix,
    }
    return updated


def _prepare_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = row["messages"]
    if messages and messages[0]["role"] == "system":
        base_messages = messages
    else:
        base_messages = [{"role": "system", "content": _build_system_prompt(row)}] + messages
    return _build_prompt_messages_generic(row, base_messages, row.get("tools", []))


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


def _normalize_response_text(text: str) -> str:
    label = _classify_text(text)
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


def _extract_input_ids(tokenized: Any) -> List[int]:
    if isinstance(tokenized, list):
        return tokenized
    if hasattr(tokenized, "get"):
        input_ids = tokenized.get("input_ids")
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return input_ids[0]
            return input_ids
    raise TypeError(f"Unsupported tokenized payload type: {type(tokenized)}")


def _load_when2call_pref_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            tools = [_normalize_tool(tool) for tool in raw.get("tools", [])]
            messages = raw.get("messages", [])
            chosen_text = _normalize_response_text(raw["chosen_response"]["content"])
            rejected_text = _normalize_response_text(raw["rejected_response"]["content"])
            if chosen_text == rejected_text:
                continue
            rows.append(
                {
                    "id": f"when2call_pref_{idx}",
                    "dataset": "when2call_pref",
                    "task": "when2call_decision",
                    "valid_decisions": ["tool_call", "ask_user", "refuse"],
                    "messages": messages,
                    "tools": tools,
                    "chosen_text": chosen_text,
                    "rejected_text": rejected_text,
                    "metadata": {
                        "source_format": "when2call_pref_raw",
                    },
                }
            )
    return rows


def _load_generic_pref_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            messages = raw.get("messages")
            chosen_text = raw.get("chosen_text")
            rejected_text = raw.get("rejected_text")
            if not isinstance(messages, list) or not chosen_text or not rejected_text:
                raise ValueError(f"Invalid generic preference row at line {idx + 1}")
            rows.append(
                {
                    "id": raw.get("id", f"pref_{idx}"),
                    "dataset": raw.get("dataset", "generic_pref"),
                    "task": raw.get("task", "function_calling"),
                    "valid_decisions": raw.get("valid_decisions", ["tool_call"]),
                    "messages": messages,
                    "tools": [_normalize_tool(tool) for tool in raw.get("tools", [])],
                    "chosen_text": chosen_text,
                    "rejected_text": rejected_text,
                    "metadata": raw.get("metadata", {}),
                }
            )
    return rows


def _load_pref_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first_nonempty = ""
        for line in f:
            line = line.strip()
            if line:
                first_nonempty = line
                break
    if not first_nonempty:
        return []
    first = json.loads(first_nonempty)
    if "chosen_response" in first and "rejected_response" in first:
        return _load_when2call_pref_rows(path)
    if "chosen_text" in first and "rejected_text" in first:
        return _load_generic_pref_rows(path)
    raise ValueError(f"Unsupported preference data format: {path}")


class DpoPreferenceDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.features = []
        for row in rows:
            prompt_messages = _prepare_messages(row)
            prompt_ids = _extract_input_ids(
                tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            chosen_ids = _extract_input_ids(
                tokenizer.apply_chat_template(
                    prompt_messages + [{"role": "assistant", "content": row["chosen_text"]}],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )
            rejected_ids = _extract_input_ids(
                tokenizer.apply_chat_template(
                    prompt_messages + [{"role": "assistant", "content": row["rejected_text"]}],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )

            chosen_prompt_len = min(len(prompt_ids), len(chosen_ids))
            rejected_prompt_len = min(len(prompt_ids), len(rejected_ids))
            if len(chosen_ids) > max_length:
                chosen_ids = chosen_ids[-max_length:]
                chosen_prompt_len = min(chosen_prompt_len, len(chosen_ids))
            if len(rejected_ids) > max_length:
                rejected_ids = rejected_ids[-max_length:]
                rejected_prompt_len = min(rejected_prompt_len, len(rejected_ids))

            self.features.append(
                {
                    "chosen_input_ids": chosen_ids,
                    "chosen_attention_mask": [1] * len(chosen_ids),
                    "chosen_labels": [-100] * chosen_prompt_len + chosen_ids[chosen_prompt_len:],
                    "rejected_input_ids": rejected_ids,
                    "rejected_attention_mask": [1] * len(rejected_ids),
                    "rejected_labels": [-100] * rejected_prompt_len + rejected_ids[rejected_prompt_len:],
                    "task": row.get("task", "generic"),
                }
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.features[idx]


@dataclass
class DpoDataCollator:
    tokenizer: Any

    def _pad(self, sequences: Sequence[List[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        return torch.tensor([seq + [pad_value] * (max_len - len(seq)) for seq in sequences], dtype=torch.long)

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        return {
            "chosen_input_ids": self._pad([f["chosen_input_ids"] for f in features], pad_id),
            "chosen_attention_mask": self._pad([f["chosen_attention_mask"] for f in features], 0),
            "chosen_labels": self._pad([f["chosen_labels"] for f in features], -100),
            "rejected_input_ids": self._pad([f["rejected_input_ids"] for f in features], pad_id),
            "rejected_attention_mask": self._pad([f["rejected_attention_mask"] for f in features], 0),
            "rejected_labels": self._pad([f["rejected_labels"] for f in features], -100),
        }


def _split_rows(rows: List[Dict[str, Any]], eval_ratio: float, seed: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_size = int(len(shuffled) * eval_ratio)
    if eval_ratio > 0 and eval_size == 0:
        eval_size = 1
    if eval_size >= len(shuffled):
        eval_size = max(1, len(shuffled) - 1)
    return shuffled[eval_size:], shuffled[:eval_size]


def _estimate_total_steps(num_examples: int, batch_size: int, grad_accum: int, epochs: float) -> int:
    if num_examples <= 0:
        return 0
    per_step_examples = max(1, batch_size * grad_accum)
    steps_per_epoch = math.ceil(num_examples / per_step_examples)
    return max(1, math.ceil(steps_per_epoch * epochs))


def _sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(shift_logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    return token_logps.sum(dim=-1)


class PreferenceTrainer(Trainer):
    def __init__(
        self,
        beta: float,
        ref_model: torch.nn.Module,
        sft_loss_weight: float,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.ref_model = ref_model
        self.sft_loss_weight = sft_loss_weight
        self._ref_model_device: Optional[torch.device] = None
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _move_batch(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _ensure_ref_model_device(self, target_device: torch.device) -> torch.device:
        if self._ref_model_device is None:
            self._ref_model_device = next(self.ref_model.parameters()).device
        if self._ref_model_device != target_device:
            self.ref_model.to(target_device)
            self._ref_model_device = target_device
        return self._ref_model_device

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        policy_chosen_logps = _sequence_logps(chosen_outputs.logits, inputs["chosen_labels"])
        policy_rejected_logps = _sequence_logps(rejected_outputs.logits, inputs["rejected_labels"])

        with torch.no_grad():
            ref_device = self._ensure_ref_model_device(policy_chosen_logps.device)
            ref_inputs = self._move_batch(inputs, ref_device)
            ref_chosen = self.ref_model(
                input_ids=ref_inputs["chosen_input_ids"],
                attention_mask=ref_inputs["chosen_attention_mask"],
            )
            ref_rejected = self.ref_model(
                input_ids=ref_inputs["rejected_input_ids"],
                attention_mask=ref_inputs["rejected_attention_mask"],
            )
            ref_chosen_logps = _sequence_logps(ref_chosen.logits, ref_inputs["chosen_labels"]).to(policy_chosen_logps.device)
            ref_rejected_logps = _sequence_logps(ref_rejected.logits, ref_inputs["rejected_labels"]).to(policy_rejected_logps.device)

        logits = self.beta * (
            (policy_chosen_logps - policy_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        )
        losses = -F.logsigmoid(logits)
        dpo_loss = losses.mean()

        chosen_lm_loss = None
        if self.sft_loss_weight > 0:
            chosen_lm_outputs = model(
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"],
                labels=inputs["chosen_labels"],
            )
            chosen_lm_loss = chosen_lm_outputs.loss
            loss = dpo_loss + self.sft_loss_weight * chosen_lm_loss
        else:
            loss = dpo_loss

        if return_outputs:
            outputs = {
                "policy_chosen_logps": policy_chosen_logps.detach(),
                "policy_rejected_logps": policy_rejected_logps.detach(),
                "losses": losses.detach(),
                "dpo_loss": dpo_loss.detach(),
            }
            if chosen_lm_loss is not None:
                outputs["chosen_lm_loss"] = chosen_lm_loss.detach()
            return loss, outputs
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach()
        if prediction_loss_only:
            return loss, None, None
        return loss, outputs, None


def _load_policy_model(
    base_model_path: str,
    adapter_init_path: Optional[str],
    bf16: bool,
    fp16: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
):
    model_kwargs = {"trust_remote_code": True}
    if bf16:
        model_kwargs["dtype"] = torch.bfloat16
    elif fp16:
        model_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if adapter_init_path:
        model = PeftModel.from_pretrained(model, adapter_init_path, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            ),
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def _load_reference_model(
    base_model_path: str,
    adapter_init_path: Optional[str],
    bf16: bool,
    fp16: bool,
):
    model_kwargs = {"trust_remote_code": True}
    if bf16:
        model_kwargs["dtype"] = torch.bfloat16
    elif fp16:
        model_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    model.config.use_cache = False
    if adapter_init_path:
        model = PeftModel.from_pretrained(model, adapter_init_path, is_trainable=False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--train-data-path", default="data/raw/when2call/train/when2call_train_pref.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-init-path")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft-loss-weight", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = _load_pref_rows(args.train_data_path)
    train_rows, eval_rows = _split_rows(rows, args.eval_ratio, args.seed)

    train_dataset = DpoPreferenceDataset(train_rows, tokenizer, args.max_length)
    eval_dataset = DpoPreferenceDataset(eval_rows, tokenizer, args.max_length) if eval_rows else None
    data_collator = DpoDataCollator(tokenizer)

    policy_model = _load_policy_model(
        base_model_path=args.base_model_path,
        adapter_init_path=args.adapter_init_path,
        bf16=args.bf16,
        fp16=args.fp16,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    ref_model = _load_reference_model(
        base_model_path=args.base_model_path,
        adapter_init_path=args.adapter_init_path,
        bf16=args.bf16,
        fp16=args.fp16,
    )

    estimated_total_steps = _estimate_total_steps(
        num_examples=len(train_rows),
        batch_size=args.per_device_train_batch_size,
        grad_accum=args.gradient_accumulation_steps,
        epochs=args.num_train_epochs,
    )
    warmup_steps = int(estimated_total_steps * args.warmup_ratio) if args.max_steps <= 0 else int(args.max_steps * args.warmup_ratio)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "dataset_split.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_size": len(train_rows),
                "eval_size": len(eval_rows),
                "train_data_path": args.train_data_path,
                "adapter_init_path": args.adapter_init_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    trainer = PreferenceTrainer(
        beta=args.beta,
        ref_model=ref_model,
        sft_loss_weight=args.sft_loss_weight,
        model=policy_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
