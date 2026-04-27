#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


DEFAULT_OUTPUT_SCHEMA = {
    "decision": "string",
    "tool_name": "string_or_null",
    "arguments": "object",
}


def _expects_multi_tool_output(row: Dict[str, Any]) -> bool:
    task = row.get("task", "generic")
    if task == "function_calling_multi":
        return True
    metadata = row.get("metadata", {})
    return isinstance(metadata, dict) and int(metadata.get("num_tool_calls", 1)) > 1


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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
        return (
            "You are an assistant for When2Call evaluation. "
            f"Your final action must be exactly one of: {joined}. "
            "Never answer the user's task directly. "
            "If the provided tools can fully satisfy the request, return a tool call. "
            "If required information is missing, ask a clarification question. "
            "If the request cannot be completed with the provided tools, refuse. "
            f"{output_shape}"
        )
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


def _build_qwen_prompt(row: Dict[str, Any], messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> List[Dict[str, str]]:
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
    updated = list(messages)
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
    return _build_qwen_prompt(row, base_messages, row.get("tools", []))


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


class JsonSFTDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.features = []
        for row in rows:
            prompt_messages = _prepare_messages(row)
            assistant_text = row.get("target_text")
            if not assistant_text:
                assistant_text = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))

            full_messages = prompt_messages + [{"role": "assistant", "content": assistant_text}]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            full_ids = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            prompt_ids = _extract_input_ids(prompt_ids)
            full_ids = _extract_input_ids(full_ids)
            if len(full_ids) > max_length:
                full_ids = full_ids[-max_length:]
                prompt_ids = prompt_ids[-min(len(prompt_ids), max_length):]

            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[: len(full_ids)]
            attention_mask = [1] * len(full_ids)
            self.features.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.features[idx]


@dataclass
class SupervisedDataCollator:
    tokenizer: Any

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(len(feature["input_ids"]) for feature in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def _split_rows(rows: List[Dict[str, Any]], eval_ratio: float, seed: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_size = int(len(shuffled) * eval_ratio)
    if eval_ratio > 0 and eval_size == 0:
        eval_size = 1
    if eval_size >= len(shuffled):
        eval_size = max(1, len(shuffled) - 1)
    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size:]
    return train_rows, eval_rows


def _estimate_total_steps(num_examples: int, batch_size: int, grad_accum: int, epochs: float) -> int:
    if num_examples <= 0:
        return 0
    per_step_examples = max(1, batch_size * grad_accum)
    steps_per_epoch = math.ceil(num_examples / per_step_examples)
    return max(1, math.ceil(steps_per_epoch * epochs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--train-data-path",
        default="data/processed/when2call/when2call_train_sft_fixed.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if args.bf16:
        model_kwargs["dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    rows = _load_jsonl(args.train_data_path)
    train_rows, eval_rows = _split_rows(rows, args.eval_ratio, args.seed)

    train_dataset = JsonSFTDataset(train_rows, tokenizer, args.max_length)
    eval_dataset = JsonSFTDataset(eval_rows, tokenizer, args.max_length) if eval_rows else None
    data_collator = SupervisedDataCollator(tokenizer)
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

    trainer = Trainer(
        model=model,
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
