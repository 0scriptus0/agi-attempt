#!/usr/bin/env python3
"""
Lewi 1 training script.

Scope, honestly stated: this fine-tunes an open-weight instruction model
(default: Qwen2.5-7B-Instruct) with QLoRA on your curated SFT data, and
optionally does a DPO pass on preference_pairs.jsonl. It produces a
capable, tool-using custom assistant shaped by your data and the agent
loop in api.py. It does not produce general intelligence — "AGI" here
means "the agent scaffolding + memory separation described in your
README," not a claim about the model's actual generality. Framing it
that way to yourself will make it much easier to notice when something
in the pipeline is actually broken vs. just "not AGI yet."

Stage 1 (default): SFT via QLoRA
    python train.py --stage sft

Stage 2 (optional, needs `pip install trl`): DPO on preference_pairs.jsonl,
starting from the SFT adapter
    python train.py --stage dpo --sft-adapter checkpoints/lewi1-sft

Every run appends step-level loss to learning_curve/training_runs/<run_id>.jsonl
and writes a final summary to the same directory, so you have a plain-file
training curve to look at before you wire up a real experiment tracker.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parent
LEARNING_CURVE = ROOT / "learning_curve" / "training_runs"
LEARNING_CURVE.mkdir(parents=True, exist_ok=True)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Records with this shape are conversational SFT examples.
SFT_KEYS = {"instruction", "input", "assistant_response"}
# Records with this shape are memory annotations, converted into a
# synthetic "recall and apply this memory" SFT example below.
MEMORY_KEYS = {"memory_type", "content"}


# ---------------------------------------------------------------------------
# Data loading + prompt construction
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record_to_messages(record: dict[str, Any]) -> list[dict[str, str]] | None:
    """
    Convert one dataset record into a chat-format example.

    Design choice (yours to change): situation_state, when present, is
    training-only context folded into the user turn — at inference time
    there's no situation_state field, so the model has to learn to infer
    the equivalent of it from the conversation itself. Memory-shaped
    records get converted into a synthetic "here's a relevant memory,
    apply it correctly" turn, since they don't represent conversations
    on their own.
    """

    if SFT_KEYS.issubset(record.keys()):
        system = record["instruction"]
        user = record["input"]
        situation = record.get("situation_state")
        if situation:
            user = f"[Context: {situation}]\n{user}"
        assistant = record["assistant_response"]
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]

    if MEMORY_KEYS.issubset(record.keys()):
        system = (
            "You are recalling a stored memory to inform your reasoning. "
            "Apply it only when it actually fits the current situation."
        )
        user = (
            f"Relevant memory ({record.get('memory_type', 'lesson')}, "
            f"scope={record.get('scope', 'global')}): {record['content']}\n"
            f"Context it came from: {record.get('context', 'unspecified')}\n"
            f"Applies when: {record.get('applicable_when', 'unspecified')}\n"
            f"Does not apply when: {record.get('not_applicable_when', 'unspecified')}"
        )
        why = record.get("why", "")
        assistant = (
            f"Understood. I'll apply this when {record.get('applicable_when', 'the described situation holds')}, "
            f"and I won't apply it when {record.get('not_applicable_when', 'that condition does not hold')}."
            + (f" Reason it matters: {why}" if why else "")
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]

    return None  # unrecognized shape — skip rather than guess


class TokenizedExamples(TorchDataset):
    """Plain list-backed dataset — avoids HF `datasets.Dataset`'s batched
    __getitems__ fast path, which returns a columnar dict instead of
    per-row dicts under datasets 5.0.1 and broke collate() below."""

    def __init__(self, examples: list[dict[str, list[int]]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


def _extract_ids(result: Any) -> list[int]:
    """
    tokenizer.apply_chat_template(..., tokenize=True) returns a plain
    list[int] on transformers v4, but a BatchEncoding (dict-like, with
    an 'input_ids' field) on v5 — even without return_tensors set. This
    normalizes either shape down to a flat list[int].
    """
    if isinstance(result, list):
        return result
    ids = result["input_ids"]
    # BatchEncoding may itself hold a batch dimension (list[list[int]])
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def build_dataset(paths: list[Path], tokenizer, max_seq_len: int) -> "TokenizedExamples":
    examples: list[dict[str, list[int]]] = []
    skipped = 0

    for path in paths:
        for record in load_jsonl(path):
            messages = record_to_messages(record)
            if messages is None:
                skipped += 1
                continue

            # Tokenize prompt (system+user) and full (system+user+assistant)
            # separately so we can mask the prompt tokens out of the loss —
            # the model should be trained to produce the assistant turn,
            # not to predict its own system/user context.
            prompt_ids = _extract_ids(tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
            ))
            full_ids = _extract_ids(tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            ))

            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]

            labels = list(full_ids)
            prompt_len = min(len(prompt_ids), len(full_ids))
            for i in range(prompt_len):
                labels[i] = -100

            examples.append({"input_ids": full_ids, "labels": labels})

    if skipped:
        print(f"[build_dataset] skipped {skipped} record(s) with an unrecognized shape")

    return TokenizedExamples(examples)


def collate(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attention_mask = [], [], []

    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# learning_curve logging
# ---------------------------------------------------------------------------

class LearningCurveCallback(TrainerCallback):
    """Appends every trainer log_history entry to a plain JSONL file."""

    def __init__(self, run_id: str):
        self.log_path = LEARNING_CURVE / f"{run_id}.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"timestamp": time.time(), "step": state.global_step, **logs}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_base_model(base_model: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model, r: int, alpha: int, dropout: float):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    return get_peft_model(model, lora_config)


# ---------------------------------------------------------------------------
# SFT stage
# ---------------------------------------------------------------------------

def run_sft(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    sft_paths = [ROOT / p for p in config["data"]["sft"]]
    print("SFT sources:")
    for p in sft_paths:
        print(f"  {p}  ({'ok' if p.exists() else 'MISSING'})")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(sft_paths, tokenizer, args.max_seq_len)
    print(f"Built {len(dataset)} training examples from {len(sft_paths)} file(s)")

    model = load_base_model(args.base_model)
    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    model.print_trainable_parameters()

    run_id = args.run_id or f"sft_{int(time.time())}"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],  # plain-file logging via the callback below instead
        warmup_steps=0.03,  # <1.0 = ratio of total steps, per transformers v5 API
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        callbacks=[LearningCurveCallback(run_id)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "run_id": run_id,
        "stage": "sft",
        "base_model": args.base_model,
        "num_examples": len(dataset),
        "epochs": args.epochs,
        "final_loss": trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None,
        "output_dir": args.output_dir,
        "timestamp": time.time(),
    }
    with (LEARNING_CURVE / f"{run_id}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved adapter to {args.output_dir}")
    print(f"Wrote run log to {LEARNING_CURVE / (run_id + '.jsonl')}")

    run_eval(args.base_model, args.output_dir, config, run_id, tokenizer)


def run_eval(base_model: str, adapter_dir: str, config: dict, run_id: str, tokenizer) -> None:
    """
    Generates responses on heldout_eval.jsonl for manual review. This does
    NOT auto-grade against success_criteria — that needs either a human
    reviewer or a separate LLM-judge script, which is a reasonable next
    tool to build once you have a few checkpoints to compare, but isn't
    included here to keep this script's scope honest.
    """
    eval_paths = [ROOT / p for p in config["data"]["evaluation"]]
    records = []
    for p in eval_paths:
        records.extend(load_jsonl(p))

    print(f"Running {len(records)} held-out eval prompts for manual review...")

    model = load_base_model(base_model)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    outputs = []
    for r in records:
        messages = [{"role": "user", "content": r["prompt"]}]
        chat_result = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        # Same v4/v5 shape difference as _extract_ids, but this path wants a
        # tensor (return_tensors="pt") rather than a plain list.
        input_ids = chat_result["input_ids"] if hasattr(chat_result, "keys") else chat_result
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=300, do_sample=False)
        text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
        outputs.append({
            "id": r["id"],
            "category": r.get("category"),
            "prompt": r["prompt"],
            "success_criteria": r.get("success_criteria"),
            "model_output": text,
        })

    eval_log = LEARNING_CURVE / f"{run_id}_eval_outputs.jsonl"
    with eval_log.open("w") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")
    print(f"Wrote eval outputs for manual review to {eval_log}")


# ---------------------------------------------------------------------------
# DPO stage (optional)
# ---------------------------------------------------------------------------

def run_dpo(args: argparse.Namespace) -> None:
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        raise SystemExit(
            "DPO stage needs trl: pip install trl"
        )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pref_paths = [ROOT / p for p in config["data"]["preference"]]
    records = []
    for p in pref_paths:
        records.extend(load_jsonl(p))

    # preference_pairs.jsonl has no 'instruction' field of its own, so we
    # fall back to the same default system prompt used in self_v2_aligned.
    default_system = (
        "You are Lewi 1, a general engineer, researcher, and agentic "
        "problem solver."
    )

    def to_pref_example(r):
        return {
            "prompt": f"{default_system}\n\n{r['input']}",
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }

    dpo_dataset = Dataset.from_list([to_pref_example(r) for r in records])

    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter)
    model = load_base_model(args.base_model)
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)

    run_id = args.run_id or f"dpo_{int(time.time())}"

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        bf16=True,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        callbacks=[LearningCurveCallback(run_id)],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved DPO adapter to {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["sft", "dpo"], default="sft")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--sft-adapter", default="checkpoints/lewi1-sft", help="required for --stage dpo")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = f"checkpoints/lewi1-{args.stage}"

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.stage == "sft":
        run_sft(args)
    else:
        run_dpo(args)