#!/usr/bin/env python3
"""Lewi QLoRA SFT/DPO training pipeline."""
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parent
LEARNING_CURVE = ROOT / "learning_curve" / "training_runs"
LEARNING_CURVE.mkdir(parents=True, exist_ok=True)
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SFT_KEYS = {"instruction", "input", "assistant_response"}
MEMORY_KEYS = {"memory_type", "content"}
TRAJECTORY_KEYS = {"trajectory"}
DEFAULT_AGENT_SYSTEM = (
    "You are Lewi 1, an agentic reasoning system. At each turn output EXACTLY one JSON object and no markdown. "
    "Allowed actions: reason, memory_search, memory_write, code_exec, file_read, file_write, final. "
    "Reason updates the plan. Memory is experience, not truth. Tools produce observations; inspect them and re-plan "
    "when they contradict expectations. Never claim success without evidence. After failure, change strategy instead "
    "of blindly repeating it. Only use final when the goal state is actually satisfied."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_to_messages(record: dict[str, Any]) -> list[dict[str, str]] | None:
    if TRAJECTORY_KEYS.issubset(record):
        trajectory = record["trajectory"]
        if not isinstance(trajectory, list) or not trajectory:
            return None
        messages: list[dict[str, str]] = []
        system = str(record.get("system", DEFAULT_AGENT_SYSTEM))
        messages.append({"role": "system", "content": system})
        for turn in trajectory:
            if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"}:
                return None
            content = turn.get("content")
            if content is None:
                return None
            messages.append({"role": turn["role"], "content": str(content)})
        if messages[-1]["role"] != "assistant":
            return None
        return messages

    if SFT_KEYS.issubset(record):
        system = str(record["instruction"])
        user = str(record["input"])
        situation = record.get("situation_state")
        if situation:
            user = f"[Context: {situation}]\n{user}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(record["assistant_response"])},
        ]

    if MEMORY_KEYS.issubset(record):
        system = "You are recalling a stored memory to inform your reasoning. Apply it only when it actually fits the current situation."
        user = (
            f"Relevant memory ({record.get('memory_type', 'lesson')}, scope={record.get('scope', 'global')}): {record['content']}\n"
            f"Context it came from: {record.get('context', 'unspecified')}\n"
            f"Applies when: {record.get('applicable_when', 'unspecified')}\n"
            f"Does not apply when: {record.get('not_applicable_when', 'unspecified')}"
        )
        why = record.get("why", "")
        assistant = f"Understood. I'll apply this when {record.get('applicable_when', 'the described situation holds')}, and I won't apply it when {record.get('not_applicable_when', 'that condition does not hold')}." + (f" Reason it matters: {why}" if why else "")
        return [{"role": "system", "content": system}, {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return None


class TokenizedExamples(TorchDataset):
    def __init__(self, examples: list[dict[str, list[int]]]): self.examples = examples
    def __len__(self) -> int: return len(self.examples)
    def __getitem__(self, idx: int) -> dict[str, list[int]]: return self.examples[idx]


def _extract_ids(result: Any) -> list[int]:
    if isinstance(result, list): return result
    ids = result["input_ids"]
    if hasattr(ids, "tolist"): ids = ids.tolist()
    if ids and isinstance(ids[0], list): ids = ids[0]
    return list(ids)


def build_dataset(paths: list[Path], tokenizer, max_seq_len: int) -> TokenizedExamples:
    examples: list[dict[str, list[int]]] = []
    skipped = 0
    for path in paths:
        if not path.exists():
            print(f"[build_dataset] MISSING: {path}")
            continue
        for record in load_jsonl(path):
            messages = record_to_messages(record)
            if messages is None:
                skipped += 1
                continue
            prompt_ids = _extract_ids(tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True))
            full_ids = _extract_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
            if not full_ids: continue
            full_ids = full_ids[:max_seq_len]
            labels = list(full_ids)
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels[:prompt_len] = [-100] * prompt_len
            if not any(label != -100 for label in labels): continue
            examples.append({"input_ids": full_ids, "labels": labels})
    if skipped: print(f"[build_dataset] skipped {skipped} unrecognized/invalid record(s)")
    if not examples: raise ValueError("No usable training examples were built from the configured SFT data")
    return TokenizedExamples(examples)


def collate(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    if not batch: raise ValueError("Cannot collate an empty batch")
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attention_mask = [], [], []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long), "attention_mask": torch.tensor(attention_mask, dtype=torch.long)}


class LearningCurveCallback(TrainerCallback):
    def __init__(self, run_id: str): self.log_path = LEARNING_CURVE / f"{run_id}.jsonl"
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        with self.log_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps({"timestamp": time.time(), "step": state.global_step, **logs}) + "\n")


def load_base_model(base_model: str):
    if not torch.cuda.is_available(): raise RuntimeError("QLoRA requires CUDA in this configuration")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map="auto")
    return prepare_model_for_kbit_training(model)


def apply_lora(model, r: int, alpha: int, dropout: float):
    return get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))


def run_sft(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as handle: config = yaml.safe_load(handle)
    sft_paths = [ROOT / p for p in config["data"]["sft"]]
    print("SFT sources:")
    for path in sft_paths: print(f"  {path}  ({'ok' if path.exists() else 'MISSING'})")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset(sft_paths, tokenizer, args.max_seq_len)
    print(f"Built {len(dataset)} training examples")
    model = apply_lora(load_base_model(args.base_model), args.lora_r, args.lora_alpha, args.lora_dropout)
    model.print_trainable_parameters()
    run_id = args.run_id or f"sft_{int(time.time())}"
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    training_args = TrainingArguments(output_dir=args.output_dir, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, learning_rate=args.lr, logging_steps=args.logging_steps, save_strategy="epoch", fp16=compute_dtype == torch.float16, bf16=compute_dtype == torch.bfloat16, gradient_checkpointing=True, report_to=[], warmup_steps=1, lr_scheduler_type="cosine")
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=lambda batch: collate(batch, tokenizer.pad_token_id), callbacks=[LearningCurveCallback(run_id)])
    trainer.train(); trainer.save_model(args.output_dir); tokenizer.save_pretrained(args.output_dir)
    summary = {"run_id": run_id, "stage": "sft", "base_model": args.base_model, "num_examples": len(dataset), "epochs": args.epochs, "final_loss": trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None, "output_dir": args.output_dir, "timestamp": time.time()}
    with (LEARNING_CURVE / f"{run_id}_summary.json").open("w", encoding="utf-8") as handle: json.dump(summary, handle, indent=2)
    print(f"Saved adapter to {args.output_dir}")
    run_eval(args.base_model, args.output_dir, config, run_id, tokenizer)


def run_eval(base_model: str, adapter_dir: str, config: dict, run_id: str, tokenizer) -> None:
    eval_paths = [ROOT / p for p in config["data"]["evaluation"]]
    records = []
    for path in eval_paths:
        if path.exists(): records.extend(load_jsonl(path))
    print(f"Running {len(records)} held-out eval prompts for manual review...")
    if not records: return
    if not torch.cuda.is_available(): raise RuntimeError("Evaluation requires CUDA")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map={"": 0})
    model = PeftModel.from_pretrained(base, adapter_dir); model.eval()
    outputs = []
    for record in records:
        messages = [{"role": "system", "content": DEFAULT_AGENT_SYSTEM}, {"role": "user", "content": str(record["prompt"])}]
        chat_result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_ids = chat_result["input_ids"] if hasattr(chat_result, "keys") else chat_result
        attention_mask = chat_result.get("attention_mask") if hasattr(chat_result, "get") else None
        input_ids = input_ids.to("cuda")
        if attention_mask is not None: attention_mask = attention_mask.to("cuda")
        with torch.no_grad(): generated = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
        outputs.append({"id": record.get("id"), "category": record.get("category"), "prompt": record["prompt"], "success_criteria": record.get("success_criteria"), "model_output": text})
    eval_log = LEARNING_CURVE / f"{run_id}_eval_outputs.jsonl"
    with eval_log.open("w", encoding="utf-8") as handle:
        for output in outputs: handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"Wrote eval outputs to {eval_log}")


def run_dpo(args: argparse.Namespace) -> None:
    try: from trl import DPOConfig, DPOTrainer
    except ImportError as exc: raise SystemExit("DPO stage needs trl: pip install trl") from exc
    with open(args.config, "r", encoding="utf-8") as handle: config = yaml.safe_load(handle)
    pref_paths = [ROOT / p for p in config["data"]["preference"]]
    records = []
    for path in pref_paths:
        if path.exists(): records.extend(load_jsonl(path))
    if not records: raise ValueError("No preference examples found")
    dpo_dataset = Dataset.from_list([{"prompt": f"{DEFAULT_AGENT_SYSTEM}\n\n{r['input']}", "chosen": r["chosen"], "rejected": r["rejected"]} for r in records])
    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter)
    model = PeftModel.from_pretrained(load_base_model(args.base_model), args.sft_adapter, is_trainable=True)
    run_id = args.run_id or f"dpo_{int(time.time())}"
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    dpo_config = DPOConfig(output_dir=args.output_dir, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, learning_rate=args.lr, logging_steps=args.logging_steps, save_strategy="epoch", fp16=compute_dtype == torch.float16, bf16=compute_dtype == torch.bfloat16, report_to=[], warmup_steps=1)
    trainer = DPOTrainer(model=model, args=dpo_config, train_dataset=dpo_dataset, processing_class=tokenizer)
    trainer.train(); trainer.save_model(args.output_dir); tokenizer.save_pretrained(args.output_dir)
    with (LEARNING_CURVE / f"{run_id}_summary.json").open("w", encoding="utf-8") as handle: json.dump({"run_id": run_id, "stage": "dpo", "num_examples": len(dpo_dataset), "timestamp": time.time()}, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Lewi with QLoRA SFT or DPO")
    parser.add_argument("--stage", choices=["sft", "dpo"], default="sft")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default="checkpoints/lewi1-sft")
    parser.add_argument("--sft-adapter", default="checkpoints/lewi1-sft")
    parser.add_argument("--run-id")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()
    if args.max_seq_len <= 0 or args.batch_size <= 0 or args.grad_accum <= 0 or args.epochs <= 0: parser.error("sequence length, batch size, grad accumulation, and epochs must be positive")
    return args


def main() -> None:
    args = parse_args()
    run_sft(args) if args.stage == "sft" else run_dpo(args)

if __name__ == "__main__": main()
