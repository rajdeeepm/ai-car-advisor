"""Run Qwen3-4B-Instruct (optionally with a LoRA adapter) on a split.

Uses batched Hugging Face generation with greedy decoding (deterministic).

  python infer.py --model /path/Qwen3-4B-Instruct-2507 --split data/study.jsonl --out preds/base_study.jsonl
  python infer.py --model ... --lora runs/lora_r16/best --split data/study.jsonl --out preds/ft_study.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import messages, parse_output


def load_model(model_path, lora=None):
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, trials, batch_size=16, max_new_tokens=1200):
    texts = []
    for i in range(0, len(trials), batch_size):
        batch = trials[i:i + batch_size]
        prompts = [tok.apply_chat_template(messages(t), tokenize=False, add_generation_prompt=True) for t in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {min(i + batch_size, len(trials))}/{len(trials)}", flush=True)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    trials = [json.loads(l) for l in open(args.split)][: args.limit]
    tok, model = load_model(args.model, args.lora)
    t0 = time.time()
    texts = generate(tok, model, trials, args.batch_size)
    secs = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for t, text in zip(trials, texts):
            rec, flags, _ = parse_output(text)
            f.write(json.dumps({"trial_id": t["trial_id"], "model_choice": rec, "model_red_flags": flags,
                                "raw_output": text}) + "\n")
    print(f"wrote {len(trials)} predictions to {args.out} in {secs:.1f}s")


if __name__ == "__main__":
    main()
