"""Run the car advisor live on one study trial and print the result (for the report screenshot).

  CUDA_VISIBLE_DEVICES=0 python demo.py --model /path/Qwen3-4B-Instruct-2507 --lora model/lora_adapter --trial CAR-130
"""
import argparse
import json
import textwrap

from infer import generate, load_model
from prompts import user_prompt, parse_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--split", default="data/study.jsonl")
    ap.add_argument("--trial", default="CAR-001")
    args = ap.parse_args()
    trial = next(t for t in map(json.loads, open(args.split)) if t["trial_id"] == args.trial)
    tok, model = load_model(args.model, args.lora)
    text = generate(tok, model, [trial], batch_size=1)[0]
    rec, flags, obj = parse_output(text)

    print("=" * 90)
    print(f"TRIAL {trial['trial_id']}  ({'fine-tuned' if args.lora else 'base'} Qwen3-4B-Instruct)")
    print("=" * 90)
    print(textwrap.fill(user_prompt(trial).split("\n\nLISTINGS:")[0], 90))
    print("\nLISTINGS:" + user_prompt(trial).split("\n\nLISTINGS:")[1])
    print("\n" + "-" * 90 + "\nMODEL OUTPUT")
    print(json.dumps(obj, indent=2) if obj else text)
    print("-" * 90)
    print(f"Recommendation: {rec} | flagged: {flags or 'none'}")
    print(f"Answer key: acceptable = {trial['acceptable_choices']}, best = {trial['best_choice']}")


if __name__ == "__main__":
    main()
