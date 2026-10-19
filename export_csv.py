"""Build the submission CSVs from the study split + scored predictions.

  python export_csv.py --model-preds preds/ft_study.jsonl --baseline-preds preds/base_study.jsonl

Writes:
  submission/study_dataset.csv          inputs + model output + evaluation columns (200 rows)
  submission/participant_trials.csv     what Assignment 3 participants see (no answer key)
"""
import argparse
import csv
import json
from pathlib import Path

from evaluate import error_category
from gen_dataset import listing_text

LABELS = "ABCD"


def issues(c):
    return "; ".join(f"{code} ({vis})" for code, vis in c["violations"]) or "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/study.jsonl")
    ap.add_argument("--model-preds", required=True)
    ap.add_argument("--baseline-preds", default=None)
    ap.add_argument("--outdir", default="submission")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    trials = [json.loads(l) for l in open(args.split)]
    trials.sort(key=lambda t: t["trial_id"])
    mp = {r["trial_id"]: r for r in map(json.loads, open(args.model_preds))}
    bp = {r["trial_id"]: r for r in map(json.loads, open(args.baseline_preds))} if args.baseline_preds else {}

    study_rows, part_rows = [], []
    for t in trials:
        options = {f"option_{l}": listing_text(l, c) for l, c in zip(LABELS, t["cars"])}
        m = mp[t["trial_id"]]
        try:
            obj = json.loads(m["raw_output"][m["raw_output"].index("{"):m["raw_output"].rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            obj = {}
        err = error_category(t, m["model_choice"])
        row = {
            "trial_id": t["trial_id"], "buyer_category": t["category"], "trial_type": t["trial_type"], "text_style": t["text_style"],
            "buyer_profile": t["buyer_profile"], **options,
            "budget_usd": t["buyer"]["budget"], "priorities": ">".join(t["buyer"]["priorities"]),
            "hard_requirements_json": json.dumps(t["buyer"]["req"]),
            **{f"issues_{l}": issues(c) for l, c in zip(LABELS, t["cars"])},
            "feasible_options": ";".join(t["feasible_options"]) or "NONE",
            "acceptable_choices": ";".join(t["acceptable_choices"]),
            "best_choice": t["best_choice"],
            "model_choice": m["model_choice"],
            "model_red_flags": ";".join(m["model_red_flags"]),
            "model_reason": obj.get("reason", ""), "model_tradeoffs": obj.get("tradeoffs", ""),
            "model_raw_output": m["raw_output"],
            "model_acceptable": err == "none", "model_exact_best": m["model_choice"] == t["best_choice"],
            "model_error_category": err,
        }
        if bp:
            b = bp[t["trial_id"]]
            berr = error_category(t, b["model_choice"])
            row.update({"baseline_choice": b["model_choice"], "baseline_acceptable": berr == "none",
                        "baseline_error_category": berr})
        study_rows.append(row)
        part_rows.append({"trial_id": t["trial_id"], "buyer_profile": t["buyer_profile"], **options,
                          "ai_recommendation": m["model_choice"], "ai_reason": obj.get("reason", ""),
                          "ai_red_flags": json.dumps(obj.get("red_flags", [])),
                          "ai_option_checks": json.dumps(obj.get("option_checks", {}))})

    for name, rows in [("study_dataset.csv", study_rows), ("participant_trials.csv", part_rows)]:
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", out / name, len(rows))


if __name__ == "__main__":
    main()
