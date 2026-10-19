"""Score model predictions against the answer key and build the error table.

  python evaluate.py --split data/study.jsonl --preds preds/base_study.jsonl preds/ft_study.jsonl
"""
import argparse
import collections
import json
from pathlib import Path

LABELS = "ABCD"


def error_category(t, choice):
    if choice == "INVALID":
        return "invalid_output"
    if choice == "NONE":
        return "none" if t["best_choice"] == "NONE" else "unnecessary_none"
    car = t["cars"][LABELS.index(choice)]
    if car["violations"]:
        order = {"note": 0, "implicit": 1, "explicit": 2}
        code, vis = sorted(car["violations"], key=lambda v: order[v[1]])[0]
        if vis == "note":
            return "missed_hidden_fees" if code == "over_budget" else "missed_seller_note_red_flag"
        return "missed_implicit_need" if vis == "implicit" else "violated_explicit_requirement"
    if choice not in t["acceptable_choices"]:
        return "dominated_tradeoff"
    return "none"


def evaluate(trials, preds):
    p = {r["trial_id"]: r for r in preds}
    rows, flag_tp, flag_fn, flag_fp, clean_n = [], 0, 0, 0, 0
    for t in trials:
        r = p[t["trial_id"]]
        ch = r["model_choice"]
        err = error_category(t, ch)
        for l, c in zip(LABELS, t["cars"]):
            noted = any(v == "note" for _, v in c["violations"])
            flagged = l in r.get("model_red_flags", [])
            if noted:
                flag_tp += flagged
                flag_fn += not flagged
            elif not c["violations"]:
                clean_n += 1
                flag_fp += flagged
        rows.append({
            "trial_id": t["trial_id"], "trial_type": t["trial_type"], "category": t["category"],
            "model_choice": ch, "best_choice": t["best_choice"], "acceptable_choices": ";".join(t["acceptable_choices"]),
            "acceptable": err == "none", "exact_best": ch == t["best_choice"],
            "constraint_compliant": err not in ("missed_seller_note_red_flag", "missed_hidden_fees", "missed_implicit_need",
                                                "violated_explicit_requirement", "invalid_output", "unnecessary_none"),
            "error_category": err,
        })
    n = len(rows)
    none_pred = [x for x in rows if x["model_choice"] == "NONE"]
    none_gold = [x for x in rows if x["best_choice"] == "NONE"]
    errors = collections.Counter(x["error_category"] for x in rows if x["error_category"] != "none")
    by_type = {}
    for k in ["clear", "trap", "tradeoff", "none"]:
        sub = [x for x in rows if x["trial_type"] == k]
        by_type[k] = round(sum(x["acceptable"] for x in sub) / len(sub), 3)
    summary = {
        "n": n,
        "acceptable_rate": round(sum(x["acceptable"] for x in rows) / n, 3),
        "exact_best_rate": round(sum(x["exact_best"] for x in rows) / n, 3),
        "constraint_compliance_rate": round(sum(x["constraint_compliant"] for x in rows) / n, 3),
        "none_precision": round(sum(x["best_choice"] == "NONE" for x in none_pred) / max(1, len(none_pred)), 3),
        "none_recall": round(sum(x["model_choice"] == "NONE" for x in none_gold) / max(1, len(none_gold)), 3),
        "red_flag_recall": round(flag_tp / max(1, flag_tp + flag_fn), 3),
        "red_flag_false_alarm_rate": round(flag_fp / max(1, clean_n), 3),
        "invalid_outputs": sum(x["model_choice"] == "INVALID" for x in rows),
        "acceptable_by_trial_type": by_type,
        "error_distribution": {k: {"count": v, "pct_of_errors": round(100 * v / sum(errors.values()), 1)}
                               for k, v in errors.most_common()},
    }
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/study.jsonl")
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    trials = [json.loads(l) for l in open(args.split)]
    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    for pf in args.preds:
        preds = [json.loads(l) for l in open(pf)]
        rows, summary = evaluate(trials, preds)
        name = Path(pf).stem
        with open(out / f"{name}_summary.json", "w") as f:
            json.dump(summary, f, indent=1)
        with open(out / f"{name}_scored.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"== {name}\n{json.dumps(summary, indent=1)}")


if __name__ == "__main__":
    main()
