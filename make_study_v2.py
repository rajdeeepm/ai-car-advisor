"""Iteration 2: rewrite ~40% of study trials with hand-written natural language.

The cars, requirements and answer key are unchanged; only the wording of seller
notes and implied buyer needs changes. Trials are chosen stratified by trial type.

  python make_study_v2.py --v1 data/study_v1.jsonl --out data/study.jsonl
"""
import argparse
import json
import random

from gen_dataset import IMPLICIT, money
from natural_text import (NATURAL_BENIGN, NATURAL_BENIGN_FEE, NATURAL_FEE_NOTES, NATURAL_IMPLICIT,
                          NATURAL_RED_FLAGS)

FRACTION = 0.4


def rewrite(t, rng):
    used = set()

    def pick(pool):
        choices = [p for p in pool if p not in used] or pool
        s = rng.choice(choices)
        used.add(s)
        return s

    # implied needs in the buyer story
    for key, tpl in IMPLICIT.items():
        variants = tpl if isinstance(tpl, list) else [s for v in tpl.values() for s in v]
        for old in variants:
            if old in t["buyer_profile"]:
                if key == "seats":
                    new = pick(NATURAL_IMPLICIT["seats"][t["buyer"]["req"]["seats"][1]])
                else:
                    new = pick(NATURAL_IMPLICIT[key])
                t["buyer_profile"] = t["buyer_profile"].replace(old, new)
    # seller notes
    for c in t["cars"]:
        note_codes = [code for code, vis in c["violations"] if vis == "note"]
        if note_codes and note_codes[0] == "over_budget":
            f1 = int(c["fees"] * rng.uniform(0.55, 0.8)) // 5 * 5
            c["note"] = pick(NATURAL_FEE_NOTES).format(f1=money(f1), f2=money(c["fees"] - f1))
        elif note_codes:
            c["note"] = pick(NATURAL_RED_FLAGS[note_codes[0]])
        elif c["fees"]:
            c["note"] = pick(NATURAL_BENIGN_FEE).format(f1=money(c["fees"]))
        elif c["note"]:
            c["note"] = pick(NATURAL_BENIGN)
    t["text_style"] = "natural"
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="data/study_v1.jsonl")
    ap.add_argument("--out", default="data/study.jsonl")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    trials = [json.loads(l) for l in open(args.v1)]
    by_type = {}
    for t in trials:
        t["text_style"] = "template"
        by_type.setdefault(t["trial_type"], []).append(t)
    for group in by_type.values():
        for t in rng.sample(group, round(FRACTION * len(group))):
            rewrite(t, rng)
    with open(args.out, "w") as f:
        for t in trials:
            f.write(json.dumps(t) + "\n")
    print(f"natural: {sum(t['text_style'] == 'natural' for t in trials)} / {len(trials)}")


if __name__ == "__main__":
    main()
