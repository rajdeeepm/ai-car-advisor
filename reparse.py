"""Re-derive model_choice / model_red_flags from raw outputs (after parser changes).

  python reparse.py preds/base_study.jsonl preds/ft_study.jsonl
"""
import json
import sys

from prompts import parse_output

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        r["model_choice"], r["model_red_flags"], _ = parse_output(r["raw_output"])
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(path, sum(r["model_choice"] == "INVALID" for r in rows), "invalid")
