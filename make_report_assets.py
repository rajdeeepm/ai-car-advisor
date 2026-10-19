"""Build the report figures and tables from saved predictions (no GPU needed).

  python make_report_assets.py
Writes results/figures/*.png and results/report_tables.md
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from evaluate import evaluate

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OUT = Path("results")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def load(p):
    return [json.loads(l) for l in open(p)]


def loss_curve():
    hist = json.load(open("model/log_history.json"))
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(*zip(*tr), color=BLUE, lw=2, label="Training loss")
    ax.plot(*zip(*ev), color=ORANGE, lw=2, marker="o", ms=5, label="Validation loss",
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Optimizer step (batch 8; 2 epochs = 406 steps)")
    ax.set_ylabel("Loss (log scale)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, labelcolor=INK)
    ax.set_title("LoRA fine-tuning of Qwen3-4B-Instruct", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "loss_curve.png", dpi=200)


def metric_bars(sb, sf):
    keys = [("acceptable_rate", "Acceptable\nchoice"), ("exact_best_rate", "Exact\nbest pick"),
            ("constraint_compliance_rate", "Meets all\nrequirements"), ("red_flag_recall", "Red-flag\nrecall"),
            ("none_recall", "NONE\nrecall")]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = range(len(keys))
    w = 0.38
    for off, s, color, label in [(-w / 2 - 0.01, sb, BLUE, "Base Qwen3-4B"), (w / 2 + 0.01, sf, ORANGE, "Fine-tuned (LoRA)")]:
        vals = [100 * s[k] for k, _ in keys]
        bars = ax.bar([i + off for i in x], vals, w, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(list(x), [l for _, l in keys])
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of study trials / listings\n(higher is better)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.12), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG / "base_vs_finetuned.png", dpi=200)


ROOT_TEXT = {
    "false_alarm_benign_note": "False alarm on a harmless note",
    "tradeoff_comparison_slip": "Trade-off comparison slip",
    "number_comparison_slip": "Number comparison slip",
    "missed_indirect_red_flag": "Missed indirectly worded deal-breaker",
    "inconsistent_implied_need": "Implied need applied inconsistently",
}


def error_figure(sb, sf):
    """(a) outcome categories, base vs fine-tuned; (b) hand-labeled root causes of fine-tuned errors."""
    import csv
    db, df = sb["error_distribution"], sf["error_distribution"]
    cats = sorted(set(db) | set(df), key=lambda c: (df.get(c, {"count": 0})["count"], db.get(c, {"count": 0})["count"]))
    roots = list(csv.DictReader(open("results/ft_error_root_causes.csv")))
    order = [k for k in ROOT_TEXT if any(r["root_cause"] == k for r in roots)][::-1]
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 6.2), gridspec_kw={"height_ratios": [len(cats), len(order)]})
    h = 0.38
    for off, d, color, label in [(h / 2 + 0.01, db, BLUE, "Base Qwen3-4B"), (-h / 2 - 0.01, df, ORANGE, "Fine-tuned (LoRA)")]:
        vals = [d.get(c, {"count": 0})["count"] for c in cats]
        a.barh([i + off for i in range(len(cats))], vals, h, color=color, label=label)
        for i, v in enumerate(vals):
            a.text(v + 0.3, i + off, str(v), va="center", fontsize=8, color=INK2)
    a.set_yticks(range(len(cats)), [ERROR_TEXT[c] for c in cats])
    a.set_xlabel("Errors on the 200 study trials (fewer is better)")
    a.set_title("(a) Outcome of each error", color=INK, loc="left", fontsize=11)
    a.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.08), labelcolor=INK)
    for k, ax in enumerate([a, b]):
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    counts = [sum(r["root_cause"] == k for r in roots) for k in order]
    natural = [sum(r["root_cause"] == k and r["text_style"] == "natural" for r in roots) for k in order]
    b.barh(range(len(order)), counts, 0.55, color=ORANGE)
    for i, (n, nat) in enumerate(zip(counts, natural)):
        b.text(n + 0.08, i, f"{n}  ({nat} on hand-written text)", va="center", fontsize=8, color=INK2)
    b.set_yticks(range(len(order)), [ROOT_TEXT[k] for k in order])
    b.set_xlim(0, max(counts) + 3.6)
    b.set_xlabel("Fine-tuned errors (fewer is better)")
    b.set_title("(b) Root cause of each fine-tuned error", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "error_analysis.png", dpi=200)


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    study = load("data/study.jsonl")
    study_v1 = load("data/study_v1.jsonl")
    res = {}
    for ver, trials, d in [("v1", study_v1, "preds/v1"), ("v2", study, "preds")]:
        for m in ["base", "ft"]:
            res[(ver, m)] = evaluate(trials, load(f"{d}/{m}_study.jsonl"))
    sb, sf = res[("v2", "base")][1], res[("v2", "ft")][1]
    loss_curve()
    metric_bars(sb, sf)
    error_figure(sb, sf)

    md = ["# Report tables (generated by make_report_assets.py)\n"]
    keys = ["acceptable_rate", "exact_best_rate", "constraint_compliance_rate", "red_flag_recall",
            "red_flag_false_alarm_rate", "none_precision", "none_recall", "invalid_outputs"]
    md.append("## Final study set (v2): base vs fine-tuned\n")
    md.append(table([[k, sb[k], sf[k]] for k in keys], ["metric", "base", "fine-tuned"]))
    md.append("\n## Acceptable rate by trial type (v2)\n")
    md.append(table([[k, sb["acceptable_by_trial_type"][k], sf["acceptable_by_trial_type"][k]]
                     for k in ["clear", "trap", "tradeoff", "none"]], ["trial type", "base", "fine-tuned"]))
    style = {t["trial_id"]: t["text_style"] for t in study}
    rows = []
    for m in ["base", "ft"]:
        sc = res[("v2", m)][0]
        for s_ in ["template", "natural"]:
            sub = [r for r in sc if style[r["trial_id"]] == s_]
            rows.append([m, s_, len(sub), round(sum(r["acceptable"] for r in sub) / len(sub), 3)])
    md.append("\n## Acceptable rate by text style (v2)\n")
    md.append(table(rows, ["model", "text style", "n", "acceptable"]))
    for m, name in [("ft", "Fine-tuned"), ("base", "Base")]:
        dist = res[("v2", m)][1]["error_distribution"]
        md.append(f"\n## Error distribution, {name} model (v2)\n")
        md.append(table([[k, v["count"], f"{v['pct_of_errors']}%"] for k, v in dist.items()],
                        ["error pattern", "count", "% of errors"]))
    md.append("\n## Iteration history (acceptable rate)\n")
    md.append(table([[ver, res[(ver, 'base')][1]["acceptable_rate"], res[(ver, 'ft')][1]["acceptable_rate"]]
                     for ver in ["v1", "v2"]], ["study set", "base", "fine-tuned"]))
    # list fine-tuned errors for the qualitative analysis
    md.append("\n## Fine-tuned model errors (v2), trial by trial\n")
    err = [r for r in res[("v2", "ft")][0] if r["error_category"] != "none"]
    md.append(table([[r["trial_id"], r["trial_type"], style[r["trial_id"]], r["model_choice"], r["acceptable_choices"],
                      r["error_category"]] for r in err],
                    ["trial", "type", "text", "model", "acceptable", "error"]))
    (OUT / "report_tables.md").write_text("\n".join(md) + "\n")
    write_latex(res, study, style)
    print("wrote results/report_tables.md, results/latex/ and results/figures/")


ERROR_TEXT = {
    "missed_seller_note_red_flag": "Missed a deal-breaker in the seller note",
    "missed_hidden_fees": "Missed required fees that break the budget",
    "missed_implicit_need": "Missed a need implied by the buyer's story",
    "violated_explicit_requirement": "Violated a stated requirement",
    "unnecessary_none": "Said None although a car qualified",
    "dominated_tradeoff": "Picked a qualifying car beaten on both priorities",
    "invalid_output": "Unparseable output",
}


def write_latex(res, study, style):
    """LaTeX macros and tables for report/report.tex, so no number in the PDF is typed by hand."""
    tex = OUT / "latex"
    tex.mkdir(exist_ok=True)
    pct = lambda v: f"{100 * v:.1f}\\%"
    macros = []
    names = {"acceptable_rate": "Acc", "exact_best_rate": "Best", "constraint_compliance_rate": "Comply",
             "red_flag_recall": "FlagRecall", "red_flag_false_alarm_rate": "FalseAlarm",
             "none_precision": "NonePrec", "none_recall": "NoneRecall"}
    for ver, vname in [("v1", "One"), ("v2", "")]:
        for m, mname in [("base", "Base"), ("ft", "Ft")]:
            s = res[(ver, m)][1]
            for k, short in names.items():
                macros.append(f"\\newcommand{{\\{mname}{short}{vname}}}{{{pct(s[k])}}}")
            macros.append(f"\\newcommand{{\\{mname}Invalid{vname}}}{{{s['invalid_outputs']}}}")
            macros.append(f"\\newcommand{{\\{mname}Errors{vname}}}"
                          f"{{{sum(v['count'] for v in s['error_distribution'].values())}}}")
            for s_ in ["template", "natural"]:
                sub = [r for r in res[(ver, m)][0] if style.get(r["trial_id"]) == s_] if ver == "v2" else []
                if sub:
                    macros.append(f"\\newcommand{{\\{mname}Acc{s_.capitalize()}}}"
                                  f"{{{pct(sum(r['acceptable'] for r in sub) / len(sub))}}}")
    (tex / "numbers.tex").write_text("\n".join(macros) + "\n")

    sb, sf = res[("v2", "base")][1], res[("v2", "ft")][1]
    rows = [("Acceptable choice (primary)", "acceptable_rate"), ("Exact best choice", "exact_best_rate"),
            ("Meets all requirements", "constraint_compliance_rate"), ("Red-flag recall", "red_flag_recall"),
            ("Red-flag false alarms ($\\downarrow$)", "red_flag_false_alarm_rate"), ("None precision", "none_precision"),
            ("None recall", "none_recall")]
    rows = [(l if "downarrow" in l else l.replace(" (primary)", ", primary") + " ($\\uparrow$)", k)
            for l, k in rows]
    lines = ["\\begin{tabular}{lrr}", "\\toprule", "Metric & Base Qwen3-4B & Fine-tuned (LoRA) \\\\", "\\midrule"]
    lines += [f"{label} & {pct(sb[k])} & {pct(sf[k])} \\\\" for label, k in rows]
    lines += [f"Unparseable outputs ($\\downarrow$) & {sb['invalid_outputs']} & {sf['invalid_outputs']} \\\\", "\\bottomrule",
              "\\end{tabular}"]
    (tex / "tab_metrics.tex").write_text("\n".join(lines) + "\n")

    n_type = {k: sum(t["trial_type"] == k for t in study) for k in ["clear", "trap", "tradeoff", "none"]}
    n_style = {k: sum(t["text_style"] == k for t in study) for k in ["template", "natural"]}
    lines = ["\\begin{tabular}{lrrr}", "\\toprule", "Subset & Trials & Base & Fine-tuned \\\\", "\\midrule"]
    for k, label in [("clear", "Clear"), ("trap", "Trap"), ("tradeoff", "Trade-off"), ("none", "No suitable car")]:
        lines.append(f"{label} & {n_type[k]} & {pct(sb['acceptable_by_trial_type'][k])} & "
                     f"{pct(sf['acceptable_by_trial_type'][k])} \\\\")
    lines.append("\\midrule")
    for k, label in [("template", "Generated wording"), ("natural", "Hand-written wording")]:
        accs = []
        for m in ["base", "ft"]:
            sub = [r for r in res[("v2", m)][0] if style[r["trial_id"]] == k]
            accs.append(pct(sum(r["acceptable"] for r in sub) / len(sub)))
        lines.append(f"{label} & {n_style[k]} & {accs[0]} & {accs[1]} \\\\")
    lines += ["\\midrule", f"All & {len(study)} & {pct(sb['acceptable_rate'])} & {pct(sf['acceptable_rate'])} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    (tex / "tab_subsets.tex").write_text("\n".join(lines) + "\n")

    db, df = sb["error_distribution"], sf["error_distribution"]
    nb, nf = sum(v["count"] for v in db.values()), sum(v["count"] for v in df.values())
    cats = sorted(set(db) | set(df), key=lambda c: (-df.get(c, {"count": 0})["count"], -db.get(c, {"count": 0})["count"]))
    cell = lambda d, c: f"{d[c]['count']} ({d[c]['pct_of_errors']:.1f}\\%)" if c in d else "0"
    lines = ["\\begin{tabular}{lrr}", "\\toprule", "Error pattern & Fine-tuned & Base \\\\", "\\midrule"]
    lines += [f"{ERROR_TEXT[c]} & {cell(df, c)} & {cell(db, c)} \\\\" for c in cats]
    lines += ["\\midrule", f"Total errors (of 200 trials) & {nf} & {nb} \\\\", "\\bottomrule", "\\end{tabular}"]
    (tex / "tab_errors.tex").write_text("\n".join(lines) + "\n")

    lines = ["\\begin{tabular}{lrr}", "\\toprule", "Study set & Base & Fine-tuned \\\\", "\\midrule",
             f"Iteration 1 (generated wording only) & {pct(res[('v1', 'base')][1]['acceptable_rate'])} & "
             f"{pct(res[('v1', 'ft')][1]['acceptable_rate'])} \\\\",
             f"Iteration 2 (final, 40\\% hand-written) & {pct(sb['acceptable_rate'])} & {pct(sf['acceptable_rate'])} \\\\",
             "\\bottomrule", "\\end{tabular}"]
    (tex / "tab_iterations.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
