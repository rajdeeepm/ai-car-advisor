"""Prompt construction, reference (training-target) responses, and output parsing.

Shared by training, inference, evaluation and the Assignment 3 interface so the
model always sees exactly the same format.
"""
import json
import re

from gen_dataset import listing_text, money

SYSTEM_PROMPT = """You are a careful used-car shopping advisor. A buyer describes their situation and four listings are shown.
Check every listing against ALL of the buyer's requirements, including ones they only imply in their story, the all-in budget (advertised price plus any required dealer fees), and anything in the seller notes that reveals hidden damage, title or mileage problems, scam-like payment requests, or major repairs needed.
Among listings that pass every requirement, prefer the one that best fits the buyer's stated priorities. If no listing passes, recommend NONE.
Respond with JSON only, in this format:
{"option_checks": {"A": "...", "B": "...", "C": "...", "D": "..."}, "red_flags": [{"option": "A", "issue": "..."}], "recommendation": "A|B|C|D|NONE", "reason": "...", "tradeoffs": "..."}"""

LABELS = "ABCD"


def user_prompt(trial):
    listings = "\n\n".join(listing_text(l, c) for l, c in zip(LABELS, trial["cars"]))
    return f"BUYER:\n{trial['buyer_profile']}\n\nLISTINGS:\n{listings}"


def messages(trial, with_target=False):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(trial)}]
    if with_target:
        msgs.append({"role": "assistant", "content": reference_response(trial)})
    return msgs


# ------------------------------------------------------- reference responses
NOTE_ISSUE = {
    "flood_damage": "Seller note reveals flood/water damage, even though the title is clean.",
    "odometer_unreliable": "Seller note says the odometer was replaced or is wrong, so the true mileage is unknown.",
    "washed_title": "Seller note reveals a past salvage/total-loss history that no longer shows on the title (title washing).",
    "scam_signals": "Seller asks for payment before any viewing via an untraceable method (gift cards, wire, Zelle, crypto, escrow) - classic scam signals.",
    "major_repair_needed": "Seller note admits a major repair is needed soon, which the buyer said they cannot afford.",
    "frame_damage": "Seller note reveals frame/structural damage.",
}
PRIORITY_NAME = {"price": "lower all-in price", "mileage": "lower mileage", "year": "newer model year", "mpg": "better fuel economy"}


def car_name(c):
    return f"{c['year']} {c['make']} {c['model']}"


def violation_text(code, vis, c, b):
    req, allin = b["req"], c["price"] + c["fees"]
    if code == "over_budget" and vis == "note":
        return (f"Required dealer fees ({money(c['fees'])}) bring the all-in price to {money(allin)}, "
                f"over the {money(b['budget'])} budget.")
    return {
        "over_budget": f"All-in price {money(allin)} is over the {money(b['budget'])} budget.",
        "branded_title": f"{c['title']} title; the buyer wants no title problems.",
        "not_automatic": "Manual transmission, but the buyer can only drive an automatic.",
        "no_awd": f"{c['drivetrain']} only; the buyer needs AWD/4WD for their conditions.",
        "too_few_seats": f"Only {c['seats']} seats; the buyer needs at least {req['seats'][1] if req['seats'] else '?'}.",
        "not_pickup": f"It's a {c['body'].lower()}, but the buyer needs a pickup bed for towing/hauling.",
        "too_many_miles": f"{c['mileage']:,} miles exceeds the buyer's {req['max_miles'][1]:,}-mile limit." if req["max_miles"] else "Too many miles.",
    }.get(code) or NOTE_ISSUE[code]


def reference_response(trial):
    b, cars = trial["buyer"], trial["cars"]
    checks, flags = {}, []
    for l, c in zip(LABELS, cars):
        if c["violations"]:
            texts = [violation_text(code, vis, c, b) for code, vis in c["violations"]]
            checks[l] = "Fails: " + " ".join(texts)
            for (code, vis), t in zip(c["violations"], texts):
                if vis == "note":
                    flags.append({"option": l, "issue": t})
        else:
            extra = f" (all-in {money(c['price'] + c['fees'])} including the {money(c['fees'])} fee)" if c["fees"] else ""
            note = f"; {BENIGN_SUMMARY_SHORT}" if c["note"] and not c["fees"] else ""
            checks[l] = f"Meets all requirements{extra}{note}."
    best = trial["best_choice"]
    feas = trial["feasible_options"]
    p1, p2 = (PRIORITY_NAME[p] for p in b["priorities"])
    if best == "NONE":
        reason = "Every listing fails at least one of the buyer's requirements, so none should be bought; keep looking."
        tradeoffs = "Waiting costs time, but each option here carries a deal-breaker."
    else:
        bc = cars[LABELS.index(best)]
        if len(feas) == 1:
            reason = f"{best} ({car_name(bc)}) is the only listing that meets every requirement."
            tradeoffs = "The other listings may look better on paper, but each has a deal-breaker."
        else:
            others = [o for o in feas if o != best]
            reason = (f"{best} ({car_name(bc)}) passes every requirement and gives the best balance of the buyer's "
                      f"priorities: {p1} first, then {p2}.")
            tradeoffs = " ".join(_compare(bc, cars[LABELS.index(o)], o) for o in others)
    return json.dumps({"option_checks": checks, "red_flags": flags, "recommendation": best,
                       "reason": reason, "tradeoffs": tradeoffs}, ensure_ascii=False)


BENIGN_SUMMARY_SHORT = "seller note mentions only minor, routine items"


def _compare(best, other, label):
    diffs = []
    dp = (other["price"] + other["fees"]) - (best["price"] + best["fees"])
    if dp:
        diffs.append(f"costs {money(abs(dp))} {'more' if dp > 0 else 'less'}")
    dm = other["mileage"] - best["mileage"]
    if dm:
        diffs.append(f"has {abs(dm):,} {'more' if dm > 0 else 'fewer'} miles")
    dy = other["year"] - best["year"]
    if dy:
        diffs.append(f"is {abs(dy)} year{'s' if abs(dy) > 1 else ''} {'newer' if dy > 0 else 'older'}")
    dg = other["mpg"] - best["mpg"]
    if dg:
        diffs.append(f"gets {abs(dg)} mpg {'more' if dg > 0 else 'less'}")
    return f"Choosing {label} instead {', '.join(diffs) if diffs else 'is nearly identical'}."


# ---------------------------------------------------------------- parsing
def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def parse_output(text):
    """Return (recommendation, flagged_options, parsed_dict_or_None)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if obj is not None:
        # be lenient about structure: the base model sometimes nests keys inside option_checks
        rec = str(_find_key(obj, "recommendation") or "").strip().upper()
        rec = "NONE" if rec.startswith("NONE") else rec[:1]
        raw_flags = _find_key(obj, "red_flags") or []
        flags = {str(f.get("option", "")).strip().upper()[:1] for f in raw_flags if isinstance(f, dict)}
        return (rec if rec in ("A", "B", "C", "D", "NONE") else "INVALID"), sorted(flags), obj
    m = re.search(r'recommendation"?\s*[:=]\s*"?(A|B|C|D|NONE)', text, flags=re.I)
    return (m.group(1).upper() if m else "INVALID"), [], None
