"""Generate the AI-Assisted Used Car Selection dataset (study + training splits).

Each trial = one buyer profile + four used-car listings (A-D). The buyer states
hard requirements (some explicitly, some implicitly in their story) and two soft
priorities in words. Some listings hide a deal-breaker in the free-text seller
note (flood damage, washed title, scam signals, required fees that break the
budget, ...). The correct answer is any listing that meets every requirement and
is not dominated on the buyer's two priorities, or NONE if no listing qualifies.

Stdlib only.  Usage:  python gen_dataset.py --out data
"""
import argparse
import csv
import json
import random
from pathlib import Path

CURRENT_YEAR = 2026

# ---------------------------------------------------------------- car catalog
# (make, model, body, seats, drivetrains, manual_available, fuel, mpg, new_price)
CATALOG = [
    ("Toyota", "Corolla", "Sedan", 5, ["FWD"], False, "Gas", 34, 23000),
    ("Honda", "Civic", "Sedan", 5, ["FWD"], True, "Gas", 35, 25000),
    ("Hyundai", "Elantra", "Sedan", 5, ["FWD"], False, "Gas", 36, 22000),
    ("Toyota", "Camry", "Sedan", 5, ["FWD", "AWD"], False, "Gas", 32, 28000),
    ("Honda", "Accord", "Sedan", 5, ["FWD"], False, "Gas", 33, 29000),
    ("Nissan", "Altima", "Sedan", 5, ["FWD", "AWD"], False, "Gas", 31, 26000),
    ("Kia", "K5", "Sedan", 5, ["FWD", "AWD"], False, "Gas", 31, 27000),
    ("Mazda", "Mazda3", "Sedan", 5, ["FWD", "AWD"], True, "Gas", 31, 24000),
    ("Subaru", "Impreza", "Hatchback", 5, ["AWD"], True, "Gas", 30, 23000),
    ("Subaru", "Legacy", "Sedan", 5, ["AWD"], False, "Gas", 29, 26000),
    ("Volkswagen", "Golf", "Hatchback", 5, ["FWD"], True, "Gas", 32, 24000),
    ("Toyota", "Prius", "Hatchback", 5, ["FWD"], False, "Hybrid", 52, 28000),
    ("Honda", "Fit", "Hatchback", 5, ["FWD"], True, "Gas", 36, 18000),
    ("Honda", "CR-V", "SUV", 5, ["FWD", "AWD"], False, "Gas", 30, 31000),
    ("Toyota", "RAV4", "SUV", 5, ["FWD", "AWD"], False, "Gas", 30, 31000),
    ("Toyota", "RAV4 Hybrid", "SUV", 5, ["AWD"], False, "Hybrid", 40, 34000),
    ("Mazda", "CX-5", "SUV", 5, ["FWD", "AWD"], False, "Gas", 28, 29000),
    ("Subaru", "Forester", "SUV", 5, ["AWD"], False, "Gas", 29, 29000),
    ("Subaru", "Crosstrek", "SUV", 5, ["AWD"], True, "Gas", 30, 26000),
    ("Hyundai", "Tucson", "SUV", 5, ["FWD", "AWD"], False, "Gas", 28, 28000),
    ("Ford", "Escape", "SUV", 5, ["FWD", "AWD"], False, "Gas", 30, 28000),
    ("Subaru", "Outback", "Wagon", 5, ["AWD"], False, "Gas", 28, 30000),
    ("Honda", "Pilot", "SUV", 8, ["FWD", "AWD"], False, "Gas", 22, 40000),
    ("Toyota", "Highlander", "SUV", 8, ["FWD", "AWD"], False, "Gas", 24, 40000),
    ("Kia", "Telluride", "SUV", 8, ["FWD", "AWD"], False, "Gas", 23, 38000),
    ("Subaru", "Ascent", "SUV", 8, ["AWD"], False, "Gas", 22, 37000),
    ("Mazda", "CX-9", "SUV", 7, ["FWD", "AWD"], False, "Gas", 23, 37000),
    ("Ford", "Explorer", "SUV", 7, ["RWD", "AWD"], False, "Gas", 24, 39000),
    ("Honda", "Odyssey", "Minivan", 8, ["FWD"], False, "Gas", 22, 40000),
    ("Toyota", "Sienna", "Minivan", 8, ["FWD", "AWD"], False, "Hybrid", 36, 41000),
    ("Chrysler", "Pacifica", "Minivan", 7, ["FWD", "AWD"], False, "Gas", 22, 39000),
    ("Kia", "Carnival", "Minivan", 8, ["FWD"], False, "Gas", 22, 36000),
    ("Toyota", "Tacoma", "Pickup", 5, ["RWD", "4WD"], True, "Gas", 20, 35000),
    ("Ford", "F-150", "Pickup", 6, ["RWD", "4WD"], False, "Gas", 20, 42000),
    ("Chevrolet", "Silverado 1500", "Pickup", 6, ["RWD", "4WD"], False, "Gas", 19, 42000),
    ("Honda", "Ridgeline", "Pickup", 5, ["AWD"], False, "Gas", 21, 40000),
    ("Mazda", "MX-5 Miata", "Convertible", 2, ["RWD"], True, "Gas", 30, 30000),
    ("Ford", "Mustang", "Coupe", 4, ["RWD"], True, "Gas", 22, 32000),
]
FOUR_WHEEL = {"AWD", "4WD"}

# Real US model-year availability (default 2012-2024) so no listing shows a car that never existed.
YEAR_RANGE = {
    "K5": (2021, 2024), "Golf": (2012, 2021), "Fit": (2012, 2020), "RAV4 Hybrid": (2016, 2024),
    "CX-5": (2013, 2024), "Crosstrek": (2013, 2024), "Telluride": (2020, 2024), "Ascent": (2019, 2024),
    "CX-9": (2012, 2023), "Sienna": (2021, 2024), "Pacifica": (2017, 2024), "Carnival": (2022, 2024),
    "Ridgeline": (2017, 2024),
}
# Drivetrains that only exist from a given model year.
DRIVE_MIN_YEAR = {("Camry", "AWD"): 2020, ("Altima", "AWD"): 2019, ("Mazda3", "AWD"): 2019}

STATES = ["MI"] * 6 + ["OH", "IN", "IL", "WI", "TX", "FL", "PA", "MN"]
CITIES = {"MI": ["Ann Arbor", "Detroit", "Grand Rapids", "Lansing", "Flint", "Kalamazoo", "Marquette"],
          "OH": ["Toledo", "Columbus"], "IN": ["Fort Wayne"], "IL": ["Chicago"], "WI": ["Milwaukee"],
          "TX": ["Houston"], "FL": ["Tampa"], "PA": ["Erie"], "MN": ["Duluth"]}

# ------------------------------------------------------- seller-note templates
# Templates marked held_out=True appear ONLY in the study split, so evaluation
# also measures whether the model generalises to unseen phrasings.
RED_FLAGS = {
    "flood_damage": [
        ("Bought it at auction out of Florida after last year's hurricane. Had some water in the cabin, but I replaced the carpet and it runs great.", False),
        ("Carpets and seats were replaced after water got in during the Houston flooding. Title is still clean since we never filed with insurance.", False),
        ("Was parked in a garage that flooded for a few days. There's a little rust under the seats and a musty smell, priced to sell.", False),
        ("Interior was professionally dried out and sanitized after the storm surge. You'd never know it was underwater.", True),
        ("Some silt behind the dash panels from when the river crested, but the electronics all work right now.", True),
    ],
    "odometer_unreliable": [
        ("The instrument cluster was swapped at some point, so the miles shown aren't the true miles.", False),
        ("History report says TMU (true mileage unknown). Engine sounds strong though.", False),
        ("Previous owner replaced the digital dash. Actual mileage is probably higher, priced accordingly.", True),
        ("Odometer stopped working for a while a couple of years back, so the reading is a bit off from real use.", True),
    ],
    "washed_title": [
        ("The title was branded salvage in Texas, but it's been retitled clean here in Michigan, so no issues.", False),
        ("Previous owner rebuilt it after an insurance total loss. The new title came back clean.", False),
        ("It was declared a total loss in another state years ago, but the current title doesn't show that anymore.", True),
        ("Insurance company paid out on it after a hail and collision claim; it was repaired and re-registered with a regular title.", True),
    ],
    "scam_signals": [
        ("I'm overseas on deployment and the car is with a shipping company. Send a deposit with gift cards and they'll deliver it to you.", False),
        ("Selling for my cousin who moved away. I need full payment by Zelle before you see it. No test drives, serious buyers only.", False),
        ("Priced low because it has to be gone this week. I can't meet in person; pay by wire transfer and I'll mail the title.", False),
        ("I'm not able to show the car, but an escrow agent will handle everything once you send payment. Title will follow in the mail.", True),
        ("Text only, no calls. Car is in storage out of state; a $500 refundable holding deposit via crypto reserves it.", True),
    ],
    "major_repair_needed": [
        ("The transmission slips a little between 2nd and 3rd, and the shop quoted about $4,000 to rebuild it. Otherwise great.", False),
        ("It needs a head gasket soon (it's using coolant). Price reflects that.", False),
        ("Check-engine light is on. Mechanic said catalytic converter and O2 sensors, roughly $2,800. Drives fine for now.", False),
        ("Makes a grinding noise in the rear end, a shop said the differential is going. Still drives to work every day.", True),
        ("Overheats if you idle too long, probably the water pump or worse. Haven't had time to fix it.", True),
    ],
    "frame_damage": [
        ("Rear-ended last winter and the frame was straightened at a local body shop. Drives straight.", False),
        ("Has some frame rot. It failed inspection in Ohio but should be fine in Michigan.", False),
        ("The unibody was pulled back into spec after a side impact. Doors close fine now.", True),
        ("Some rust-through on the frame rails, typical for a Michigan car. Body itself looks great.", True),
    ],
}
# Dealer fee notes are filled in programmatically; {fees} = list of fee phrases.
FEE_TEMPLATES = [
    ("Advertised price does not include a mandatory {f1} reconditioning package and a {f2} documentation fee.", False),
    ("Plus {f1} dealer prep and {f2} admin fee, required on every vehicle.", False),
    ("Price shown is before our {f1} protection package (pre-installed, non-removable) and {f2} doc fee.", True),
]
BENIGN_FEE_TEMPLATES = [
    ("Plus a {f1} documentation fee.", False),
    ("A {f1} doc fee applies at signing.", True),
]
BENIGN = [
    ("One owner, all service records available. New tires last fall.", False),
    ("Some curb rash on the wheels and a small door ding on the passenger side.", False),
    ("The airbag recall was completed at the dealer in 2024, paperwork included.", False),
    ("Smells a little like dog. I'll get it detailed before the sale.", False),
    ("Check-engine light came on last month. It turned out to be a loose gas cap, replaced and cleared.", False),
    ("Faded paint on the roof but mechanically solid. Just had the brakes done.", False),
    ("Non-smoker, garage kept, two keys included.", False),
    ("Small chip in the windshield; insurance will cover the repair.", False),
    ("Includes a set of winter tires on steel wheels.", False),
    ("Clean history report, no accidents reported. Happy to let your mechanic inspect it.", False),
    ("Small scratches on the rear bumper from parking.", False),
    ("Timing belt and water pump replaced at 90k, receipts in the glovebox.", False),
    ("Minor fender-bender in 2021, cosmetic only, repaired at the dealer and listed on the history report.", True),
    ("A/C needed a recharge last summer, blows cold now. Battery is two years old.", True),
    ("Cosmetic hail dimples on the hood, no structural damage. Priced a bit below book because of it.", True),
    ("Had the transmission fluid flushed and the brakes inspected at 60k, everything checked out.", True),
]
BENIGN_SUMMARY = "the seller note only mentions minor, routine items"

# ------------------------------------------------------- buyer text templates
CATEGORIES = ["First-time buyer", "Daily commuter", "Family vehicle", "Cold-weather driver", "Lifestyle / outdoor use"]
OPENERS = {
    "First-time buyer": [
        "I'm a college senior buying my first car with money I saved from a part-time job.",
        "This will be my first car. I just started an entry-level job and need something to get around.",
        "I'm a grad student and have never owned a car before. I mostly need it for groceries and getting to campus.",
    ],
    "Daily commuter": [
        "I commute about 45 miles each way on the highway, five days a week.",
        "I drive to work every weekday through city traffic and want a sensible car for it.",
        "My new job is across town, so I'll be putting a lot of miles on whatever I buy.",
    ],
    "Family vehicle": [
        "We're a growing family and need a dependable car for school runs and weekend trips.",
        "I'm shopping for our main family car. It does school drop-offs, sports practice and grocery runs.",
        "Our old car finally died and we need something for the whole family.",
    ],
    "Cold-weather driver": [
        "I live in northern Michigan where winters are long and snowy.",
        "I work early shifts, so I'm often driving before the plows have been out.",
        "Winter driving is my biggest worry; we get lake-effect snow most of the season.",
    ],
    "Lifestyle / outdoor use": [
        "I spend most weekends outdoors, camping and fishing up north.",
        "My hobbies take me off the beaten path, and I haul a lot of gear.",
        "I'm buying a vehicle for weekend adventures and home-improvement projects.",
    ],
}
BUDGET_TPL = [
    "My all-in budget is ${b:,}, and that has to cover any dealer fees.",
    "I can spend at most ${b:,} total, including any required dealer fees.",
    "The absolute most I can pay is ${b:,} out the door (fees included).",
]
SAFETY_TPL = [
    "Like anyone, I don't want a car with hidden damage or title problems, a seller who seems like a scam, or a big repair bill right after I buy it.",
    "I need the car to be honestly represented: no flood or frame damage, no title or mileage issues, no sketchy payment setups, and nothing that needs a major repair right away.",
    "I can't afford surprises, so no hidden damage, no questionable title or mileage history, no scam-like sellers, and no big repairs looming.",
]
EXPLICIT = {
    "automatic": ["It has to be an automatic.", "Automatic transmission only."],
    "awd": ["AWD or 4WD is a must.", "It needs all-wheel or four-wheel drive."],
    "max_miles": ["Nothing over {m:,} miles on the odometer.", "I won't consider anything with more than {m:,} miles."],
    "seats": ["It needs at least {s} seats.", "We need seating for at least {s}."],
    "body": ["It has to be a pickup truck.", "I only want a pickup."],
}
IMPLICIT = {
    "automatic": ["I never learned to drive stick, and neither did my partner who will share the car.",
                  "I only know how to drive cars with no clutch pedal."],
    "awd": ["My driveway is a steep hill that ices over all winter, and front-wheel-drive cars have gotten stuck on it every year.",
            "Our road is unpaved and the county plows it last, so I've been stuck in two-wheel-drive cars before and can't risk it again."],
    "seats": {6: ["It's me, my spouse, our three kids, and my mother usually rides with us on weekends."],
              7: ["We have three kids and my parents ride along with us most weekends.",
                  "It's the two of us, our four kids, and we regularly drive my father-in-law to appointments."],
              8: ["We have four kids and we carpool two of their friends to school every day."]},
    "body": ["I need to tow a 5,000-lb camper and throw muddy gear in an open bed.",
             "I'm hauling lumber and a dirt bike most weekends, so I need an open cargo bed."],
}
PRIORITY_WORDS = {
    "price": ["keeping the cost as low as possible", "spending as little as I can"],
    "mileage": ["low miles on the odometer", "as few miles as possible"],
    "year": ["getting something newer", "a more recent model year"],
    "mpg": ["good fuel economy", "saving money on gas"],
}
PRIORITY_TPL = [
    "Within those limits, what I care about most is {p1}, and after that {p2}.",
    "Beyond the deal-breakers, my top priority is {p1}; second is {p2}.",
    "If several cars work, I'd lean toward {p1} first and {p2} second.",
]
ASK = "Which listing would you recommend? Choose A, B, C, D, or None if no listing meets my needs."

# ---------------------------------------------------------------- generation
def money(x):
    return f"${x:,}"


class Gen:
    def __init__(self, seed, held_out_ok):
        self.rng = random.Random(seed)
        self.held_out_ok = held_out_ok  # study split may use held-out phrasings

    def pick_tpl(self, items):
        pool = [t for t, h in items if self.held_out_ok or not h]
        return self.rng.choice(pool)

    # ---- buyer
    def make_buyer(self, cat):
        r = self.rng
        req = {"automatic": None, "awd": None, "max_miles": None, "seats": None, "body": None}
        if cat == "First-time buyer":
            budget = r.randrange(7000, 15001, 500)
        elif cat == "Daily commuter":
            budget = r.randrange(11000, 24001, 500)
        elif cat == "Family vehicle":
            budget = r.randrange(16000, 36001, 500)
        elif cat == "Cold-weather driver":
            budget = r.randrange(11000, 30001, 500)
        else:
            budget = r.randrange(15000, 36001, 500)
        mode = lambda: "implicit" if r.random() < 0.5 else "explicit"
        if r.random() < (0.45 if cat != "First-time buyer" else 0.7):
            req["automatic"] = mode()
        if cat == "Cold-weather driver" or r.random() < 0.15:
            req["awd"] = mode()
        if r.random() < 0.45:
            req["max_miles"] = ("explicit", r.choice([60000, 80000, 90000, 100000, 120000]))
        if cat == "Family vehicle":
            req["seats"] = (mode(), r.choice([6, 7, 7, 8]))
        if cat == "Lifestyle / outdoor use" and r.random() < 0.6:
            req["body"] = mode()
        prios = r.sample(["price", "mileage", "year", "mpg"], 2)
        if cat == "Daily commuter" and "mpg" not in prios and r.random() < 0.6:
            prios[1] = "mpg"
        return {"category": cat, "budget": budget, "req": req, "priorities": prios}

    def buyer_text(self, b):
        r, req = self.rng, b["req"]
        parts = [r.choice(OPENERS[b["category"]]), r.choice(BUDGET_TPL).format(b=b["budget"])]
        reqs = []
        for k in ["automatic", "awd", "seats", "body", "max_miles"]:
            v = req[k]
            if v is None:
                continue
            m = v[0] if isinstance(v, tuple) else v
            if k == "max_miles":
                reqs.append(r.choice(EXPLICIT[k]).format(m=v[1]))
            elif k == "seats":
                reqs.append(r.choice(EXPLICIT[k]).format(s=v[1]) if m == "explicit" else r.choice(IMPLICIT["seats"][v[1]]))
            else:
                reqs.append(r.choice(EXPLICIT[k] if m == "explicit" else IMPLICIT[k]))
        r.shuffle(reqs)
        parts += reqs
        parts.append(r.choice(SAFETY_TPL))
        p1, p2 = (r.choice(PRIORITY_WORDS[p]) for p in b["priorities"])
        parts.append(r.choice(PRIORITY_TPL).format(p1=p1, p2=p2))
        parts.append(ASK)
        return " ".join(parts)

    # ---- cars
    def random_car(self, pool=None):
        r = self.rng
        make, model, body, seats, drives, manual_ok, fuel, mpg, new = r.choice(pool or CATALOG)
        year = r.randint(*YEAR_RANGE.get(model, (2012, 2024)))
        drives = [d for d in drives if year >= DRIVE_MIN_YEAR.get((model, d), 0)]
        age = CURRENT_YEAR - year
        miles = int(max(4000, age * r.uniform(7000, 17000)) // 500 * 500)
        value = new * (0.86 ** age) * (1 - 0.12 * (miles - 12000 * age) / 50000) * r.uniform(0.9, 1.1)
        price = max(2500, int(round(value / 500.0)) * 500 - 100)
        state = r.choice(STATES)
        return {
            "year": year, "make": make, "model": model, "body": body, "seats": seats,
            "drivetrain": r.choice(drives),  # filtered by model year above
            "transmission": "Manual" if manual_ok and r.random() < 0.3 else "Automatic",
            "fuel": fuel, "mpg": mpg + r.choice([-1, 0, 0, 1]), "price": price, "mileage": miles,
            "title": "Clean", "seller": "Dealer" if r.random() < 0.55 else "Private",
            "location": f"{r.choice(CITIES[state])}, {state}", "note": "", "fees": 0,
            "violations": [],  # list of (code, visibility) where visibility in explicit/implicit/note
        }

    def struct_violations(self, car, b):
        """Violations visible from structured fields + buyer requirements (no notes)."""
        req, out = b["req"], []
        if car["price"] + car["fees"] > b["budget"]:
            out.append(("over_budget", "explicit"))
        if car["title"] != "Clean":
            out.append(("branded_title", "explicit"))
        for k, code in [("automatic", "not_automatic"), ("awd", "no_awd"), ("seats", "too_few_seats"), ("body", "not_pickup")]:
            v = req[k]
            if v is None:
                continue
            mode = v[0] if isinstance(v, tuple) else v
            bad = ((k == "automatic" and car["transmission"] != "Automatic") or
                   (k == "awd" and car["drivetrain"] not in FOUR_WHEEL) or
                   (k == "seats" and car["seats"] < v[1]) or
                   (k == "body" and car["body"] != "Pickup"))
            if bad:
                out.append((code, mode))
        if req["max_miles"] and car["mileage"] > req["max_miles"][1]:
            out.append(("too_many_miles", "explicit"))
        return out

    def body_pool(self, b):
        cat = b["category"]
        if cat == "Family vehicle":
            return [c for c in CATALOG if c[3] >= 5 and c[2] in ("SUV", "Minivan", "Wagon", "Sedan")]
        if cat == "Lifestyle / outdoor use":
            return [c for c in CATALOG if c[2] in ("Pickup", "SUV", "Wagon")]
        if cat == "Cold-weather driver":
            return [c for c in CATALOG if c[2] not in ("Convertible", "Coupe")]
        return CATALOG

    def sample_car(self, b, want, tries=4000):
        """Sample a car whose structured violation set equals `want` (list of codes)."""
        pool = self.body_pool(b)
        for i in range(tries):
            car = self.random_car(pool if i < tries // 2 else None)
            if car["seller"] == "Dealer" and self.rng.random() < 0.35:
                car["fees"] = self.rng.choice([199, 299, 399, 499, 699])
            if "branded_title" in want:
                car["title"] = self.rng.choice(["Rebuilt", "Salvage"])
            codes = sorted(c for c, _ in self.struct_violations(car, b))
            if codes == sorted(want):
                return car
        return None

    def utility(self, car, b):
        """Raw priority values (higher = better) used for ranking."""
        vals = {"price": -(car["price"] + car["fees"]), "mileage": -car["mileage"], "year": car["year"], "mpg": car["mpg"]}
        return [vals[p] for p in b["priorities"]]

    def available_hidden(self, b):
        opts = [("note", k) for k in list(RED_FLAGS) + ["hidden_fees"]]
        for k, code in [("automatic", "not_automatic"), ("awd", "no_awd"), ("seats", "too_few_seats"), ("body", "not_pickup")]:
            v = b["req"][k]
            if v is not None and (v[0] if isinstance(v, tuple) else v) == "implicit":
                opts.append(("implicit", code))
        return opts

    def explicit_codes(self, b):
        codes = ["over_budget", "branded_title"]
        for k, code in [("automatic", "not_automatic"), ("awd", "no_awd"), ("seats", "too_few_seats"), ("body", "not_pickup")]:
            v = b["req"][k]
            if v is not None and (v[0] if isinstance(v, tuple) else v) == "explicit":
                codes.append(code)
        if b["req"]["max_miles"]:
            codes.append("too_many_miles")
        return codes

    def make_hidden_car(self, b, kind, code):
        r = self.rng
        if kind == "implicit":
            car = self.sample_car(b, [code])
            if car:
                car["note"] = self.benign_note(car, b)
                car["violations"] = self.struct_violations(car, b)
            return car
        if code == "hidden_fees":
            for _ in range(200):
                car = self.sample_car(b, [])
                if car is None:
                    return None
                car["seller"], car["fees"] = "Dealer", 0
                gap = b["budget"] - car["price"]
                if 0 <= gap <= 2200:
                    total = gap + r.randrange(300, 2600, 50)
                    f1 = int(total * r.uniform(0.55, 0.8)) // 5 * 5
                    f2 = total - f1
                    car["note"] = self.pick_tpl(FEE_TEMPLATES).format(f1=money(f1), f2=money(f2))
                    car["fees"] = total
                    car["violations"] = [("over_budget", "note")]
                    return car
            return None
        car = self.sample_car(b, [])
        if car is None:
            return None
        if code == "scam_signals":
            car["seller"], car["fees"] = "Private", 0
        car["note"] = self.pick_tpl(RED_FLAGS[code])
        car["violations"] = [(code, "note")]
        return car

    def benign_note(self, car, b):
        r = self.rng
        if car["fees"]:
            return self.pick_tpl(BENIGN_FEE_TEMPLATES).format(f1=money(car["fees"]))
        return self.pick_tpl(BENIGN) if r.random() < 0.85 else ""

    def make_ok_car(self, b):
        car = self.sample_car(b, [])
        if car:
            car["note"] = self.benign_note(car, b)
        return car

    def make_explicit_bad(self, b):
        codes = self.explicit_codes(b)
        weights = [3 if c == "over_budget" else 1 if c == "branded_title" else 2 for c in codes]
        code = self.rng.choices(codes, weights)[0]
        car = self.sample_car(b, [code])
        if car:
            car["note"] = self.benign_note(car, b)
            car["violations"] = self.struct_violations(car, b)
        return car

    def generate_trial(self, cat, ttype):
        r = self.rng
        for _ in range(300):
            b = self.make_buyer(cat)
            n_ok = {"clear": 1, "trap": 1, "tradeoff": r.choice([2, 2, 3]), "none": 0}[ttype]
            n_hidden = {"clear": 0, "trap": r.choice([1, 1, 2]), "tradeoff": 1 if r.random() < 0.6 else 0,
                        "none": r.choice([1, 2, 2])}[ttype]
            n_hidden = min(n_hidden, 4 - n_ok)
            cars, hidden = [], []
            ok = [self.make_ok_car(b) for _ in range(n_ok)]
            for _ in range(n_hidden):
                opts = self.available_hidden(b)
                implicit = [o for o in opts if o[0] == "implicit"]
                kind, code = r.choice(implicit) if implicit and r.random() < 0.45 else r.choice([o for o in opts if o[0] == "note"])
                hidden.append(self.make_hidden_car(b, kind, code))
            bad = [self.make_explicit_bad(b) for _ in range(4 - n_ok - n_hidden)]
            cars = ok + hidden + bad
            if any(c is None for c in cars):
                continue
            # The hidden-problem car should be a tempting decoy: it beats every
            # qualifying car on the buyer's top priority.
            if hidden and ok:
                best_ok = max(self.utility(c, b)[0] for c in ok)
                if not all(self.utility(h, b)[0] > best_ok for h in hidden):
                    continue
            # avoid exact duplicate listings
            if len({(c["year"], c["make"], c["model"], c["price"]) for c in cars}) < 4:
                continue
            r.shuffle(cars)
            return b, cars
        raise RuntimeError(f"could not generate {cat}/{ttype}")


def answer_key(b, cars, gen):
    labels = "ABCD"
    feasible = [i for i, c in enumerate(cars) if not c["violations"]]
    if not feasible:
        return "NONE", ["NONE"], feasible
    utils = {i: gen.utility(cars[i], b) for i in feasible}

    def dominated(i):
        return any(all(utils[j][k] >= utils[i][k] for k in range(2)) and utils[j] != utils[i] for j in feasible if j != i)

    acceptable = [i for i in feasible if not dominated(i)]
    # best: weighted min-max score over the four listings (0.6 / 0.4)
    allu = [gen.utility(c, b) for c in cars]
    def norm(k, v):
        lo, hi = min(u[k] for u in allu), max(u[k] for u in allu)
        return 0.5 if hi == lo else (v - lo) / (hi - lo)
    score = {i: 0.6 * norm(0, utils[i][0]) + 0.4 * norm(1, utils[i][1]) for i in feasible}
    best = max(feasible, key=lambda i: score[i])
    return labels[best], [labels[i] for i in acceptable], feasible


def listing_text(label, c):
    lines = [f"Option {label}: {c['year']} {c['make']} {c['model']}",
             f"  Price: {money(c['price'])} | Mileage: {c['mileage']:,} mi | {c['body']}, {c['seats']} seats | "
             f"{c['drivetrain']} | {c['transmission']} | {c['fuel']}, {c['mpg']} mpg combined",
             f"  Title: {c['title']} | Seller: {c['seller']} ({c['location']})"]
    lines.append(f"  Seller notes: \"{c['note']}\"" if c["note"] else "  Seller notes: (none)")
    return "\n".join(lines)


def build_split(n_per_cell, seed, held_out_ok, prefix):
    gen = Gen(seed, held_out_ok)
    # per category: clear / trap / tradeoff / none  -> scaled from 30/70/70/30 of 200
    mix = {"clear": 6, "trap": 14, "tradeoff": 14, "none": 6}
    trials = []
    k = 0
    for cat in CATEGORIES:
        for ttype, n in mix.items():
            for _ in range(n * n_per_cell):
                k += 1
                b, cars = gen.generate_trial(cat, ttype)
                best, acceptable, feasible = answer_key(b, cars, gen)
                trials.append(dict(
                    trial_id=f"{prefix}-{k:04d}" if prefix == "TRN" else f"{prefix}-{k:03d}",
                    category=cat, trial_type=ttype, buyer=b, buyer_profile=gen.buyer_text(b),
                    cars=cars, best_choice=best, acceptable_choices=acceptable,
                    feasible_options=["ABCD"[i] for i in feasible]))
    gen.rng.shuffle(trials)
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--study-seed", type=int, default=594)
    ap.add_argument("--train-seed", type=int, default=20261014)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    study = build_split(1, args.study_seed, True, "CAR")          # 200 trials
    train = build_split(9, args.train_seed, False, "TRN")         # 1,800 trials
    # drop any training trial whose buyer text exactly matches a study trial
    study_texts = {t["buyer_profile"] for t in study}
    train = [t for t in train if t["buyer_profile"] not in study_texts]
    n_val = 180
    splits = {"study": study, "train": train[n_val:], "val": train[:n_val]}
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for t in rows:
                f.write(json.dumps(t) + "\n")
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
