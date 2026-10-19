"""Hand-written, natural-language text used ONLY in the study split (iteration 2).

None of these sentences appear in training data. They are written the way real
private sellers and dealers write: problems are buried inside otherwise positive
notes, several items are mixed together, fees sit in the fine print, and some
harmless notes contain alarming-sounding words. Buyer needs are implied in
everyday terms that take a moment of thought.
"""

# ---- red flags: each must be a clear deal-breaker to a careful human reader
NATURAL_RED_FLAGS = {
    "flood_damage": [
        "Great little commuter, new brakes and tires in March. Full disclosure, my basement garage took on about a foot of water during the August storms, but I had it detailed and it starts every time.",
        "Got this one cheap from a guy in Louisiana after Hurricane season. Interior was redone (new carpet, seat foam) so it looks brand new inside. Runs strong!",
        "Clean inside and out, smells fresh. We had it sitting in the lot when the creek came up last spring - water got to about the door sills. Mechanic gave it a once-over and said it's fine.",
        "Highway miles, cold AC, new battery. Speakers crackle a bit and there's some fog in the headlights since it sat in standing water for a couple days, otherwise perfect.",
    ],
    "odometer_unreliable": [
        "Second owner, runs and drives great. The gauge cluster died in 2022 and I put in a used one from a junkyard, so the miles on the dash aren't the car's real miles - probably around 40k more.",
        "Daily driver, never let me down. FYI the odometer quit for about a year before I got it fixed, so don't go by the number on the dash too much.",
        "Love this car, just need something bigger. Carfax shows a mileage inconsistency in 2020 (dealer said it was a clerical thing, who knows). Priced to move.",
        "New tires, new wipers, oil changed every 5k. The digital dash was swapped by the last owner and I honestly don't know what the real mileage is.",
    ],
    "washed_title": [
        "Title is clean and in my name. It did get totaled by insurance in Florida back in 2021 after a fender bender and was fixed up and re-registered here, but you'd never know.",
        "Bought it from an auction - was a buy-back after a crash, then a shop in Indiana rebuilt it and it got a fresh title. Drives straight, no lights on the dash.",
        "My uncle bought this at a salvage auction and his body shop fixed it up. Title transferred clean when he registered it in Michigan.",
        "Good car, but I'll be honest: Carfax shows it was reported as a total loss in another state. The title I have doesn't say salvage though, so it's legally clean.",
    ],
    "scam_signals": [
        "I relocated for work so the car is at my parents' place in another state. If you send a $400 deposit by Venmo they'll hold it and ship it to you with the title. Sorry, no in-person viewings.",
        "Priced well below KBB because I need it gone before I leave the country Friday. eBay Motors protection - you pay through their agent and get 5 days to inspect after delivery.",
        "Please email only, I'm hard of hearing. Car is perfect. Will ship anywhere once payment clears via cashier's check sent to my shipping agent.",
        "Military family moving overseas. Must sell fast. Payment through Google Pay gift cards or wire, and I'll have the car delivered to your driveway within 3 days.",
    ],
    "major_repair_needed": [
        "Great on gas, new tires. It does shudder when shifting from 2nd to 3rd once it warms up - shop quoted me a transmission rebuild, which I don't have time for.",
        "Starts every morning, heat works great. It's been burning a quart of oil every 500 miles and the mechanic says the piston rings are going. Still drives fine for now.",
        "Selling because I got a company car. Heads up, there's a check engine light for a misfire and compression on cylinder 3 is low, so it probably needs engine work.",
        "Interior is spotless. AWD system makes a loud whine on turns and the dealer said the rear differential and transfer case both need replacing, about $3,500.",
    ],
    "frame_damage": [
        "Body is straight and paint looks great. Was in a moderate accident last year and the shop had to pull the frame back into spec, plus a new quarter panel.",
        "Typical Michigan car underneath - the frame rails have rusted through in a couple spots near the rear wheels but the body is solid.",
        "Rear end was hit pretty hard in a parking lot. Unibody was repaired at an independent shop (not on Carfax since we paid cash). Trunk closes fine now.",
        "Runs great, cold AC. Inspector in Ohio flagged structural rust on the subframe mounts so I'm selling it here instead.",
    ],
}

# Dealer fine print. {f1}/{f2} are filled so price + fees exceeds the budget.
NATURAL_FEE_NOTES = [
    "Financing available for all credit types! Ask about our extended warranty. *Advertised price excludes {f1} market adjustment and {f2} dealer documentary fee, which apply to all purchases.",
    "One owner, local trade, just serviced. Internet price valid with dealer-arranged financing only; cash buyers add {f1}. A {f2} doc fee applies.",
    "Certified inspection completed! Every vehicle includes our Peace of Mind package ({f1}) - nitrogen tires, VIN etching, and paint protection - plus a {f2} processing fee.",
    "Low miles, sharp looking. Price shown after all rebates. Excludes {f1} reconditioning charge and {f2} electronic filing fee.",
]

# Harmless notes that use alarming-sounding words. A careful human sees they are fine.
NATURAL_BENIGN = [
    "I rebuilt the front suspension myself last year - new struts, control arms and sway bar links. Drives tight and quiet.",
    "Odometer reads exactly what's on the listing, mostly highway miles from my commute. Oil changes every 5k, receipts included.",
    "Not a scam, I promise! Happy to meet at the police station parking lot, bring your mechanic, and we can do the title transfer at the Secretary of State.",
    "Water pump and timing belt were replaced at 95k as recommended. Next big service isn't due for a long time.",
    "Had a minor parking-lot scrape in 2022, fixed at the dealer with OEM parts. It's on the Carfax, no structural damage.",
    "Title in hand, clean, I'm the second owner. Car has always been garaged.",
    "Some salt rust on the exhaust heat shields (normal for Michigan) but the frame and underbody were undercoated every fall and look solid.",
    "Check engine light came on once - turned out to be a loose gas cap. Had it scanned at AutoZone, no codes since.",
    "Ate a pothole and needed a new tire and rim last winter. Alignment done right after, drives straight.",
    "Selling because we had twins and need a bigger car. Smells a bit like a new-car air freshener because I just had it detailed.",
    "The flood lights on the roof rack were added by me for camping; they're wired to a switch and can be removed.",
    "Seller is a mechanic. Brakes, rotors, battery all done this year. Happy to show you the paperwork.",
]
NATURAL_BENIGN_FEE = [
    "Clean Carfax, one owner. Only fee is a {f1} state-capped doc fee, no add-ons or packages.",
    "Fresh oil change and inspection. We charge a {f1} documentation fee and nothing else - no hidden add-ons.",
]

# ---- implicit needs in everyday language (same requirement, different wording)
NATURAL_IMPLICIT = {
    "automatic": [
        "My son is borrowing it to learn to drive this summer, and his driving school only teaches on cars without a clutch.",
        "I hurt my left knee skiing and my doctor said no clutch work for at least a year.",
        "My wife and I share the car, and neither of us has ever driven a stick shift.",
    ],
    "awd": [
        "We live at the top of a long gravel hill outside Houghton, and last winter our front-wheel-drive car slid back down it twice.",
        "I'm a home health nurse covering rural roads up north, and the back roads aren't plowed until noon. I need something that puts power to every wheel.",
        "Our cabin road turns to mud every spring and I've had to get towed out with a two-wheel-drive car before.",
    ],
    "seats": {
        6: ["It's my husband and me, our three kids, and Grandma comes with us to church every Sunday."],
        7: ["We have three kids in school, and two days a week I also drive my neighbor's twins to practice.",
            "Our family is five, and my in-laws live with us and come along to most outings."],
        8: ["We have four kids and I run the soccer carpool, which adds two more kids most afternoons."],
    },
    "body": [
        "I just bought a small travel trailer that I need to tow up north, and I'm always hauling bags of mulch and firewood that I don't want inside the car.",
        "I flip furniture as a side gig, so I'm constantly loading dressers and tables that I need to strap down in an open bed.",
    ],
}
