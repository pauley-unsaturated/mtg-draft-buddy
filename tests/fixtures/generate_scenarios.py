"""One-time curation script for behavioral fixtures (PLAN P1.T3).

Run manually (`uv run python tests/fixtures/generate_scenarios.py`) to write
tests/fixtures/scenarios/*.json. Fixtures are FROZEN once committed — never
regenerate to make a failing test pass (guardrail). Card strength references
(full-season GIH WR, 2026-07-28 snapshot) are cited in each rationale.
"""

import json
from pathlib import Path

OUT = Path(__file__).parent / "scenarios"

# Shorthand card groups (names must match cards.parquet exactly).
W_GOOD = ["Hero in Training", "Murdock's Crusade", "Web Up"]
U_GOOD = ["Trickster's Stratagem", "S.H.I.E.L.D. Deployment Drone", "We Say Thee Nay!"]
B_GOOD = ["Cruel Alliance", "HYDRA Infiltration", "Widow's Bite"]
R_GOOD = ["Lightning Strike", "HULK SMASH!", "Crimson Operative"]
G_GOOD = ["Undercover Skrull", "Go Nuts!", "Rapid Rescue"]
W_WEAK = ["Kree Commandos", "Panther Pounce", "Agents of S.H.I.E.L.D."]
U_WEAK = ["Atlantean Cavalry", "Hydraulic Helper", "Super Suit"]
B_WEAK = ["Decoy Ploy", "Agents of HYDRA", "Project Deathlok Soldier"]
R_WEAK = ["Hire a Crew", "Super Speed", "Volcanic Villain"]
G_WEAK = ["Super Strength", "Powerful Broker", "Call Damage Control"]
C_WEAK = ["Vibranium Energy Daggers", "A.I.M. Synthoids", "Ultron Drone",
          "Dependable Quinjet"]

FULL14 = W_WEAK[:2] + U_WEAK[:2] + B_WEAK[:2] + R_WEAK[:2] + G_WEAK[:2] + C_WEAK  # 14 weak cards


def scenario(name, tier, position, pack, pool, any_of, k, rationale):
    assert len(pool) == position, f"{name}: pool len {len(pool)} != position {position}"
    assert len(pack) <= 14 and len(pack) >= 1
    assert all(c in pack for c in any_of)
    return {"name": name, "tier": tier, "position": position, "pack": pack,
            "pool": pool, "expect": {"k": k, "any_of": any_of},
            "rationale": rationale}


S = []

# ---------------------------------------------------------------- easy tier ---
bombs = [
    ("Light of Promise", "GIH 0.81, format-best mythic"),
    ("Captain Marvel, Earth's Protector", "GIH 0.69 mythic finisher"),
    ("Black Panther, Wakandan King", "GIH 0.69, 2-mana mythic"),
    ("Doctor Doom", "GIH 0.66 bomb"),
    ("Leader, Super-Genius", "GIH 0.68 rare"),
    ("Sword of Fire and Ice", "GIH 0.67 colorless mythic — takeable by any deck"),
]
for i, (bomb, why) in enumerate(bombs):
    S.append(scenario(
        f"easy-p1p1-bomb-{i}", "easy", 0, [bomb] + FULL14[:13], [],
        [bomb], 1, f"P1P1 obvious bomb: {why} vs weak commons."))

S.append(scenario(
    "easy-best-common-p1p2", "easy", 1,
    ["Trickster's Stratagem"] + W_WEAK + B_WEAK + R_WEAK + C_WEAK[:3],
    ["Light of Promise"],
    ["Trickster's Stratagem"], 1,
    "P1P2: premium common (GIH 0.605) vs replacement-level cards."))
S.append(scenario(
    "easy-common-over-trap-mythic", "easy", 0,
    ["Hero in Training", "Horn of Greed"] + FULL14[:12], [],
    ["Hero in Training"], 1,
    "Horn of Greed is a trap mythic (GIH 0.22); the premium common is right. "
    "rarity-first fails this by design."))
S.append(scenario(
    "easy-no-basic-over-playable", "easy", 30,
    ["Plains", "Hero in Training", "Murdock's Crusade"] + U_WEAK[:3] + B_WEAK[:3] + R_WEAK[:3],
    W_GOOD * 5 + W_WEAK * 5,  # 30-card mono-W pool
    ["Hero in Training", "Murdock's Crusade"], 1,
    "Never take a basic land over on-color playables (play boosters put basics "
    "in packs; 3.4% of human picks are basics but not over playables)."))
S.append(scenario(
    "easy-two-good-one-great", "easy", 0,
    ["Final Showdown", "Hero in Training", "Undercover Skrull"] + FULL14[:11], [],
    ["Final Showdown"], 1,
    "Mythic removal (GIH 0.68) over premium commons P1P1."))

# -------------------------------------------------------------- medium tier ---
S.append(scenario(
    "medium-stay-in-lane-p3", "medium", 30,
    ["Trickster's Stratagem", "Lightning Strike", "HULK SMASH!"] + W_WEAK + G_WEAK + C_WEAK[:5],
    (B_GOOD + R_GOOD) * 5,  # heavily committed BR pool, 30 cards
    ["Lightning Strike", "HULK SMASH!"], 1,
    "P3 with a committed BR pool: on-color removal over a slightly higher-GIH "
    "off-color common. Pure GIH-greedy fails; pool context wins."))
S.append(scenario(
    "medium-stay-in-lane-p2", "medium", 20,
    ["Undercover Skrull", "Murdock's Crusade", "Web Up"] + R_WEAK + B_WEAK + C_WEAK[:4],
    (W_GOOD + U_GOOD) * 3 + ["Hero in Training", "We Say Thee Nay!"],
    ["Murdock's Crusade", "Web Up"], 1,
    "P2P7 committed WU: take the on-color common over the off-color G common "
    "with marginally better raw stats."))
S.append(scenario(
    "medium-fixing-for-splash", "medium", 25,
    ["Avengers Tower", "Kree Commandos", "Super Suit"] + B_WEAK + R_WEAK[:2] + G_WEAK + C_WEAK[:2],
    (W_GOOD + U_GOOD) * 4 + ["The Super Hero Civil War"],  # WU + a RW splash bomb
    ["Avengers Tower"], 2,
    "Pool has a splashy off-color bomb: the fixing land (GIH 0.60, ALSA 4.5) "
    "should rank top-2 over replacement-level spells."))
S.append(scenario(
    "medium-late-wheel-playable", "medium", 11,
    ["Widow's Bite", "Super Speed", "Vibranium Energy Daggers"],
    B_GOOD + B_WEAK + W_WEAK + ["Doctor Doom", "Cruel Alliance"],
    ["Widow's Bite"], 1,
    "P1P12 wheel, mono-B leaning pool: the on-color playable over chaff."))
S.append(scenario(
    "medium-removal-over-vanilla", "medium", 5,
    ["Lightning Strike", "Volcanic Villain", "Hire a Crew"] + G_WEAK + U_WEAK + C_WEAK[:3],
    R_GOOD[1:] + ["HULK SMASH!", "Crimson Operative", "Wonder Man, Hollywood Hero"],
    ["Lightning Strike"], 1,
    "Committed red: premium removal over vanilla creatures, easy for anything "
    "stat-aware, medium because the pool must not distract."))
S.append(scenario(
    "medium-colorless-bomb-early-lane-open", "medium", 2,
    ["Sword of Fire and Ice", "Hero in Training", "Cruel Alliance"] + R_WEAK + U_WEAK + C_WEAK[:4],
    ["Light of Promise", "Murdock's Crusade"],
    ["Sword of Fire and Ice"], 2,
    "P1P3, pool leaning W: the colorless mythic keeps options open and is the "
    "strongest card; top-2 tolerance for taking the on-color common instead."))
S.append(scenario(
    "medium-second-color-commit", "medium", 14,
    ["We Say Thee Nay!", "Rapid Rescue", "Decoy Ploy"] + W_WEAK + R_WEAK + C_WEAK[:4],
    W_GOOD * 3 + ["Trickster's Stratagem", "S.H.I.E.L.D. Deployment Drone",
                  "Light of Promise", "Captain Marvel, Earth's Protector", "Web Up"],
    ["We Say Thee Nay!"], 1,
    "P2P1: pool is W + light U; take the U playable that confirms the second "
    "color over off-color options."))

# ---------------------------------------------------------------- hard tier ---
S.append(scenario(
    "hard-speculative-p1p1-flexibility", "hard", 0,
    ["Jennifer Walters", "Trickster's Stratagem", "Undercover Skrull"] + FULL14[:11], [],
    ["Jennifer Walters", "Trickster's Stratagem"], 2,
    "Close P1P1 between a strong mythic and the best common; either is defensible, "
    "model should have both in top-2."))
S.append(scenario(
    "hard-pivot-signal", "hard", 8,
    ["Leader, Super-Genius", "Murdock's Crusade", "Widow's Bite"] + G_WEAK + R_WEAK + C_WEAK[:3],
    ["Hero in Training", "Web Up", "Kree Commandos", "Panther Pounce",
     "Atlantean Cavalry", "Hydraulic Helper", "Super Suit", "Agents of S.H.I.E.L.D."],
    ["Leader, Super-Genius"], 2,
    "P1P9: a late rare this strong is a signal U is open; pivoting beats "
    "doubling down on a weak W start."))
S.append(scenario(
    "hard-curve-consideration", "hard", 35,
    ["HULK SMASH!", "Crimson Operative", "Super Speed"] + W_WEAK + U_WEAK + C_WEAK[:2],
    (B_GOOD + R_GOOD) * 5 + B_WEAK + ["Doctor Doom", "M.O.D.O.K."],
    ["HULK SMASH!", "Crimson Operative"], 2,
    "P3P8 with a full BR pool: late playables — either on-color pick fine, "
    "basics/off-color wrong."))
S.append(scenario(
    "hard-dual-land-vs-spell-2c", "hard", 18,
    ["Gathering Place", "Widow's Bite", "Rapid Rescue"] + W_WEAK + U_WEAK + C_WEAK[:3],
    (B_GOOD + G_GOOD) * 3,
    ["Widow's Bite", "Rapid Rescue", "Gathering Place"], 2,
    "2-color BG pool: spell usually beats fixing, but the land is defensible; "
    "expect top-2 among these three."))

for s in S:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{s['name']}.json").write_text(json.dumps(s, indent=1))
print(f"wrote {len(S)} scenarios to {OUT}")
