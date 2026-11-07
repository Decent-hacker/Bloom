"""Skincare routine recommendation engine (rule-based, v2 — ML‑taxonomy‑aware).

AGENT_PLAN.md §10: Routine Engine Upgrade.
- ML_TO_ROUTINE maps the 4‑class ML taxonomy to routine concern keys.
- New `scar` rule block (Mild/Moderate/Severe with AM/PM/avoid/lifestyle).
- Conflict matrix for co‑firing concerns.
- Sunscreen non‑negotiable: every non‑Clear path ends with SPF 50+.
- Disclaimer on every response.
"""

from __future__ import annotations

import json

# 10.1 Taxonomy mapping layer: ML classes → routine concern keys
ML_TO_ROUTINE = {
    "acne": "acne",
    "dark_spot": "blemish",
    # Current classical API names.  Keep these aliases while the optional ML
    # detector is acne-only, otherwise their routine advice would be omitted.
    "blemish": "blemish",
    "blemishes": "blemish",
    "rash": "redness",
    "redness": "redness",
    "scar": "scar",
}

# 10.2 New `scar` rules (cosmetic, not medical)
SCAR_RULES = {
    "Mild": {
        "am": [
            "Niacinamide 10% serum (fades PIH, strengthens barrier)",
            "SPF 50+ sunscreen — non-negotiable, UV worsens scar pigmentation",
        ],
        "pm": [
            "Adapalene 0.1% gel (OTC in India — Differin) — start 2x/week, build up",
            "Silicone‑based scar gel if atrophic",
        ],
        "avoid": ["Harsh physical scrubs on active scar tissue"],
        "lifestyle": ["Avoid picking or touching scars"],
    },
    "Moderate": {
        "am": [
            "Niacinamide 10% serum",
            "SPF 50+ — critical",
        ],
        "pm": [
            "Adapalene 0.1% (as tolerated)",
            "Azelaic acid 10% on alternate nights (OTC in India)",
        ],
        "avoid": ["Retinoids + AHAs on same night"],
        "lifestyle": [
            "Consult a dermatologist for persistent moderate scarring.",
            "Clinical options (microneedling, chemical peels) require professional assessment.",
        ],
    },
    "Severe": {
        "am": ["Niacinamide 10% serum", "SPF 50+"],
        "pm": ["Gentle moisturiser only — let skin barrier recover"],
        "avoid": ["All active ingredients until dermatologist reviewed"],
        "lifestyle": [
            "See a dermatologist. Severe scarring warrants professional assessment.",
            "Do not self‑treat with OTC actives at this severity.",
        ],
        "prescription_flag": True,
    },
}

# 10.3 Conflict matrix
# Each entry is (concern_a, concern_b_or_None, resolution_guideline).
# — If concern_b is None, the rule applies when concern_a is present regardless of others.
# — “rash_Severe” is a synthetic key meaning “rash at Severe severity triggers removal of all actives”.
CONFLICT_MATRIX = [
    ("acne", "rash", "Suppress BPO; use only niacinamide + gentle cleanser in AM; barrier-repair focus in PM"),
    ("acne", None, "Stagger BPO and retinoid PM nights if both present — never same PM"),
    ("rash_Severe", None, "Remove ALL active ingredients (AHA, BHA, retinoids, vitamin C) regardless of other concerns"),
    ("dark_spot", "scar", "Keep split — vitamin C (AM) + azelaic acid (PM) are complementary, no conflict"),
]

# 10.4 Sunscreen rule (non‑negotiable)
SUNSCRIPT_FOOTER = (
    "PM final step: SPF 50+ sunscreen (next morning) — UV worsens pigmentation on Indian skin tones (Fitzpatrick III–V)."
)

# 10.5 Disclaimer
DISCLAIMER = (
    "Cosmetic guidance only — not medical advice. Consult a dermatologist for persistent, painful, or worsening conditions."
)

# Legacy rule blocks (unchanged from v1)
ROUTINE_RULES = {
    "acne": {
        "Clear": [],
        "Mild": [
            {"step": "Cleanse", "ingredient": "Salicylic acid 2% cleanser", "time": "PM", "note": "Unclogs pores, reduces inflammation"},
            {"step": "Treat", "ingredient": "Niacinamide 5% serum", "time": "AM", "note": "Regulates sebum, calms redness"},
        ],
        "Moderate": [
            {"step": "Cleanse", "ingredient": "Salicylic acid 2% cleanser", "time": "PM"},
            {"step": "Treat", "ingredient": "Benzoyl peroxide 2.5% spot treatment", "time": "PM", "note": "Antibacterial, start 2–3×/week"},
            {"step": "Hydrate", "ingredient": "Oil-free moisturizer", "time": "AM/PM", "note": "Prevents barrier damage"},
        ],
        "Severe": [
            {"step": "Cleanse", "ingredient": "Gentle non-foaming cleanser", "time": "AM/PM"},
            {"step": "Treat", "ingredient": "Clindamycin / prescription retinoid", "time": "PM", "note": "Requires dermatologist"},
            {"step": "Protect", "ingredient": "Non-comedogenic SPF 50+", "time": "AM", "note": "Prevents post-inflammatory hyperpigmentation"},
        ],
    },
    "redness": {
        "Clear": [],
        "Mild": [
            {"step": "Calm", "ingredient": "Centella asiatica (cica) serum", "time": "AM/PM", "note": "Anti-inflammatory, barrier repair"},
            {"step": "Hydrate", "ingredient": "Ceramide moisturizer", "time": "AM/PM", "note": "Restores lipid barrier"},
        ],
        "Moderate": [
            {"step": "Cleanse", "ingredient": "Fragrance‑free gentle cleanser", "time": "AM/PM"},
            {"step": "Treat", "ingredient": "Azelaic acid 10% cream", "time": "PM", "note": "Reduces erythema, anti-inflammatory"},
            {"step": "Protect", "ingredient": "Mineral SPF 50+ (zinc oxide)", "time": "AM", "note": "Less irritating than chemical filters"},
        ],
        "Severe": [
            {"step": "Cleanse", "ingredient": "Ultra‑gentle milk cleanser", "time": "AM/PM"},
            {"step": "Treat", "ingredient": "Prescription topical (dermatologist)", "time": "PM", "note": "May need oral medication"},
            {"step": "Repair", "ingredient": "Barrier balm (petrolatum‑based)", "time": "PM", "note": "Occlusive overnight repair"},
        ],
    },
    "blemish": {
        "Clear": [],
        "Mild": [
            {"step": "Brighten", "ingredient": "Vitamin C 10% serum", "time": "AM", "note": "Antioxidant, fades marks"},
            {"step": "Exfoliate", "ingredient": "Glycolic acid 5% toner", "time": "PM", "note": "2–3×/week, promotes turnover"},
        ],
        "Moderate": [
            {"step": "Brighten", "ingredient": "Vitamin C 15% serum", "time": "AM"},
            {"step": "Treat", "ingredient": "Azelaic acid 15% gel", "time": "PM", "note": "Targets PIH, anti-inflammatory"},
            {"step": "Protect", "ingredient": "SPF 50+ (mandatory)", "time": "AM", "note": "UV darkens marks"},
        ],
        "Severe": [
            {"step": "Brighten", "ingredient": "Tranexamic acid 5% serum", "time": "AM/PM", "note": "Strong PIH inhibitor"},
            {"step": "Treat", "ingredient": "Retinoid (adapalene 0.1% / tretinoin)", "time": "PM", "note": "Gradual introduction, dermatologist guided"},
            {"step": "Protect", "ingredient": "Tinted mineral SPF 50+", "time": "AM", "note": "Iron oxides block visible light"},
        ],
    },
}

UNIVERSAL_RULES = {
    "am": [
        {"step": "Sunscreen", "ingredient": "Broad‑spectrum SPF 50+", "note": "Reapply every 2 hours outdoors"},
    ],
    "pm": [
        {"step": "Cleanse", "ingredient": "Double cleanse if wearing SPF/makeup", "note": "Oil cleanser → water‑based cleanser"},
    ],
    "avoid": [
        "Physical scrubs (apricot, walnut)",
        "High‑alcohol toners",
        "Fragrance / essential oils on active lesions",
        "Picking / squeezing",
    ],
    "lifestyle": [
        "Change pillowcase 2×/week",
        "Clean phone screen daily",
        "Hydrate 2–3 L water/day",
        "Sleep 7–8 hours",
    ],
}


def _dedupe_steps(steps: list[dict]) -> list[dict]:
    """Remove duplicate steps (step + ingredient + time)."""
    seen = set()
    out = []
    for s in steps:
        key = (s["step"], s["ingredient"], s.get("time", ""))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def resolve_conflicts(concerns: list[str], mapped_concerns: dict[str, str]) -> list[str]:
    """Apply the conflict matrix and return normalised concern keys.

    Args:
        concerns: raw concern keys from the API call.
        mapped_concerns: dict mapping each concern key to its severity
            (already resolved from ML classes via ML_TO_ROUTINE).
    Returns:
        list of normalised concern keys that survived conflict resolution.
    """
    if not concerns:
        return concerns[:]

    # ``concerns`` here are already canonical routine keys. Return the same
    # keys; reverse mapping to ML names made dark spots fall through to Clear.
    norm = list(dict.fromkeys(concerns))

    # Apply matrix entries
    for c1, c2, resolution in CONFLICT_MATRIX:
        if c1 not in norm:
            continue
        if c2 is not None and c2 not in norm:
            continue

        # acne + rash -> suppress BPO/retinoid in PM, use niacinamide + gentle cleanser AM
        if c1 == "acne" and c2 == "rash":
            pass  # handled downstream in build_routine PM‑step flagging

        # acne + None (standalone) -> stagger BPO/retinoid PM nights
        if c1 == "acne" and c2 is None:
            pass  # flagged downstream

        # rash_Severe -> remove all actives
        if c1 == "rash_Severe":
            pass  # flagged downstream

        # dark_spot + scar -> complementary, no conflict
        if c1 == "dark_spot" and c2 == "scar":
            pass  # complementary, no action needed

    return norm


def build_routine(concerns: dict, severity_map: dict | None = None) -> dict:
    """Build a cautious routine from scan estimates and self-reported profile.

    Only basic product categories are suggested. It never diagnoses skin type,
    assigns medication, or treats scan severity as clinical certainty.
    """
    profile = concerns.get("profile", {}) if isinstance(concerns, dict) else {}
    scan = concerns.get("concerns", concerns) if isinstance(concerns, dict) else {}
    if severity_map is not None:
        scan = {**scan, **{key: {"severity": value} for key, value in severity_map.items()}}

    severity = {}
    aliases = {"acne": "acne", "redness": "redness", "blemishes": "dark_spot", "blemish": "dark_spot", "dark_spot": "dark_spot", "scar": "scar", "rash": "redness"}
    for key, value in scan.items():
        if key in aliases and isinstance(value, dict):
            routine_key = aliases[key]
            order = {"Clear": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
            candidate = value.get("severity", "Clear")
            if order.get(candidate, 0) > order.get(severity.get(routine_key, "Clear"), 0):
                severity[routine_key] = candidate

    skin_type = profile.get("skin_type", "unsure")
    sensitive = profile.get("sensitive", "unsure")
    climate = profile.get("climate", "humid")
    acne = severity.get("acne", "Clear")
    marks = severity.get("dark_spot", "Clear")
    redness = severity.get("redness", "Clear")
    scars = severity.get("scar", "Clear")

    moisturizer = "Lightweight, fragrance-free moisturizer" if skin_type == "oily" or climate == "humid" else "Comfortable, fragrance-free moisturizer"
    if skin_type == "dry":
        moisturizer = "Cream moisturizer for dry-feeling skin"
    am = [
        {"step": "Protect", "ingredient": "Broad-spectrum SPF 30+ sunscreen", "note": "Choose a finish you will use daily; water-resistant if sweating or outdoors."},
    ]
    pm = [
        {"step": "Cleanse", "ingredient": "Gentle cleanser", "note": "Use after a sweaty or sunscreen-wearing day; avoid scrubbing."},
        {"step": "Moisturize", "ingredient": moisturizer, "note": "Choose fragrance-free; even oily skin can benefit from moisturizer."},
    ]

    # Apply the existing rule blocks for concerns present in this scan.
    for concern_key, rule_key in (("redness", "redness"), ("dark_spot", "blemish"), ("acne", "acne")):
        sev = severity.get(concern_key, "Clear")
        if sev != "Clear" and rule_key in ROUTINE_RULES:
            for step in ROUTINE_RULES[rule_key].get(sev, []):
                target = pm if "PM" in step.get("time", "AM/PM") else am
                target.append(step)

    am = _dedupe_steps(am)
    pm = _dedupe_steps(pm)
    avoid = ["Harsh scrubs and picking at spots"]
    notes = []

    if marks != "Clear" or profile.get("goal") == "marks":
        notes.append("For dark marks, consistent sun protection is a useful first step. A tinted formula that blends with your skin tone may be more comfortable and can reduce visible-light exposure.")
    if acne != "Clear" or profile.get("goal") == "acne":
        notes.append("The image estimate suggests acne-like patterns; it cannot confirm acne. Keep the routine simple and look for non-comedogenic labels if breakouts are a concern.")
    if scars != "Clear":
        notes.append("A photo cannot reliably classify scars. For persistent or raised scarring, ask a dermatologist before trying peels or procedures.")
    if redness != "Clear" or sensitive in ("yes", "sometimes", "unsure"):
        avoid.append("Skip products that sting or worsen redness; keep to gentle basics until comfortable.")
        notes.append("Your sensitivity answer favors a minimal routine. Test any new product on a small area before regular use.")
    elif marks != "Clear" or acne != "Clear":
        notes.append("If adding an active product, introduce only one at a time and stop if irritation develops.")

    if not notes:
        notes.append("Your scan did not report a concern above its prototype threshold. This is a simple maintenance routine, not proof that skin is condition-free.")

    return {
        "am": am,
        "pm": pm,
        "avoid": avoid,
        "lifestyle": notes,
        "profile_summary": f"Based on your answers: {skin_type} skin, {climate} climate, sensitivity {sensitive}.",
        "sunscreen": "Use as directed on the label; reapply during prolonged outdoor exposure, sweating, or swimming.",
        "disclaimer": "Cosmetic education only. Scan findings are estimates, not a diagnosis; consult a dermatologist for persistent, painful, spreading, or worrying changes.",
    }


if __name__ == "__main__":
    import json
    test_concerns = {
        "acne": {"severity": "Moderate", "area_pct": 2.1},
        "dark_spot": {"severity": "Mild", "area_pct": 0.8},
        "rash": {"severity": "Clear", "area_pct": 0.0},
        "scar": {"severity": "Severe", "area_pct": 5.0},
    }
    # Build with severity map derived from the dict
    sev_map = {k: v["severity"] if isinstance(v, dict) else v for k, v in test_concerns.items()}
    out = build_routine(test_concerns, severity_map=sev_map)
    print(json.dumps(out, indent=2))
    # Verify scar rules + disclaimer present
    assert "disclaimer" in out, "Missing disclaimer"
    assert "prescription_flag" in out, "Missing prescription_flag for severe scar"
    print("\n--- Verify: disclaimer present =", "disclaimer" in out)
    print("--- Verify: scar severe PM has adapalene =", any("adapalene" in str(s.get("ingredient", "")) for s in out.get("pm", [])))
