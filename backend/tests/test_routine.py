from routine import build_routine


def test_current_pipeline_names_produce_routine():
    """Test with current pipeline output key names."""
    routine = build_routine({
        "surface_anomalies": {"severity": "Minimal surface irregularities"},
        "redness_contrast": {"severity": "Low variation"},
        "dark_regions": {"severity": "Moderate variation"},
    })
    assert routine["disclaimer"]
    assert len(routine["am"]) >= 1
    assert len(routine["pm"]) >= 1
    assert len(routine["lifestyle"]) >= 1


def test_legacy_names_still_work():
    """Test that old pipeline key names still produce a valid routine."""
    routine = build_routine({
        "acne": {"severity": "Clear"},
        "redness": {"severity": "Mild"},
        "blemishes": {"severity": "Moderate"},
    })
    assert routine["disclaimer"]
    assert len(routine["am"]) >= 1
    assert len(routine["pm"]) >= 1


def test_disclaimer_always_present():
    """Disclaimer must appear on every routine output."""
    routine = build_routine({})
    assert "disclaimer" in routine
    assert len(routine["disclaimer"]) > 0


def test_no_prescription_ingredients():
    """Prescription drug names must not appear in any routine output."""
    import json
    routine = build_routine({
        "surface_anomalies": {"severity": "High variation"},
        "redness_contrast": {"severity": "High variation"},
    })
    routine_str = json.dumps(routine).lower()
    blocked = ["clindamycin", "tretinoin", "prescription retinoid"]
    for word in blocked:
        assert word not in routine_str, f"Prescription ingredient found: {word}"
