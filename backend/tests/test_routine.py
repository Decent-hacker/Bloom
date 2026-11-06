from routine import build_routine


def test_classical_concern_names_produce_routine_steps():
    routine = build_routine({
        "acne": {"severity": "Clear"},
        "redness": {"severity": "Mild"},
        "blemishes": {"severity": "Moderate"},
    })

    ingredients = " ".join(step["ingredient"] for step in routine["am"] + routine["pm"])
    assert "Centella" in ingredients
    assert "Vitamin C" in ingredients
    assert routine["disclaimer"]
