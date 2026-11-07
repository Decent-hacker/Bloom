import base64

import cv2
import numpy as np

from src.visualization import annotate_concerns, generate_all


def _concern(mask, regions):
    return {"mask": mask, "regions": regions}


def test_annotated_image_contains_labelled_region_boxes():
    image = np.full((80, 80, 3), 240, dtype=np.uint8)
    acne = np.zeros((80, 80), dtype=np.uint8)
    acne[20:40, 20:40] = 255
    concerns = {
        "acne": _concern(acne, [{"bbox": (20, 20, 40, 40)}]),
        "redness": _concern(np.zeros((80, 80), dtype=np.uint8), []),
        "blemish": _concern(np.zeros((80, 80), dtype=np.uint8), []),
    }

    annotated = annotate_concerns(image, concerns)

    # A blue acne box changes pixels at the expected bounding-box edge.
    assert not np.array_equal(annotated[20, 20], image[20, 20])
    assert annotated.shape == image.shape


def test_frequency_maps_are_encoded_as_viewable_jpegs():
    image = np.full((32, 32, 3), 140, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    concerns = {
        name: {"mask": mask, "regions": [], "severity": "Minimal variation"}
        for name in ("surface_anomalies", "redness_contrast", "dark_regions")
    }
    frequency = {
        "low_pass": np.full((32, 32), 140, dtype=np.float32),
        "high_pass": np.full((32, 32), 128, dtype=np.float32),
        "dog": np.zeros((32, 32), dtype=np.float32),
    }

    images = generate_all(image, image, mask, concerns, mask, frequency)

    for name in ("low_pass", "high_pass", "dog"):
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(images[name]), np.uint8), cv2.IMREAD_GRAYSCALE)
        assert decoded is not None
        assert decoded.shape == mask.shape
