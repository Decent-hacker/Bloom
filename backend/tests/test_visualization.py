import numpy as np

from src.visualization import annotate_concerns


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
