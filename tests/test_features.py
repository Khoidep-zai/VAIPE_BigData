import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.features import compare_images, compare_feature_vectors, extract_features


class TestFeatures(unittest.TestCase):
    def test_compare_feature_vectors_identical(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[32:96, 32:96] = [240, 240, 40]
        fv_a = extract_features(image.astype(np.float32))
        fv_b = extract_features(image.astype(np.float32))
        result = compare_feature_vectors(fv_a, fv_b)
        self.assertTrue(result["verdict"])
        self.assertGreaterEqual(result["score"], 3)

    def test_compare_images_api(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            p1 = Path(tmp_dir) / "a.png"
            p2 = Path(tmp_dir) / "b.png"

            arr = np.full((120, 120, 3), 240, dtype=np.uint8)
            arr[20:100, 20:100] = [20, 200, 20]
            Image.fromarray(arr).save(p1)
            Image.fromarray(arr).save(p2)

            result = compare_images(str(p1), str(p2))
            self.assertIn("score", result)
            self.assertIn("verdict", result)


if __name__ == "__main__":
    unittest.main()
