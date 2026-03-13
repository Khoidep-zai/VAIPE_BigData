import tempfile
import unittest
from pathlib import Path

from src.inference import find_sample_image


class TestInferenceUtils(unittest.TestCase):
    def test_find_sample_image(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "train" / "class_0"
            target.mkdir(parents=True)
            image_path = target / "x.jpg"
            image_path.write_bytes(b"fake")

            found = find_sample_image(str(root), "class_0")
            self.assertTrue(found.endswith("x.jpg"))


if __name__ == "__main__":
    unittest.main()
