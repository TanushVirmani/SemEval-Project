import tempfile
import hashlib
import json
import sys
from unittest.mock import patch
import unittest
from pathlib import Path
from model_utils import add_language_prefix, read_labeled_csv, validate_checkpoint


class DatasetTests(unittest.TestCase):
    def csv(self, content):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "test.csv"
        path.write_text(content)
        return path

    def test_fractional_label_rejected(self):
        with self.assertRaises(ValueError):
            read_labeled_csv(
                self.csv("code,label,language\na,0,Python\nb,1,Java\nc,0.5,Python\n")
            )

    def test_missing_values_rejected(self):
        with self.assertRaises(ValueError):
            read_labeled_csv(self.csv("code,label,language\na,0,Python\n,1,Java\n"))

    def test_both_classes_required(self):
        with self.assertRaises(ValueError):
            read_labeled_csv(self.csv("code,label,language\na,0,Python\n"))

    def test_valid_data(self):
        df = read_labeled_csv(self.csv("code,label,language\na,0,Python\nb,1,Java\n"))
        self.assertEqual(df.label.tolist(), [0, 1])

    def test_numeric_code_remains_text(self):
        df = read_labeled_csv(
            self.csv("code,label,language\n123,0,Python\n456,1,Python\n")
        )
        self.assertEqual(df.code.tolist(), ["123", "456"])

    def test_prefix_consistent(self):
        self.assertEqual(
            add_language_prefix("Python", " x "), add_language_prefix(" python ", "x")
        )
        self.assertEqual(
            add_language_prefix("cpp", "x"), add_language_prefix("C++", "x")
        )

    def test_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                validate_checkpoint(folder)


class LeakageTests(unittest.TestCase):
    def test_evaluator_rejects_training_code_before_loading_weights(self):
        import evaluate_model
        from model_utils import LABELS

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(json.dumps({"id2label": LABELS}))
            (checkpoint / "model.safetensors").touch()
            (checkpoint / "training_metadata.json").write_text(
                json.dumps(
                    {
                        "training_code_hashes": [
                            hashlib.sha256(b"seen code").hexdigest()
                        ],
                        "validation_code_hashes": [],
                        "max_length": 256,
                        "steps_completed": 1,
                        "smoke_run": True,
                    }
                )
            )
            csv = root / "test.csv"
            csv.write_text(
                "code,label,language\nseen code,0,Python\nunseen code,1,Java\n"
            )
            with patch.object(
                sys,
                "argv",
                [
                    "evaluate_model.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--test-csv",
                    str(csv),
                ],
            ):
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    evaluate_model.main()


if __name__ == "__main__":
    unittest.main()
