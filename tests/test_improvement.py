import unittest
import numpy as np
from improve_model import structural_features, normalize, scores


class ImprovementTests(unittest.TestCase):
    def test_features_finite_for_empty_and_numeric_code(self):
        x = structural_features(["", "123", "def f(x):\n    return x + 1"])
        self.assertEqual(x.shape[0], 3)
        self.assertGreater(x.shape[1], 50)
        self.assertTrue(np.isfinite(x).all())

    def test_normalization_removes_comment_and_literal(self):
        text = normalize('x = 123 # example comment\ns = "hello"')
        self.assertNotIn("example comment", text)
        self.assertNotIn("hello", text)
        self.assertIn("NUM", text)
        self.assertIn("STR", text)

    def test_accuracy_is_not_macro_f1(self):
        result = scores([0, 0, 0, 1], [0.1, 0.1, 0.1, 0.1])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertLess(result["macro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()


class FragmentTests(unittest.TestCase):
    def test_extracts_function_without_execution(self):
        from fragment_views import fragments

        code = "def add(a,b):\n    return a+b\n\ndef subtract(a,b):\n    result = a-b\n    return result\n"
        views = fragments(code, "Python")
        self.assertTrue(views)
        self.assertTrue(all(v in code for v in views))
        self.assertEqual(views, fragments(code, "Python"))

    def test_normalized_group_ignores_renaming(self):
        from improve_model import normalized_fingerprint

        self.assertEqual(
            normalized_fingerprint("def f(x): return x + 1"),
            normalized_fingerprint("def other(y): return y + 99"),
        )


class ThresholdTests(unittest.TestCase):
    def test_threshold_mapping_preserves_decisions(self):
        from detector import decision_scores

        p = np.array([0.1, 0.5, 0.6, 0.9])
        adjusted = decision_scores(p, 0.6)
        np.testing.assert_allclose(adjusted.sum(axis=1), 1)
        self.assertAlmostEqual(adjusted[2, 1], 0.5)
        np.testing.assert_array_equal(adjusted[:, 1] > 0.5, p > 0.6)

    def test_default_threshold_is_identity(self):
        from detector import decision_scores

        p = np.array([0, 0.2, 0.5, 0.8, 1.0])
        np.testing.assert_allclose(decision_scores(p)[:, 1], p)

    def test_threshold_selected_from_validation(self):
        from calibrate_detector import choose_threshold

        best, _ = choose_threshold([0, 0, 1, 1], [0.1, 0.55, 0.7, 0.9])
        self.assertGreater(best["threshold"], 0.55)
        self.assertLessEqual(best["threshold"], 0.7)
        self.assertEqual(best["accuracy"], 1.0)


class TokenizationTests(unittest.TestCase):
    def test_head_tail_preserves_both_ends_with_special_tokens(self):
        from model_utils import encode_texts

        class Tokenizer:
            def num_special_tokens_to_add(self, pair=False):
                return 2

            def __call__(self, texts, **kwargs):
                return {"input_ids": [list(range(20)) for _ in texts]}

            def prepare_for_model(self, ids, **kwargs):
                return {
                    "input_ids": [101] + ids + [102],
                    "attention_mask": [1] * (len(ids) + 2),
                }

        encoded = encode_texts(Tokenizer(), ["long code"], 8, "head_tail")
        self.assertEqual(encoded["input_ids"][0], [101, 0, 1, 2, 17, 18, 19, 102])
        self.assertEqual(len(encoded["attention_mask"][0]), 8)
