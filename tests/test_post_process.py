import argparse
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from wisper_cli import AppError, maybe_post_process_text, normalize_spoken_numerics, post_process_text


class NumericPostProcessTest(unittest.TestCase):
    def test_spoken_decimals_become_literal_numbers(self) -> None:
        self.assertEqual(normalize_spoken_numerics("zero point one"), "0.1")
        self.assertEqual(normalize_spoken_numerics("six point three"), "6.3")
        self.assertEqual(normalize_spoken_numerics("version twelve point zero"), "version 12.0")
        self.assertEqual(normalize_spoken_numerics("zero point zero five"), "0.05")
        self.assertEqual(normalize_spoken_numerics("one hundred and five point six"), "105.6")

    def test_numeric_prefix_becomes_literal_number(self) -> None:
        self.assertEqual(normalize_spoken_numerics("numeric one"), "1")
        self.assertEqual(normalize_spoken_numerics("numeric three"), "3")
        self.assertEqual(normalize_spoken_numerics("numeric zero"), "0")
        self.assertEqual(normalize_spoken_numerics("numeric twenty one"), "21")
        self.assertEqual(normalize_spoken_numerics("numeric one hundred and five"), "105")

    def test_conjunctions_do_not_get_consumed_as_number_words(self) -> None:
        self.assertEqual(normalize_spoken_numerics("one and two point three"), "one and 2.3")
        self.assertEqual(normalize_spoken_numerics("numeric one and numeric zero"), "1 and 0")

    def test_short_transcript_skips_openai_post_processing(self) -> None:
        old_api_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            self.assertEqual(
                post_process_text(
                    "zero point one",
                    model_name="gpt-test",
                    prompt="clean",
                    glossary_file=None,
                    timeout=1.0,
                    verbose=False,
                ),
                "0.1",
            )
        finally:
            if old_api_key is not None:
                os.environ["OPENAI_API_KEY"] = old_api_key

    def test_five_word_transcript_skips_openai_post_processing(self) -> None:
        old_api_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            self.assertEqual(
                post_process_text(
                    "zero point one is done",
                    model_name="gpt-test",
                    prompt="clean",
                    glossary_file=None,
                    timeout=1.0,
                    verbose=False,
                ),
                "0.1 is done",
            )
        finally:
            if old_api_key is not None:
                os.environ["OPENAI_API_KEY"] = old_api_key

    def test_six_word_transcript_does_not_skip_openai_post_processing(self) -> None:
        old_api_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(AppError):
                post_process_text(
                    "zero point one is done now",
                    model_name="gpt-test",
                    prompt="clean",
                    glossary_file=None,
                    timeout=1.0,
                    verbose=False,
                )
        finally:
            if old_api_key is not None:
                os.environ["OPENAI_API_KEY"] = old_api_key

    def test_numeric_cleanup_runs_without_model(self) -> None:
        args = argparse.Namespace(post_process_model=None)
        self.assertEqual(maybe_post_process_text("numeric three", args), "3")


if __name__ == "__main__":
    unittest.main()
