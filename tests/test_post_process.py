import argparse
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from wisper_cli import (
    DEFAULT_POST_PROCESS_PROMPT,
    AppError,
    maybe_post_process_text,
    normalize_final_transcript,
    normalize_spoken_numerics,
    post_process_text,
)


class NumericPostProcessTest(unittest.TestCase):
    def test_spoken_decimals_become_literal_numbers(self) -> None:
        self.assertEqual(normalize_spoken_numerics("zero point one"), "0.1")
        self.assertEqual(normalize_spoken_numerics("six point three"), "6.3")
        self.assertEqual(
            normalize_spoken_numerics("version twelve point zero"), "version 12.0"
        )
        self.assertEqual(normalize_spoken_numerics("zero point zero five"), "0.05")
        self.assertEqual(
            normalize_spoken_numerics("one hundred and five point six"), "105.6"
        )
        self.assertEqual(normalize_final_transcript("zero point one."), "0.1")

    def test_numeric_prefix_becomes_literal_number(self) -> None:
        self.assertEqual(normalize_spoken_numerics("numeric one"), "1")
        self.assertEqual(normalize_spoken_numerics("numeric three"), "3")
        self.assertEqual(normalize_spoken_numerics("numeric zero"), "0")
        self.assertEqual(normalize_spoken_numerics("numeric twenty one"), "21")
        self.assertEqual(
            normalize_spoken_numerics("numeric one hundred and five"), "105"
        )

    def test_conjunctions_do_not_get_consumed_as_number_words(self) -> None:
        self.assertEqual(
            normalize_spoken_numerics("one and two point three"), "one and 2.3"
        )
        self.assertEqual(
            normalize_spoken_numerics("numeric one and numeric zero"), "1 and 0"
        )

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

    def test_short_statement_style_removes_sentence_case_and_final_period(self) -> None:
        self.assertEqual(normalize_final_transcript("Fair point."), "fair point")
        self.assertEqual(
            normalize_final_transcript("Because it will be simpler this way."),
            "because it will be simpler this way",
        )
        self.assertEqual(
            normalize_final_transcript("Version zero point one."), "version 0.1"
        )
        self.assertEqual(normalize_final_transcript("A fair point."), "a fair point")
        self.assertEqual(normalize_final_transcript("i mean"), "I mean")
        self.assertEqual(normalize_final_transcript("i think so"), "I think so")
        self.assertEqual(normalize_final_transcript("i'm sure"), "I'm sure")
        self.assertEqual(normalize_final_transcript("I mean."), "I mean")
        self.assertEqual(normalize_final_transcript("It's fine."), "it's fine")

    def test_short_statement_style_preserves_questions_and_two_sentence_text(
        self,
    ) -> None:
        self.assertEqual(
            normalize_final_transcript(
                "That's a fair point. Let's go with this approach."
            ),
            "That's a fair point. Let's go with this approach.",
        )
        self.assertEqual(
            normalize_final_transcript("Use option 1. Then option 2."),
            "Use option 1. Then option 2.",
        )
        self.assertEqual(
            normalize_final_transcript("How can we solve it?"), "How can we solve it?"
        )

    def test_short_statement_style_preserves_acronyms_and_identifiers(self) -> None:
        self.assertEqual(normalize_final_transcript("API request."), "API request")
        self.assertEqual(normalize_final_transcript("Use API."), "use API")
        self.assertEqual(normalize_final_transcript("use API"), "use API")
        self.assertEqual(normalize_final_transcript("for i in items"), "for i in items")
        self.assertEqual(normalize_final_transcript("i in items"), "i in items")
        self.assertEqual(
            normalize_final_transcript("TypeScript type."), "TypeScript type"
        )
        self.assertEqual(
            normalize_final_transcript("JavaScript module."), "JavaScript module"
        )

    def test_long_single_statement_uses_sentence_style(self) -> None:
        self.assertEqual(
            normalize_final_transcript(
                "because it will be simpler this way for all now."
            ),
            "because it will be simpler this way for all now",
        )
        self.assertEqual(
            normalize_final_transcript(
                "because it will be simpler this way and it reduces complexity overall"
            ),
            "Because it will be simpler this way and it reduces complexity overall.",
        )
        self.assertEqual(
            normalize_final_transcript(
                "i think this approach will be simpler because it reduces complexity overall"
            ),
            "I think this approach will be simpler because it reduces complexity overall.",
        )
        self.assertEqual(
            normalize_final_transcript(
                "TypeScript type inference should stay unchanged when it starts the statement"
            ),
            "TypeScript type inference should stay unchanged when it starts the statement.",
        )

    def test_non_latin_transcripts_are_not_restyled_locally(self) -> None:
        self.assertEqual(normalize_final_transcript("Хорошая мысль."), "Хорошая мысль.")
        self.assertEqual(
            normalize_final_transcript("Как это исправить?"), "Как это исправить?"
        )
        self.assertEqual(normalize_final_transcript("Привет 123."), "Привет 123.")

    def test_default_prompt_preserves_coherent_non_english_text(self) -> None:
        self.assertIn(
            "Preserve the transcript's original language", DEFAULT_POST_PROCESS_PROMPT
        )
        self.assertIn(
            "Never translate complete coherent non-English text into English",
            DEFAULT_POST_PROCESS_PROMPT,
        )
        self.assertIn(
            "Never translate English or code-heavy transcripts into another language",
            DEFAULT_POST_PROCESS_PROMPT,
        )
        self.assertIn("wrong keyboard layout", DEFAULT_POST_PROCESS_PROMPT)

    def test_post_processing_rejects_non_latin_translation_of_english_input(
        self,
    ) -> None:
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return {
                    "output_text": "Давайте сначала зафиксируем commit для test harness."
                }

        old_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            with patch("wisper_cli.requests.post", return_value=FakeResponse()):
                self.assertEqual(
                    post_process_text(
                        "Let's commit the test harness fix first.",
                        model_name="gpt-test",
                        prompt="clean",
                        glossary_file=None,
                        timeout=1.0,
                        verbose=False,
                    ),
                    "let's commit the test harness fix first",
                )
        finally:
            if old_api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_api_key


if __name__ == "__main__":
    unittest.main()
