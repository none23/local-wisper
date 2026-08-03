import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from wisper_cli import (
    DEFAULT_POST_PROCESS_PROMPT,
    AppError,
    apply_guaranteed_corrections,
    maybe_post_process_text,
    normalize_final_transcript,
    normalize_spoken_numerics,
    parse_correction_glossary,
    post_process_text,
)


class NumericPostProcessTest(unittest.TestCase):
    def test_structured_glossary_parses_confidence_sections(self) -> None:
        glossary = parse_correction_glossary(
            """
            # Local guarantees
            [always]
            engine x -> nginx

            [likely]
            cloud code -> Claude Code

            [contextual]
            codecs -> Codex

            [terms]
            TypeScript
            """
        )

        self.assertEqual(glossary.always, (("engine x", "nginx"),))
        self.assertEqual(glossary.likely, (("cloud code", "Claude Code"),))
        self.assertEqual(glossary.contextual, (("codecs", "Codex"),))
        self.assertEqual(glossary.terms, ("TypeScript",))
        self.assertIsNone(glossary.legacy_text)

    def test_unsectioned_glossary_remains_legacy_prompt_text(self) -> None:
        raw = "Common intended terms:\nTypeScript\nengine x -> nginx"
        self.assertEqual(parse_correction_glossary(raw).legacy_text, raw)

    def test_structured_glossary_rejects_duplicate_sources(self) -> None:
        with self.assertRaisesRegex(AppError, "appears in both"):
            parse_correction_glossary(
                "[always]\ncodecs -> Codex\n[contextual]\ncodecs -> Codex"
            )

    def test_guaranteed_corrections_are_boundary_aware_and_do_not_cascade(
        self,
    ) -> None:
        rules = (("code", "Codex"), ("cloud code", "Claude Code"), ("cat", "dog"))
        self.assertEqual(
            apply_guaranteed_corrections("Cloud code and cat scatter", rules),
            "Claude Code and dog scatter",
        )

    def test_short_transcript_applies_guaranteed_glossary_without_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            glossary_path = Path(tempdir) / "glossary.txt"
            glossary_path.write_text(
                "[always]\nengine x -> nginx\n[terms]\nnginx\n",
                encoding="utf-8",
            )
            with patch("wisper_cli.requests.post") as post:
                result = post_process_text(
                    "Engine X works.",
                    model_name="gpt-test",
                    prompt="clean",
                    glossary_file=str(glossary_path),
                    timeout=1.0,
                    verbose=False,
                )

        self.assertEqual(result, "nginx works")
        post.assert_not_called()

    def test_structured_glossary_describes_model_adherence_levels(self) -> None:
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return {"output_text": "Use Claude Code and Codex in this workflow."}

        old_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                glossary_path = Path(tempdir) / "glossary.txt"
                glossary_path.write_text(
                    "[likely]\ncloud code -> Claude Code\n"
                    "[contextual]\ncodecs -> Codex\n"
                    "[terms]\nTypeScript\n",
                    encoding="utf-8",
                )
                with patch("wisper_cli.requests.post", return_value=FakeResponse()) as post:
                    post_process_text(
                        "Use cloud code and codecs in this workflow.",
                        model_name="gpt-test",
                        prompt="clean",
                        glossary_file=str(glossary_path),
                        timeout=1.0,
                        verbose=False,
                    )

            instructions = post.call_args.kwargs["json"]["instructions"]
            self.assertIn("<likely>", instructions)
            self.assertIn("unless surrounding context clearly contradicts", instructions)
            self.assertIn("<contextual>", instructions)
            self.assertIn("only when surrounding context positively supports", instructions)
            self.assertIn("<canonical_terms>\nTypeScript", instructions)
        finally:
            if old_api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_api_key

    def test_model_failure_keeps_guaranteed_local_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            glossary_path = Path(tempdir) / "glossary.txt"
            glossary_path.write_text(
                "[always]\nengine x -> nginx\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                post_process_model="gpt-test",
                post_process_prompt="clean",
                post_process_glossary_file=str(glossary_path),
                post_process_timeout=1.0,
                verbose=False,
            )
            with patch(
                "wisper_cli.post_process_text", side_effect=AppError("offline")
            ), patch("sys.stderr"):
                result = maybe_post_process_text(
                    "Engine X works for this service.", args
                )

        self.assertEqual(result, "nginx works for this service")

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
        self.assertIn("not as a request to answer", DEFAULT_POST_PROCESS_PROMPT)
        self.assertIn("do not answer it", DEFAULT_POST_PROCESS_PROMPT)

    def test_post_processing_wraps_question_transcript_as_source_text(self) -> None:
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return {
                    "output_text": "How should we wrap the transcript for the model?"
                }

        old_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        transcript = "How should we wrap the transcript for the model?"
        try:
            with patch("wisper_cli.requests.post", return_value=FakeResponse()) as post:
                self.assertEqual(
                    post_process_text(
                        transcript,
                        model_name="gpt-test",
                        prompt=DEFAULT_POST_PROCESS_PROMPT,
                        glossary_file=None,
                        timeout=1.0,
                        verbose=False,
                    ),
                    transcript,
                )

            payload = post.call_args.kwargs["json"]
            self.assertIn("not as a request to answer", payload["instructions"])
            self.assertIn("do not answer it", payload["instructions"])
            self.assertIn(
                "<transcript>\n" + transcript + "\n</transcript>",
                payload["input"],
            )
            self.assertNotEqual(transcript, payload["input"])
        finally:
            if old_api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_api_key

    def test_luna_post_processing_disables_reasoning(self) -> None:
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return {"output_text": "This transcript has enough words to process."}

        old_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            with patch("wisper_cli.requests.post", return_value=FakeResponse()) as post:
                post_process_text(
                    "This transcript has enough words to process.",
                    model_name="gpt-5.6-luna",
                    prompt="clean",
                    glossary_file=None,
                    timeout=1.0,
                    verbose=False,
                )

            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["model"], "gpt-5.6-luna")
            self.assertEqual(payload["reasoning"], {"effort": "none"})
        finally:
            if old_api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_api_key

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
