from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class CompatibilityTest(unittest.TestCase):
    def test_legacy_python_module_is_the_application_module(self) -> None:
        legacy = importlib.import_module("wisper_cli")
        implementation = importlib.import_module("local_wisper.cli")

        self.assertIs(legacy, implementation)
        self.assertEqual(
            implementation._daemon_script_path(),
            REPO_ROOT / "scripts" / "transcribe_daemon.py",
        )

    def test_legacy_cli_path_still_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "wisper_cli.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sway-start", result.stdout)
        self.assertIn("sway-stop", result.stdout)

    def test_sway_wrapper_keeps_the_lw_command_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args_path = tmp_path / "args"
            fake_lw = tmp_path / "lw"
            fake_lw.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"${LW_TEST_ARGS_PATH}\"\n"
            )
            fake_lw.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(tmp_path),
                    "LW_BIN": str(fake_lw),
                    "LW_ENV_FILE": str(tmp_path / "missing-env"),
                    "LW_TEST_ARGS_PATH": str(args_path),
                }
            )
            result = subprocess.run(
                [str(REPO_ROOT / "integrations" / "sway" / "local-wisper.sh"), "sway-stop"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                args_path.read_text().splitlines(),
                [
                    "--backend",
                    "parakeet",
                    "--device",
                    "cuda",
                    "--sample-rate",
                    "16000",
                    "--compute-type",
                    "float16",
                    "--no-vad-filter",
                    "--type-output",
                    "sway-stop",
                ],
            )


if __name__ == "__main__":
    unittest.main()
