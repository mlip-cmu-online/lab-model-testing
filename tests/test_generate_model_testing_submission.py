import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate-model-testing-submission.py"


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def complete_evidence() -> dict[str, Any]:
    slice_names = ["emoji", "negation", "hashtag", "mention", "long"]
    stress_rows = []
    for index in range(10):
        expected = "positive" if index % 2 else "negative"
        for model, prediction, confidence in (
            ("baseline_model", expected, 0.81),
            ("candidate_model", expected if index != 3 else "neutral", 0.72),
        ):
            stress_rows.append(
                {
                    "text": f"Synthetic test case {index}",
                    "expected_label": expected,
                    "model": model,
                    "pred": prediction,
                    "conf": confidence,
                    "correct": prediction == expected,
                }
            )
    return {
        "schema_version": "1.0",
        "wandb_run_url": "https://wandb.ai/test-learner/mlip-lab4-slices-2026/runs/run123",
        "wandb_artifacts_logged": [
            "predictions_table",
            "slice_metrics",
            "regression_metrics",
            "df_eval",
            "synthetic_tests",
        ],
        "overall_metrics": [
            {"model": "baseline_model", "accuracy": 0.71},
            {"model": "candidate_model", "accuracy": 0.76},
        ],
        "slice_metrics": [
            {"slice": slice_name, "model": model, "accuracy": 0.5 + index / 100}
            for index, slice_name in enumerate(slice_names)
            for model in ("baseline_model", "candidate_model")
        ],
        "slice_notes": [
            f"The {slice_name} slice exposed a model difference worth reviewing."
            for slice_name in slice_names
        ],
        "stress_test_hypothesis": "Negated praise may be mistaken for positive sentiment.",
        "stress_test_results": stress_rows,
    }


def write_notebook(path: Path, evidence: dict[str, Any], source: str = "x = 1") -> None:
    payload = json.dumps(evidence)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [source],
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": [
                            "MLIP_SUBMISSION_EVIDENCE_V1\n",
                            payload + "\n",
                            "MLIP_SUBMISSION_EVIDENCE_END\n",
                        ],
                    }
                ],
                "source": ["print('submission evidence')"],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


class GenerateModelTestingSubmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.notebook = self.root / "lab4.ipynb"
        self.screenshot = self.root / "wandb-slices.png"
        self.screenshot.write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def generate(self) -> subprocess.CompletedProcess[str]:
        return run(
            "python3",
            str(SCRIPT),
            "--learner",
            "Test Learner",
            "--notebook",
            str(self.notebook),
            "--wandb-evidence",
            str(self.screenshot),
            cwd=self.root,
        )

    def test_generates_complete_self_contained_report_and_manifest(self) -> None:
        write_notebook(self.notebook, complete_evidence())

        result = self.generate()

        self.assertEqual(result.returncode, 0, result.stderr)
        report = (self.root / "submission" / "model-testing-report.html").read_text(
            encoding="utf-8"
        )
        manifest = json.loads(
            (self.root / "submission" / "model-testing-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(manifest["complete"])
        self.assertTrue(all(item["status"] == "present" for item in manifest["checks"]))
        self.assertIn("data:image/png;base64,", report)
        self.assertIn("download=\"lab4.ipynb\"", report)
        self.assertIn("Synthetic test case 9", report)
        self.assertEqual(manifest["identifiers"]["wandb_run_id"], "run123")

    def test_writes_incomplete_outputs_for_missing_slice_and_stress_evidence(self) -> None:
        evidence = complete_evidence()
        evidence["slice_metrics"] = evidence["slice_metrics"][:-2]
        evidence["stress_test_results"] = evidence["stress_test_results"][:-2]
        write_notebook(self.notebook, evidence)

        result = self.generate()

        self.assertEqual(result.returncode, 1)
        manifest = json.loads(
            (self.root / "submission" / "model-testing-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        checks = {item["name"]: item["status"] for item in manifest["checks"]}
        self.assertEqual(checks["labeled_slice_metrics"], "missing")
        self.assertEqual(checks["complete_stress_test_results"], "missing")

    def test_flags_secret_and_does_not_copy_it_to_generated_files(self) -> None:
        secret = "a" * 40
        write_notebook(
            self.notebook,
            complete_evidence(),
            source=f'wandb.login(key="{secret}")',
        )

        result = self.generate()

        self.assertEqual(result.returncode, 1)
        report = (self.root / "submission" / "model-testing-report.html").read_text(
            encoding="utf-8"
        )
        manifest_text = (
            self.root / "submission" / "model-testing-manifest.json"
        ).read_text(encoding="utf-8")
        self.assertNotIn(secret, report + manifest_text)
        self.assertIn("embedding was withheld", report)
        manifest = json.loads(manifest_text)
        checks = {item["name"]: item["status"] for item in manifest["checks"]}
        self.assertEqual(checks["obvious_credential_leakage"], "missing")

    def test_rejects_missing_input_without_creating_outputs(self) -> None:
        result = self.generate()

        self.assertEqual(result.returncode, 2)
        self.assertIn("Notebook not found", result.stderr)
        self.assertFalse((self.root / "submission").exists())


if __name__ == "__main__":
    unittest.main()
