#!/usr/bin/env python3
"""Generate the Lab 4 model-testing report and machine-readable manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit


REPORT_NAME = "model-testing-report.html"
MANIFEST_NAME = "model-testing-manifest.json"
EVIDENCE_START = "MLIP_SUBMISSION_EVIDENCE_V1"
EVIDENCE_END = "MLIP_SUBMISSION_EVIDENCE_END"
MODELS = {"baseline_model", "candidate_model"}
LABELS = {"negative", "neutral", "positive"}
REQUIRED_WANDB_ARTIFACTS = {
    "predictions_table",
    "slice_metrics",
    "regression_metrics",
    "df_eval",
    "synthetic_tests",
}
SUPPORTED_EVIDENCE = {
    ".csv": "text/csv",
    ".html": "text/html",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".webp": "image/webp",
}
PLACEHOLDERS = {"", "...", "todo", "tbd", "replace me", "your answer here"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Lab 4 model-testing HTML report and JSON manifest."
    )
    parser.add_argument("--learner", required=True, help="Learner name shown in the report")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("lab4.ipynb"),
        help="Executed notebook (default: lab4.ipynb)",
    )
    parser.add_argument(
        "--wandb-evidence",
        type=Path,
        required=True,
        help="W&B table export or screenshot (CSV, HTML, JSON, PDF, PNG, JPEG, or WebP)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("submission"),
        help="Output directory (default: submission)",
    )
    return parser.parse_args()


def check(
    name: str,
    present: bool,
    location: str,
    identifier: str,
    detail: str,
) -> dict[str, str]:
    return {
        "name": name,
        "status": "present" if present else "missing",
        "location": location,
        "identifier": identifier,
        "detail": detail,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_text(output: dict[str, Any]) -> str:
    value: Any = ""
    if output.get("output_type") == "stream":
        value = output.get("text", "")
    elif output.get("output_type") in {"display_data", "execute_result"}:
        value = output.get("data", {}).get("text/plain", "")
    elif output.get("output_type") == "error":
        value = "\n".join(output.get("traceback", []))
    return "".join(value) if isinstance(value, list) else str(value)


def extract_submission_evidence(notebook: dict[str, Any]) -> tuple[Optional[dict[str, Any]], str]:
    transcript = "\n".join(
        output_text(output)
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
    )
    matches = re.findall(
        rf"{re.escape(EVIDENCE_START)}\s*(.*?)\s*{re.escape(EVIDENCE_END)}",
        transcript,
        flags=re.DOTALL,
    )
    if not matches:
        return None, "The notebook does not contain the final structured evidence output."
    try:
        value = json.loads(matches[-1])
    except json.JSONDecodeError as exc:
        return None, f"The final structured evidence output is not valid JSON: {exc.msg}."
    if not isinstance(value, dict):
        return None, "The final structured evidence output must be a JSON object."
    return value, "Found and parsed the final structured evidence output."


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def valid_metric_rows(
    rows: Any, grouping_key: Optional[str] = None
) -> tuple[bool, set[str], str]:
    if not isinstance(rows, list):
        return False, set(), "Expected a list of metric rows."
    groups: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        if not isinstance(row, dict):
            return False, set(), "Every metric row must be an object."
        model = row.get("model")
        accuracy = row.get("accuracy")
        raw_group = row.get(grouping_key) if grouping_key else "overall"
        group = str(raw_group).strip()
        if not group:
            return False, set(groups), "Every slice metric needs a nonempty slice name."
        if model not in MODELS or not is_number(accuracy) or not 0 <= accuracy <= 1:
            return False, set(groups), "Metrics need a recognized model and accuracy from 0 to 1."
        groups[group].add(model)
    complete_groups = {name for name, models in groups.items() if models == MODELS}
    exactly_one_per_model = len(rows) == len(groups) * len(MODELS)
    complete = bool(rows) and len(complete_groups) == len(groups) and exactly_one_per_model
    if complete:
        detail = ""
    elif len(complete_groups) != len(groups):
        detail = "Every metric group needs both baseline_model and candidate_model."
    else:
        detail = "Each model must appear exactly once in every metric group."
    return (
        complete,
        complete_groups,
        detail,
    )


def valid_notes(notes: Any, slice_names: set[str]) -> bool:
    if not isinstance(notes, list) or len(notes) < len(slice_names):
        return False
    cleaned = [str(note).strip().lower() for note in notes]
    return all(note not in PLACEHOLDERS and len(note) >= 12 for note in cleaned)


def valid_hypothesis(value: Any) -> bool:
    cleaned = str(value or "").strip()
    return cleaned.lower() not in PLACEHOLDERS and len(cleaned) >= 20


def validate_stress_rows(rows: Any) -> tuple[bool, list[dict[str, Any]], str]:
    if not isinstance(rows, list):
        return False, [], "Expected a list of stress-test result rows."
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            return False, [], "Every stress-test result must be an object."
        text = str(row.get("text", "")).strip()
        expected = row.get("expected_label")
        model = row.get("model")
        prediction = row.get("pred")
        confidence = row.get("conf")
        correct = row.get("correct")
        if (
            not text
            or expected not in LABELS
            or model not in MODELS
            or prediction not in LABELS
            or not is_number(confidence)
            or not 0 <= confidence <= 1
            or not isinstance(correct, bool)
            or correct != (prediction == expected)
        ):
            return False, [], "Each result needs valid labels, model, confidence, and correctness."
        grouped[text].append(row)
    if len(grouped) != 10:
        return False, [], f"Expected 10 unique cases; found {len(grouped)}."
    if any({row["model"] for row in case_rows} != MODELS for case_rows in grouped.values()):
        return False, [], "Each case needs one baseline and one candidate result."
    if any(len(case_rows) != 2 for case_rows in grouped.values()):
        return False, [], "Each case must appear exactly once for each model."
    return True, rows, "Found 10 cases with labeled baseline and candidate scores."


def sanitize_wandb_url(value: Any) -> tuple[str, bool, str]:
    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return "invalid-url-redacted", False, "none"
    host = (parsed.hostname or "").lower()
    safe = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    parts = [part for part in parsed.path.split("/") if part]
    is_valid = (
        parsed.scheme == "https"
        and host in {"wandb.ai", "www.wandb.ai"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and "runs" in parts
        and parts.index("runs") < len(parts) - 1
    )
    run_id = parts[parts.index("runs") + 1] if is_valid else "none"
    return safe, is_valid, run_id


SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:wandb[_-]?api[_-]?key|api[_-]?key|access[_-]?token)\b\s*[:=]\s*['\"]([A-Za-z0-9_\-]{20,})['\"]"),
    re.compile(r"(?i)wandb\.login\s*\([^)]*\bkey\s*=\s*['\"]([^'\"]{20,})['\"]"),
    re.compile(r"\b(?:sk|ghp|github_pat|hf)_[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)https://[^\s/@:]+:[^\s/@]+@"),
    re.compile(r"(?i)https://[^\s/@]+@"),
    re.compile(r"(?i)[?&](?:token|api[_-]?key|access[_-]?token)=([^&\s'\"]{8,})"),
)


def has_obvious_secret(notebook: dict[str, Any], evidence_path: Path) -> bool:
    chunks = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        chunks.append("".join(source) if isinstance(source, list) else str(source))
        chunks.extend(output_text(output) for output in cell.get("outputs", []))
    searchable = "\n".join(chunks)
    if evidence_path.suffix.lower() in {".csv", ".html", ".json"}:
        searchable += "\n" + evidence_path.read_text(encoding="utf-8", errors="ignore")
    return any(pattern.search(searchable) for pattern in SECRET_PATTERNS)


def data_uri(path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def valid_evidence_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    content = path.read_bytes()
    if not content:
        return False
    if suffix == ".png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {".jpg", ".jpeg"}:
        return content.startswith(b"\xff\xd8\xff")
    if suffix == ".webp":
        return len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP"
    if suffix == ".pdf":
        return content.startswith(b"%PDF-")
    text_content = content.decode("utf-8", errors="replace")
    if suffix == ".json":
        try:
            json.loads(text_content)
        except json.JSONDecodeError:
            return False
        return True
    if suffix == ".csv":
        lines = [line for line in text_content.splitlines() if line.strip()]
        return len(lines) >= 2 and "," in lines[0]
    if suffix == ".html":
        return re.search(r"(?i)<(?:html|body|table)\b", text_content) is not None
    return False


def metric_table(rows: list[dict[str, Any]], include_slice: bool) -> str:
    heading = "<th>Slice</th>" if include_slice else ""
    body = []
    sort_keys = ("slice", "model") if include_slice else ("model",)
    rows = [row for row in rows if isinstance(row, dict)]
    for row in sorted(rows, key=lambda item: tuple(str(item.get(key, "")) for key in sort_keys)):
        slice_cell = f"<td>{html.escape(str(row.get('slice', '')))}</td>" if include_slice else ""
        accuracy = row.get("accuracy")
        accuracy_text = format(float(accuracy), ".4f") if is_number(accuracy) else str(accuracy or "missing")
        body.append(
            f"<tr>{slice_cell}<td>{html.escape(str(row.get('model', '')))}</td>"
            f"<td>{html.escape(accuracy_text)}</td></tr>"
        )
    return (
        f"<table><thead><tr>{heading}<th>Model</th><th>Accuracy</th></tr></thead>"
        f"<tbody>{''.join(body) or '<tr><td colspan=\"3\">No valid metrics found.</td></tr>'}</tbody></table>"
    )


def stress_table(rows: list[dict[str, Any]]) -> str:
    grouped: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    expected: dict[str, str] = {}
    for row in rows:
        text = str(row.get("text", ""))
        grouped[text][str(row.get("model", ""))] = row
        expected[text] = str(row.get("expected_label", ""))
    rendered = []
    for index, (case_text, model_rows) in enumerate(grouped.items(), start=1):
        cells = []
        for model in ("baseline_model", "candidate_model"):
            row = model_rows.get(model, {})
            score = row.get("conf")
            score_text = format(float(score), ".4f") if is_number(score) else "missing"
            correctness = "correct" if row.get("correct") is True else "incorrect"
            cells.append(
                f"<td>{html.escape(str(row.get('pred', 'missing')))}<br>"
                f"confidence {html.escape(score_text)}; {correctness}</td>"
            )
        rendered.append(
            f"<tr><td>{index}</td><td>{html.escape(case_text)}</td>"
            f"<td>{html.escape(expected.get(case_text, ''))}</td>{''.join(cells)}</tr>"
        )
    return (
        "<table><thead><tr><th>#</th><th>Case</th><th>Expected</th>"
        "<th>Baseline result</th><th>Candidate result</th></tr></thead>"
        f"<tbody>{''.join(rendered) or '<tr><td colspan=\"5\">No valid results found.</td></tr>'}</tbody></table>"
    )


def render_wandb_evidence(path: Path, mime_type: str) -> str:
    uri = data_uri(path, mime_type)
    suffix = path.suffix.lower()
    if mime_type.startswith("image/"):
        return f'<img class="evidence" src="{uri}" alt="W&amp;B exported evidence">'
    if suffix in {".csv", ".json", ".html"}:
        content = path.read_text(encoding="utf-8", errors="replace")
        return f"<pre>{html.escape(content)}</pre>"
    if suffix == ".pdf":
        return f'<object class="pdf" data="{uri}" type="application/pdf">Embedded W&amp;B PDF evidence.</object>'
    return "<p>Unsupported W&amp;B evidence format.</p>"


def render_report(
    learner: str,
    generated_at: str,
    complete: bool,
    checks: list[dict[str, str]],
    notebook_path: Path,
    notebook_mime: str,
    wandb_path: Path,
    wandb_mime: str,
    evidence: dict[str, Any],
    safe_wandb_url: str,
    wandb_url_valid: bool,
    safe_to_embed: bool,
) -> str:
    check_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td class=\"{item['status']}\">{item['status']}</td>"
        f"<td>{html.escape(item['identifier'])}</td>"
        f"<td>{html.escape(item['detail'])}</td>"
        "</tr>"
        for item in checks
    )
    overall_rows = evidence.get("overall_metrics", [])
    slice_rows = evidence.get("slice_metrics", [])
    stress_rows = evidence.get("stress_test_results", [])
    notes = evidence.get("slice_notes", [])
    notes_html = "".join(f"<li>{html.escape(str(note))}</li>" for note in notes)
    wandb_link = (
        f'<a href="{html.escape(safe_wandb_url)}">{html.escape(safe_wandb_url)}</a>'
        if wandb_url_valid
        else html.escape(safe_wandb_url)
    )
    notebook_download = (
        f'<a download="{html.escape(notebook_path.name)}" href="{data_uri(notebook_path, notebook_mime)}">'
        "Download the complete executed notebook embedded in this report</a>."
        if safe_to_embed
        else "Notebook embedding was withheld because a possible plaintext credential was detected."
    )
    wandb_content = (
        render_wandb_evidence(wandb_path, wandb_mime)
        if safe_to_embed
        else "<p>W&amp;B evidence embedding was withheld because a possible plaintext credential was detected.</p>"
    )
    safety_text = (
        "The checker found no obvious plaintext API key or access-token pattern. "
        "It cannot inspect secrets visible inside an image, so review the report before uploading it."
        if safe_to_embed
        else "The checker found a possible plaintext credential. Learner-provided evidence was withheld from this report; remove and revoke the credential, then regenerate."
    )
    overall_status = "complete" if complete else "incomplete"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lab 4 Model-Testing Submission Report</title>
  <style>
    body {{ color: #17202a; font: 16px/1.45 system-ui, sans-serif; margin: 2rem auto; max-width: 78rem; padding: 0 1rem; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; margin: .75rem 0 1.5rem; width: 100%; }}
    th, td {{ border: 1px solid #bcccdc; padding: .55rem; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    pre {{ background: #f5f7fa; border: 1px solid #bcccdc; max-height: 32rem; overflow: auto; padding: 1rem; white-space: pre-wrap; }}
    code {{ overflow-wrap: anywhere; }}
    .present, .complete {{ color: #176b3a; font-weight: 700; }}
    .missing, .incomplete {{ color: #a61b1b; font-weight: 700; }}
    .note {{ background: #fffbea; border-left: .3rem solid #d69e2e; padding: .7rem 1rem; }}
    .evidence {{ border: 1px solid #bcccdc; height: auto; max-width: 100%; }}
    .pdf {{ height: 48rem; width: 100%; }}
  </style>
</head>
<body>
  <h1>Lab 4: Model Testing with W&amp;B and LLMs</h1>
  <table>
    <tr><th>Learner</th><td>{html.escape(learner)}</td></tr>
    <tr><th>Generated</th><td>{html.escape(generated_at)}</td></tr>
    <tr><th>Overall completeness</th><td class="{overall_status}">{overall_status}</td></tr>
    <tr><th>W&amp;B run</th><td>{wandb_link}</td></tr>
    <tr><th>Checker version</th><td>1.0</td></tr>
  </table>

  <h2>Completeness</h2>
  <table>
    <thead><tr><th>Required evidence</th><th>Status</th><th>Identifier</th><th>Detail</th></tr></thead>
    <tbody>{check_rows}</tbody>
  </table>

  <h2>Overall metrics</h2>
  {metric_table(overall_rows if isinstance(overall_rows, list) else [], False)}

  <h2>Slice metrics</h2>
  {metric_table(slice_rows if isinstance(slice_rows, list) else [], True)}

  <h2>Saved slice notes</h2>
  <ul>{notes_html or '<li>No saved slice notes found.</li>'}</ul>

  <h2>Targeted stress test</h2>
  <p><strong>Hypothesis:</strong> {html.escape(str(evidence.get('stress_test_hypothesis', 'missing')))}</p>
  {stress_table(stress_rows if isinstance(stress_rows, list) else [])}

  <h2>W&amp;B export or screenshot</h2>
  <p><code>{html.escape(str(wandb_path))}</code></p>
  {wandb_content}

  <h2>Raw audit evidence</h2>
  <p>{notebook_download}</p>
  <p>The W&amp;B evidence above and the embedded notebook are copies captured when this report was generated.</p>
  <p class="note">This checker verifies objective completeness and internal consistency only. Complete the slice and deployment interpretation spot checks separately in Canvas.</p>

  <h2>Safety check</h2>
  <p>{safety_text}</p>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    if not args.notebook.is_file():
        print(f"Notebook not found: {args.notebook}", file=sys.stderr)
        return 2
    if not args.wandb_evidence.is_file():
        print(f"W&B evidence not found: {args.wandb_evidence}", file=sys.stderr)
        return 2

    evidence_suffix = args.wandb_evidence.suffix.lower()
    if evidence_suffix not in SUPPORTED_EVIDENCE:
        print(
            "W&B evidence must be CSV, HTML, JSON, PDF, PNG, JPEG, or WebP.",
            file=sys.stderr,
        )
        return 2
    try:
        notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read notebook JSON: {exc}", file=sys.stderr)
        return 2
    if (
        not isinstance(notebook, dict)
        or not isinstance(notebook.get("cells"), list)
        or not all(isinstance(cell, dict) for cell in notebook["cells"])
    ):
        print("Notebook JSON does not contain a cells list.", file=sys.stderr)
        return 2

    evidence, evidence_detail = extract_submission_evidence(notebook)
    evidence = evidence or {}
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    executed_count = sum(cell.get("execution_count") is not None for cell in code_cells)
    all_executed = bool(code_cells) and executed_count == len(code_cells)
    error_count = sum(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    safe_wandb_url, wandb_url_valid, wandb_run_id = sanitize_wandb_url(
        evidence.get("wandb_run_url")
    )

    overall_valid, overall_groups, overall_detail = valid_metric_rows(
        evidence.get("overall_metrics")
    )
    slice_valid, slice_names, slice_detail = valid_metric_rows(
        evidence.get("slice_metrics"), grouping_key="slice"
    )
    slice_valid = slice_valid and len(slice_names) >= 5
    stress_valid, stress_rows, stress_detail = validate_stress_rows(
        evidence.get("stress_test_results")
    )
    artifacts = evidence.get("wandb_artifacts_logged")
    artifact_set = set(artifacts) if isinstance(artifacts, list) and all(isinstance(item, str) for item in artifacts) else set()
    missing_artifacts = sorted(REQUIRED_WANDB_ARTIFACTS - artifact_set)
    notes_valid = valid_notes(evidence.get("slice_notes"), slice_names)
    hypothesis_valid = valid_hypothesis(evidence.get("stress_test_hypothesis"))
    has_secret = has_obvious_secret(notebook, args.wandb_evidence)
    evidence_valid = valid_evidence_file(args.wandb_evidence)
    overall_models = {
        row.get("model")
        for row in evidence.get("overall_metrics", [])
        if isinstance(row, dict)
    } if isinstance(evidence.get("overall_metrics"), list) else set()

    report_location = REPORT_NAME
    checks = [
        check(
            "learner_name",
            bool(args.learner.strip()),
            report_location,
            args.learner.strip() or "missing",
            "Learner name recorded." if args.learner.strip() else "Provide a nonempty learner name.",
        ),
        check(
            "executed_notebook",
            all_executed,
            "embedded notebook",
            f"{executed_count}/{len(code_cells)} code cells",
            "Every code cell has an execution count." if all_executed else "Run all notebook cells in order and save the notebook.",
        ),
        check(
            "structured_notebook_evidence",
            evidence.get("schema_version") == "1.0",
            report_location,
            str(evidence.get("schema_version", "none")),
            evidence_detail,
        ),
        check(
            "notebook_error_outputs",
            error_count == 0,
            "embedded notebook",
            f"{error_count} error outputs",
            "No saved error outputs were found." if error_count == 0 else "Resolve saved cell errors, rerun all cells, and save the notebook.",
        ),
        check(
            "wandb_run_identifier",
            wandb_url_valid,
            report_location,
            wandb_run_id,
            "Found a credential-free W&B run URL." if wandb_url_valid else "Expected an HTTPS wandb.ai run URL without credentials or query parameters.",
        ),
        check(
            "wandb_export_or_screenshot",
            evidence_valid,
            report_location,
            f"{args.wandb_evidence.name} ({args.wandb_evidence.stat().st_size} bytes)",
            "Embedded a recognized local W&B export or screenshot; the checker did not contact W&B." if evidence_valid else "The file is empty or its contents do not match the selected supported format.",
        ),
        check(
            "required_wandb_artifacts",
            not missing_artifacts,
            report_location,
            ", ".join(sorted(artifact_set)) or "none",
            "All required logged artifact names are recorded." if not missing_artifacts else "Missing logged artifact names: " + ", ".join(missing_artifacts),
        ),
        check(
            "labeled_overall_metrics",
            overall_valid and overall_groups == {"overall"},
            report_location,
            ", ".join(sorted(str(model) for model in overall_models)) or "none",
            "Found baseline and candidate accuracy from 0 to 1." if overall_valid else overall_detail,
        ),
        check(
            "labeled_slice_metrics",
            slice_valid,
            report_location,
            f"{len(slice_names)} complete slices",
            "Found at least five slices with baseline and candidate accuracy." if slice_valid else (slice_detail or "At least five slices need both model accuracies."),
        ),
        check(
            "saved_slice_notes",
            notes_valid,
            report_location,
            f"{len(evidence.get('slice_notes', [])) if isinstance(evidence.get('slice_notes'), list) else 0} notes",
            "Found a non-placeholder note for every reported slice." if notes_valid else "Save a substantive note for every reported slice.",
        ),
        check(
            "stress_test_hypothesis",
            hypothesis_valid,
            report_location,
            "recorded" if hypothesis_valid else "missing",
            "Found a targeted stress-test hypothesis." if hypothesis_valid else "Replace the hypothesis placeholder with your tested hypothesis.",
        ),
        check(
            "complete_stress_test_results",
            stress_valid,
            report_location,
            "10 cases × 2 models" if stress_valid else "incomplete",
            stress_detail,
        ),
        check(
            "obvious_credential_leakage",
            not has_secret,
            report_location,
            "none detected" if not has_secret else "possible secret detected",
            "No obvious plaintext secret pattern was found." if not has_secret else "Remove the plaintext credential from the notebook or text export and revoke it before submission.",
        ),
    ]

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    complete = all(item["status"] == "present" for item in checks)
    notebook_sha = sha256(args.notebook)
    wandb_sha = sha256(args.wandb_evidence)
    safe_evidence = evidence if not has_secret else {}
    manifest = {
        "schema_version": "1.0",
        "checker_version": "1.0",
        "lab": "Lab 4: Model Testing with W&B and LLMs",
        "learner": args.learner,
        "generated_at": generated_at,
        "complete": complete,
        "report": REPORT_NAME,
        "identifiers": {
            "wandb_run_url": safe_wandb_url,
            "wandb_run_id": wandb_run_id,
            "notebook_sha256": notebook_sha,
            "wandb_evidence_sha256": wandb_sha,
        },
        "artifacts": {
            "notebook": str(args.notebook),
            "wandb_evidence": str(args.wandb_evidence),
        },
        "evidence": {
            "overall_metrics": safe_evidence.get("overall_metrics", []),
            "slice_metrics": safe_evidence.get("slice_metrics", []),
            "slice_notes": safe_evidence.get("slice_notes", []),
            "stress_test_hypothesis": safe_evidence.get("stress_test_hypothesis", ""),
            "stress_test_results": stress_rows if not has_secret else [],
            "wandb_artifacts_logged": sorted(artifact_set) if not has_secret else [],
        },
        "checks": checks,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / REPORT_NAME
    manifest_path = args.output_dir / MANIFEST_NAME
    notebook_mime = mimetypes.guess_type(args.notebook.name)[0] or "application/x-ipynb+json"
    wandb_mime = SUPPORTED_EVIDENCE[evidence_suffix]
    report_path.write_text(
        render_report(
            args.learner,
            generated_at,
            complete,
            checks,
            args.notebook,
            notebook_mime,
            args.wandb_evidence,
            wandb_mime,
            safe_evidence,
            safe_wandb_url,
            wandb_url_valid,
            not has_secret,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"Wrote {manifest_path}")
    print(
        "Submission evidence is complete."
        if complete
        else "Submission evidence is incomplete; review the missing checks."
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
