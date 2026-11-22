"""Kaggle V9 / STEP 18: one-command end-to-end experiment completion and integrity audit.

Install at <Brain_Tumor_Reproduction_V9>/src/step18_final_audit.py, run:
  ./.venv/bin/python -m src.step18_final_audit

By default, runs ONLY MISSING post-hoc steps (12-17), verifies saved artifacts and
image/file joins, and writes a timestamped audit report. Does not retrain, rewrite
old results, change a checkpoint or edit MRI images. For validation only:
  ./.venv/bin/python -m src.step18_final_audit --check-only
For slower full SHA256 validation of the source images:
  ./.venv/bin/python -m src.step18_final_audit --rehash-source

This is a technical audit, NOT patient-independent or clinical validation.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from . import config as C
from .common import sha_file
from .verify import verify


class Audit:
    def __init__(self) -> None:
        self.entries: list[dict[str, str]] = []
        self.runs: list[dict[str, str]] = []
        self.fingerprint: str | None = None
        self.manifest: list[dict[str, str]] = []
        self.metrics: dict[str, Any] = {}

    def add(self, status: str, name: str, details: str = "") -> None:
        assert status in ("PASS", "WARN", "FAIL")
        self.entries.append({"status": status, "check": name, "details": str(details)})
        print(f"[{status:4}] {name}" + (f" — {details}" if details else ""), flush=True)

    def read_json(self, relative: str, *, required: bool = True) -> Any:
        file = C.OUTPUT / relative
        if not file.is_file():
            if required:
                self.add("FAIL", f"Missing: {relative}")
            return None
        try:
            result = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            return result
        except (OSError, ValueError) as exc:
            self.add("FAIL", f"Invalid JSON: {relative}", str(exc))
            return None

    def read_csv(self, relative: str, *, required: bool = True) -> list[dict[str, str]]:
        file = C.OUTPUT / relative
        if not file.is_file():
            if required:
                self.add("FAIL", f"Missing: {relative}")
            return []
        try:
            with file.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                self.add("FAIL", f"Empty CSV: {relative}")
            return rows
        except (OSError, csv.Error, UnicodeError) as exc:
            self.add("FAIL", f"Unreadable CSV: {relative}", str(exc))
            return []


def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def keyset(rows: list[dict[str, str]]) -> set[str]:
    return {r.get("file", "") for r in rows if r.get("file")}


def validate_checkpoint_config(a: Audit) -> bool:
    training = a.read_json("07_training/experiment_config.json")
    test = a.read_json("08_test/test_metrics.json")
    a.read_json("07_training/training_summary.json")
    check = C.CHECKPOINT
    if not check.is_file():
        a.add("FAIL", "Trained model checkpoint missing", str(check))
        return False
    a.add("PASS", "Trained checkpoint exists", f"{check.stat().st_size / (1024*1024):.1f} MiB")
    ok = True
    for name, m in (("Training", training), ("Test", test)):
        if not m:
            ok = False
        elif m.get("split_fingerprint") != a.fingerprint:
            ok = False
            a.add("FAIL", f"{name} split fingerprint", "Does not match current dataset split")
        else:
            a.add("PASS", f"{name} split fingerprint", "Matches the current split")
    if test:
        split_count = sum(r["split"] == "test" for r in a.manifest)
        n = test.get("test_count")
        correct = test.get("correct")
        acc = test.get("accuracy")
        if not (isinstance(n, int) and n == split_count and isinstance(correct, int)
                and 0 <= correct <= n and is_finite(acc) and abs(float(acc) - correct / n) < 1e-5):
            ok = False
            a.add("FAIL", "Held-out test metrics", "Count, accuracy, or correct predictions inconsistent with split")
        else:
            a.add("PASS", "Held-out test metrics", f"{correct}/{n} = {float(acc):.4%}")
            a.metrics["test_accuracy"] = float(acc)
            a.metrics["test_count"] = n
        if is_finite(test.get("macro_f1")):
            a.metrics["test_macro_f1"] = float(test["macro_f1"])
    return ok


def verify_dataset(a: Audit, rehash: bool) -> bool:
    try:
        rows, fingerprint = verify(rehash=rehash, detail=False)
        a.manifest = rows
        a.fingerprint = fingerprint
        splits = Counter(r["split"] for r in rows)
        a.metrics["dataset_splits"] = dict(splits)
        a.metrics["dataset_unique_representatives"] = len(rows)
        a.add("PASS", "Dataset grouping and split isolation",
              f"{len(rows)} images; train={splits['train']}, val={splits['val']}, test={splits['test']}; "
              "no exact/flagged-near group crosses splits")
        a.add("PASS", "Dataset fingerprint", fingerprint)
        if rehash:
            a.add("PASS", "Original file SHA256 hashes", "Every split representative matches the inventory")
        else:
            a.add("WARN", "Original file SHA256 hashes", "Not recomputed (use --rehash-source for full source-file check)")
        return True
    except Exception as exc:
        a.add("FAIL", "Dataset verification failed", f"{type(exc).__name__}: {exc}")
        return False


def audit_core_artifacts(a: Audit) -> None:
    files = [
        ("01 Inventory", "01_inventory/image_manifest.csv"),
        ("02 Exact duplicates", "02_exact_duplicates/dedup_manifest.csv"),
        ("03 Near-duplicate groups", "03_near_duplicates/near_groups.csv"),
        ("04 Frozen dataset splits", "04_split/split_manifest.csv"),
        ("05 Dataset verification", "05_verification/verification.json"),
        ("06 Sample preview", "06_preview/split_samples.png"),
        ("07 Training history", "07_training/train_history.csv"),
        ("08 Test metrics", "08_test/test_metrics.json"),
        ("09 MC Dropout per-MRI results", "09_uncertainty/uncertainty.csv"),
        ("11 Research methodology draft", "11_report/V9_methodology_and_results.md"),
    ]
    for label, rel in files:
        path = C.OUTPUT / rel
        a.add("PASS" if path.is_file() and path.stat().st_size > 0 else "FAIL", label,
              "Saved output found" if path.is_file() and path.stat().st_size > 0 else f"Missing/empty: {rel}")
    cam_images = list((C.OUTPUT / "10_gradcam").glob("*_gradcam.png"))
    a.add("PASS" if cam_images else "FAIL", "10 Grad-CAM examples", f"{len(cam_images)} PNG figures found")
    a.add("WARN", "Manual MRI review", "Image quality and medically correct heatmap localization cannot be verified by this script")


def audit_mc_and_calibration(a: Audit) -> None:
    manifest_test = {r["file"] for r in a.manifest if r["split"] == "test"}
    for label, relative in (("09 Prediction MC Dropout", "09_uncertainty/uncertainty.csv"),
                            ("12 Calibrated predictions", "12_calibration/per_image_test_calibration.csv")):
        rows = a.read_csv(relative)
        if not rows:
            continue
        files = [r.get("file", "") for r in rows]
        if len(files) != len(set(files)) or set(files) != manifest_test:
            a.add("FAIL", f"{label} IDs", f"Rows={len(files)}, unique={len(set(files))}, expected={len(manifest_test)}")
        else:
            a.add("PASS", f"{label} IDs", f"Exactly {len(manifest_test)} known test MRIs")
        class_index = {r["file"]: r["class_name"] for r in a.manifest if r["split"] == "test"}
        wrong_labels = [r.get("file") for r in rows if r.get("true_class") != class_index.get(r.get("file"))]
        if wrong_labels:
            a.add("FAIL", f"{label} ground-truth labels", f"{len(wrong_labels)} inconsistent")
        else:
            a.add("PASS", f"{label} ground-truth labels")
    cal = a.read_json("12_calibration/calibration_summary.json")
    mc = a.read_json("09_uncertainty/mc_summary.json")
    for title, summary in (("09 MC uncertainty", mc), ("12 Temperature calibration", cal)):
        if summary:
            a.add("PASS" if summary.get("split_fingerprint") == a.fingerprint else "FAIL",
                  f"{title} fingerprint", "Matches split" if summary.get("split_fingerprint") == a.fingerprint else "Mismatch")
    if cal:
        if cal.get("temperature_fit_on") == "validation_only" and is_finite(cal.get("temperature")) and float(cal["temperature"]) > 0:
            a.add("PASS", "Temperature fitting set", "Validation only; temperature is positive")
        else:
            a.add("FAIL", "Temperature fitting set", "Expected validation-only fit")
        try:
            raw = cal["calibration"]["test_raw"]
            scaled = cal["calibration"]["test_temperature_scaled"]
            if is_finite(raw.get("ece")) and is_finite(scaled.get("ece")):
                a.metrics["test_ece_raw"] = float(raw["ece"])
                a.metrics["test_ece_calibrated"] = float(scaled["ece"])
                a.add("PASS", "Calibration ECE available", f"Raw {float(raw['ece']):.5f} → scaled {float(scaled['ece']):.5f}")
            if is_finite(raw.get("accuracy")) and is_finite(scaled.get("accuracy")) and abs(float(raw["accuracy"]) - float(scaled["accuracy"])) < 1e-9:
                a.add("PASS", "Calibration preserves predicted class accuracy")
            else:
                a.add("FAIL", "Calibration changed accuracy", "Scalar temperature must not change argmax for T>0")
        except (KeyError, TypeError) as exc:
            a.add("FAIL", "Calibration metrics structure", str(exc))
    if mc and is_finite(mc.get("mc_test_accuracy")):
        a.metrics["mc_test_accuracy"] = float(mc["mc_test_accuracy"])
        a.add("PASS", "Prediction uncertainty metrics available", f"MC accuracy {float(mc['mc_test_accuracy']):.4%}")


def audit_experiment_samples(a: Audit) -> set[str]:
    test_set = {r["file"] for r in a.manifest if r["split"] == "test"}
    mapping = {r["file"]: r["class_name"] for r in a.manifest}
    rels = {
        "13 Faithfulness": "13_faithfulness_test/faithfulness_trials.csv",
        "14 Stability": "14_stability_test/stability_trials.csv",
        "15 Explanation uncertainty": "15_explanation_uncertainty_test_final/explanation_uncertainty.csv",
    }
    datasets: dict[str, set[str]] = {}
    for label, rel in rels.items():
        rows = a.read_csv(rel)
        names = keyset(rows)
        datasets[label] = names
        if not names:
            continue
        invalid = names - test_set
        counts = Counter(mapping.get(name) for name in names)
        wrong_true = [r for r in rows if r.get("true_class") != mapping.get(r.get("file"))]
        if invalid or wrong_true:
            a.add("FAIL", f"{label} sample membership", f"Unknown test IDs={len(invalid)}, incorrect labels={len(wrong_true)}")
        else:
            a.add("PASS", f"{label} sample membership", f"{len(names)} valid test MRIs; class counts={dict(counts)}")
        # Expected one row for each condition. Avoid silently treating incomplete runs as complete.
        if label.startswith("13"):
            expected = len(names) * 2 * 3 * 2 * 3
            fields = {"final", "earlier"}
            ok = len(rows) == expected and fields == {r.get("layer") for r in rows}
        elif label.startswith("14"):
            expected = len(names) * 2 * 3
            ok = len(rows) == expected and {r.get("layer") for r in rows} == {"final", "earlier"}
        else:
            expected = len(names)
            ok = len(rows) == expected and all(r.get("layer") == "final" for r in rows)
        a.add("PASS" if ok else "FAIL", f"{label} trial completeness", f"Found {len(rows)} rows; expected {expected}")
    if any(not x for x in datasets.values()):
        a.add("FAIL", "13/14/15 MRI matching", "At least one experiment has no test CSV")
        return set()
    unique = set.union(*datasets.values())
    common = set.intersection(*datasets.values())
    same = len(common) == len(unique) == 16 and all(len(s) == 16 for s in datasets.values())
    a.add("PASS" if same else "FAIL", "13/14/15 MRI matching",
          f"Overlapping={len(common)}; union={len(unique)}; expected 16 identical MRIs (4/class)")
    return common if same else set()


def audit_posthoc_summaries(a: Audit) -> None:
    rels = [
        ("13 Validation faithfulness", "13_faithfulness_val/faithfulness_summary.json"),
        ("13 Test faithfulness", "13_faithfulness_test/faithfulness_summary.json"),
        ("14 Validation stability", "14_stability_val/stability_summary.json"),
        ("14 Test stability", "14_stability_test/stability_summary.json"),
        ("15 Explanation uncertainty", "15_explanation_uncertainty_test_final/uncertainty_summary.json"),
        ("16 Integrated reliability", "16_integrated_test_final/integrated_summary.json"),
    ]
    for label, file in rels:
        summary = a.read_json(file)
        if summary:
            a.add("PASS" if summary.get("split_fingerprint", a.fingerprint) == a.fingerprint else "FAIL",
                  label, "Summary found; split identity checked where included")
        if label.startswith("13 Test") and summary:
            try:
                result = next(r for r in summary["comparison"] if r["layer"] == "final" and r["method"] == "deletion" and abs(float(r["area_fraction"]) - .2) < 1e-6)
                value = float(result["high_minus_random"])
                a.metrics["faithfulness_high_minus_random_20pct"] = value
                a.add("PASS", "20% high-vs-random faithfulness difference recorded", f"{value:+.4f} logit margin (descriptive)")
            except (KeyError, StopIteration, TypeError, ValueError):
                a.add("FAIL", "Faithfulness summary extraction", "Expected final-layer 20% deletion entry missing")
        if label.startswith("14 Test") and summary:
            try:
                rows = [r for r in summary["results"] if r["layer"] == "final"]
                if len(rows) == 3 and all(is_finite(r.get("mean_top20_iou")) for r in rows):
                    a.metrics["stability_top20_iou"] = {r["perturbation"]: float(r["mean_top20_iou"]) for r in rows}
                    a.add("PASS", "Stability IoU values present", ", ".join(f"{r['perturbation']}: {float(r['mean_top20_iou']):.3f}" for r in rows))
                else:
                    a.add("FAIL", "Stability IoU values", "Expected 3 finite test conditions")
            except (KeyError, TypeError):
                a.add("FAIL", "Stability summary shape", "Missing expected results")
        if label.startswith("15") and summary:
            a.metrics["explanation_uncertainty_mean_std"] = summary.get("mean_normalized_spatial_attribution_std")
            if summary.get("mc_passes") != 20:
                a.add("WARN", "Step 15 MC passes", f"Found {summary.get('mc_passes')}; intended 20")


def audit_integration(a: Audit, same_ids: set[str]) -> bool:
    rows = a.read_csv("16_integrated_test_final/per_image_reliability_profile.csv")
    summary = a.read_json("16_integrated_test_final/integrated_summary.json")
    if not rows or not summary:
        return False
    names = keyset(rows)
    ids_ok = bool(same_ids) and names == same_ids and len(rows) == len(same_ids)
    a.add("PASS" if ids_ok else "FAIL", "Integrated per-image MRI join",
          f"{len(rows)} profiles; exactly same MRI IDs as Steps 13–15: {'yes' if ids_ok else 'NO'}")
    if not ids_ok:
        return False
    expected = ["predictive_entropy", "mc_mutual_information", "raw_confidence", "calibrated_confidence",
                "faithfulness_deletion_high_minus_random", "mean_stability_cosine",
                "explanation_shape_uncertainty", "mc_explanation_flip_rate"]
    missing = {k: sum(not is_finite(r.get(k)) for r in rows) for k in expected}
    if any(missing.values()):
        a.add("FAIL", "Complete reliability fields", f"Missing or nonnumeric values: { {k:v for k,v in missing.items() if v} }")
        return False
    a.add("PASS", "Complete reliability fields", f"All eight measurements present for each of {len(rows)} MRIs")
    if summary.get("matched_examples") == len(rows):
        a.add("PASS", "Integrated summary count", f"{len(rows)} matched profiles")
    else:
        a.add("FAIL", "Integrated summary count", f"Expected {len(rows)}, found {summary.get('matched_examples')}")
        return False
    a.metrics["integrated_cases"] = len(rows)
    return True


def run_step(a: Audit, label: str, result_files: list[str], module: str, arguments: list[str], check_only: bool) -> bool:
    existing = [(C.OUTPUT / f).is_file() for f in result_files]
    if all(existing):
        a.runs.append({"step": label, "action": "SKIP", "details": "Already completed; preserved existing files"})
        print(f"\n[SKIP] {label}: existing outputs kept intact", flush=True)
        return True
    if any(existing):
        a.add("FAIL", f"{label}: incomplete existing output", "Some files exist and some are missing. Refusing potential overwrite; inspect manually")
        a.runs.append({"step": label, "action": "BLOCKED", "details": "Partial output; would risk overwriting"})
        return False
    if check_only:
        a.runs.append({"step": label, "action": "NOT RUN", "details": "--check-only selected"})
        a.add("WARN", f"{label}: missing", "Not executed in --check-only mode")
        return False
    command = [sys.executable, "-m", module, *arguments]
    print(f"\n{'='*72}\n[RUN] {label}\n$ {' '.join(command)}\n{'='*72}", flush=True)
    try:
        p = subprocess.run(command, cwd=C.ROOT, check=False)
        if p.returncode == 0 and all((C.OUTPUT / f).is_file() for f in result_files):
            a.runs.append({"step": label, "action": "DONE", "details": "Script exited successfully and output files exist"})
            a.add("PASS", f"{label}: generated", "New results saved")
            return True
        a.add("FAIL", f"{label}: script failed", f"Exit code={p.returncode}; output completeness={[(C.OUTPUT / f).is_file() for f in result_files]}")
        a.runs.append({"step": label, "action": "FAILED", "details": f"Exit code={p.returncode}"})
        return False
    except OSError as exc:
        a.add("FAIL", f"{label}: cannot launch", str(exc))
        a.runs.append({"step": label, "action": "FAILED", "details": str(exc)})
        return False


def write_report(a: Audit, audit_dir: Path, check_only: bool) -> dict[str, Any]:
    audit_dir.mkdir(parents=True, exist_ok=False)
    counts = Counter(e["status"] for e in a.entries)
    total = sum(counts.values())
    technical_status = "FAIL - requires attention" if counts["FAIL"] else "CHECKED - with research limitations"
    warnings = [
        "Research prototype ONLY. Not clinically validated; model output and Grad-CAM may be incorrect even at high confidence.",
        "No proven patient-independent split: patient identifiers were not available.",
        "Near-duplicate detection is threshold-dependent; unknown duplicates or patient overlap may remain.",
        "Test dataset was repeatedly inspected; post-hoc Steps 13–16 are exploratory and not fresh external validation.",
        "Faithfulness and stability interventions are artificial diagnostics, not validation of tumor localization.",
        "Repeated No Tumor background-corner activation and questionable pituitary localization need investigation with clinically annotated scans.",
        "The Step 15 sample contains only 16 images; no clinical reliability thresholds or validated composite safety score.",
        "A final clinical/real-world system needs independent external testing, specialist review, error analysis and clinical governance.",
        "This audit checks saved project artifacts and internal consistency; it does not establish medical correctness.",
        "The Streamlit interface is a separate project deliverable; this audit does not create or validate a deployment interface.",
    ]
    payload = {
        "created_at_local": dt.datetime.now().astimezone().isoformat(),
        "project_root": str(C.ROOT),
        "results_root": str(C.OUTPUT),
        "mode": "check_only" if check_only else "run_missing_then_audit",
        "technical_status": technical_status,
        "checks": dict(counts),
        "metrics": a.metrics,
        "steps": a.runs,
        "entries": a.entries,
        "important_limitations": warnings,
        "not_a_clinical_approval": True,
    }
    (audit_dir / "FINAL_AUDIT.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# KAGGLE V9 — FINAL EXPERIMENT AUDIT", "", f"**Technical result:** {technical_status}",
             f"**Checked:** {total}; PASS={counts['PASS']}, WARN={counts['WARN']}, FAIL={counts['FAIL']}",
             f"**Created:** {payload['created_at_local']}", "",
             "## What this means", "",
             "This verifies technical experiment completion and consistency. It is NOT a clinical validity certificate or a finished clinical product.",
             "", "## Run status", ""]
    for x in a.runs:
        lines.append(f"- **{x['action']}** — {x['step']}: {x['details']}")
    lines.extend(["", "## Key saved measurements", ""])
    if a.metrics:
        for k, v in a.metrics.items():
            lines.append(f"- **{k.replace('_',' ')}:** {v}")
    else:
        lines.append("No measurement results could be verified.")
    lines.extend(["", "## Checks", "", "| Result | Check | Detail |", "|---|---|---|"])
    for e in a.entries:
        def clean(s: str) -> str:
            return s.replace("|", "/").replace("\n", " ")
        lines.append(f"| {e['status']} | {clean(e['check'])} | {clean(e['details'])} |")
    lines.extend(["", "## Scientific and medical limitations", ""])
    for w in warnings:
        lines.append(f"- {w}")
    lines.extend(["", "## Mandatory warning for any future UI", "",
                  "> ⚠️ RESEARCH USE ONLY: This AI model and its explanations may be incorrect, including when confidence is high. This prototype is not clinically validated and must not be used for diagnosis, treatment or independent medical decisions. Consult a qualified medical professional.", ""])
    (audit_dir / "FINAL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    # HTML with simple standalone tables so it opens in a browser without packages.
    def esc(value: Any) -> str:
        return html.escape(str(value))
    colors = {"PASS": "#167a55", "WARN": "#a35c12", "FAIL": "#b33232"}
    html_rows = "\n".join(
        f"<tr><td style='color:{colors[e['status']]};font-weight:700'>{esc(e['status'])}</td><td>{esc(e['check'])}</td><td>{esc(e['details'])}</td></tr>"
        for e in a.entries)
    run_rows = "\n".join(
        f"<tr><td>{esc(x['action'])}</td><td>{esc(x['step'])}</td><td>{esc(x['details'])}</td></tr>" for x in a.runs)
    metric_rows = "\n".join(f"<tr><td>{esc(k.replace('_',' '))}</td><td>{esc(v)}</td></tr>" for k, v in a.metrics.items())
    list_warn = "\n".join(f"<li>{esc(w)}</li>" for w in warnings)
    doc = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>Kaggle V9 | Final experiment audit</title><meta name='viewport' content='width=device-width,initial-scale=1'>
<style>body{{font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#203041;background:#f4f7fa;margin:0}}
main{{max-width:1040px;margin:2.5rem auto;background:white;padding:2rem;border-radius:14px;box-shadow:0 4px 20px #14364a16}}
h1{{margin:0}}h2{{margin-top:2rem}}.kicker{{color:#587080;text-transform:uppercase;letter-spacing:.1em;font-size:12px}}
.status{{background:#f0f4f9;padding:12px 16px;border-left:5px solid {'#b33232' if counts['FAIL'] else '#167a55'};margin:18px 0}}
table{{width:100%;border-collapse:collapse;margin:10px 0}}th,td{{text-align:left;border-bottom:1px solid #e5ebee;padding:10px 8px;vertical-align:top}}th{{background:#f0f4f9}}.warning{{background:#fff6eb;border-left:4px solid #a35c12;padding:14px}}small{{color:#647684}}p{{max-width:850px}}li{{margin:7px 0}}</style></head><body><main>
<p class='kicker'>Research prototype · Internal consistency check</p><h1>Kaggle V9: Final Experiment Audit</h1>
<div class='status'><strong>{esc(technical_status)}</strong><p>PASS {counts['PASS']} &nbsp; WARN {counts['WARN']} &nbsp; FAIL {counts['FAIL']}</p>
<small>Generated {esc(payload['created_at_local'])}</small></div>
<p>This verifies saved results and matching images. It does not certify medical correctness or complete clinical validation.</p>
<h2>Steps and execution</h2><table><tr><th>Action</th><th>Step</th><th>Detail</th></tr>{run_rows}</table>
<h2>Key results</h2><table><tr><th>Metric</th><th>Saved measurement</th></tr>{metric_rows}</table>
<h2>Internal checks</h2><table><tr><th>Status</th><th>Check</th><th>Details</th></tr>{html_rows}</table>
<h2>Important limitations</h2><ul>{list_warn}</ul>
<div class='warning'><strong>⚠️ RESEARCH USE ONLY.</strong> AI predictions and explanations may be wrong, even at high confidence. This system is not clinically validated. Do not use it independently for diagnosis or treatment; obtain assessment from a qualified medical professional.</div>
</main></body></html>"""
    (audit_dir / "FINAL_AUDIT.html").write_text(doc, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Complete missing post-hoc Kaggle V9 steps and audit saved evidence; never retrain.")
    parser.add_argument("--check-only", action="store_true", help="Inspect files only, do not execute missing experiments")
    parser.add_argument("--rehash-source", action="store_true", help="Slow: recompute SHA256 of every original split image")
    args = parser.parse_args()
    print("\n" + "="*78 + "\nV9 FINAL AUDIT — EXISTING CHECKPOINT, NO RETRAINING\n" + "="*78, flush=True)
    print(f"Project: {C.ROOT}\nSaved results: {C.OUTPUT}", flush=True)
    a = Audit()
    checkpoint_start = sha_file(C.CHECKPOINT) if C.CHECKPOINT.is_file() else None
    try:
        dataset_ok = verify_dataset(a, args.rehash_source)
        core_ok = validate_checkpoint_config(a) if dataset_ok else False
        if core_ok:
            # Only missing post-hoc results are generated; prior findings remain immutable.
            steps = [
                ("Step 12 — Calibration", ["12_calibration/calibration_summary.json", "12_calibration/per_image_test_calibration.csv"], "src.step12_calibration", []),
                ("Step 13 — Faithfulness (validation)", ["13_faithfulness_val/faithfulness_summary.json", "13_faithfulness_val/faithfulness_trials.csv"], "src.step13_faithfulness", ["--split", "val", "--n-per-class", "8"]),
                ("Step 14 — Stability (validation)", ["14_stability_val/stability_summary.json", "14_stability_val/stability_trials.csv"], "src.step14_stability", ["--split", "val", "--n-per-class", "8"]),
                ("Step 13 — Faithfulness (test pilot)", ["13_faithfulness_test/faithfulness_summary.json", "13_faithfulness_test/faithfulness_trials.csv"], "src.step13_faithfulness", ["--split", "test", "--n-per-class", "4"]),
                ("Step 14 — Stability (test pilot)", ["14_stability_test/stability_summary.json", "14_stability_test/stability_trials.csv"], "src.step14_stability", ["--split", "test", "--n-per-class", "4"]),
                ("Step 15 — MC Grad-CAM uncertainty", ["15_explanation_uncertainty_test_final/uncertainty_summary.json", "15_explanation_uncertainty_test_final/explanation_uncertainty.csv"], "src.step15_explanation_uncertainty", ["--split", "test", "--layer", "final", "--n-per-class", "4", "--passes", "20"]),
            ]
            for label, files, mod, arguments in steps:
                run_step(a, label, files, mod, arguments, args.check_only)
            # Avoid generating an integrated report from unmatched MRI samples.
            sources_present = all((C.OUTPUT / p).is_file() for p in [
                "13_faithfulness_test/faithfulness_trials.csv",
                "14_stability_test/stability_trials.csv",
                "15_explanation_uncertainty_test_final/explanation_uncertainty.csv"])
            if sources_present:
                matches = [keyset(a.read_csv(p, required=False)) for p in [
                    "13_faithfulness_test/faithfulness_trials.csv",
                    "14_stability_test/stability_trials.csv",
                    "15_explanation_uncertainty_test_final/explanation_uncertainty.csv"]]
                consistent = all(len(s) == 16 for s in matches) and matches[0] == matches[1] == matches[2]
            else:
                consistent = False
            if consistent:
                did_integrate = run_step(a, "Step 16 — Integrated reliability", [
                    "16_integrated_test_final/integrated_summary.json", "16_integrated_test_final/per_image_reliability_profile.csv"],
                    "src.step16_integrate_reliability", ["--split", "test", "--layer", "final"], args.check_only)
            else:
                did_integrate = False
                a.add("FAIL", "Step 16 blocked: samples do not match", "Missing files or different MRI IDs; refusing misleading combination")
            if did_integrate:
                run_step(a, "Step 17 — Thesis results report", [
                    "17_thesis_report/Kaggle_V9_thesis_results.md"], "src.step17_thesis_report", [], args.check_only)
            else:
                a.add("FAIL", "Step 17 blocked", "Integration was missing or incomplete")
        else:
            a.add("FAIL", "Post-hoc execution blocked", "Verify dataset, training fingerprint and checkpoint first")
        # Always attempt a full audit of everything that's available.
        if dataset_ok:
            audit_core_artifacts(a)
            audit_mc_and_calibration(a)
            audit_posthoc_summaries(a)
            common_ids = audit_experiment_samples(a)
            audit_integration(a, common_ids)
            report_md = C.OUTPUT / "17_thesis_report/Kaggle_V9_thesis_results.md"
            report_ok = report_md.is_file() and report_md.stat().st_size > 0
            a.add("PASS" if report_ok else "FAIL", "17 Final findings report",
                  "Found" if report_ok else "Not generated")
            integrated_json = C.OUTPUT / "16_integrated_test_final/integrated_summary.json"
            if report_ok and integrated_json.is_file():
                fresh = report_md.stat().st_mtime >= integrated_json.stat().st_mtime
                a.add("PASS" if fresh else "FAIL", "17 Report freshness",
                      "Report generated after the integrated analysis" if fresh else "Report older than Step 16: archive and regenerate report")
        a.add("WARN", "Clinical applicability", "NOT established: no patient-disjoint/external evaluation or clinician-confirmed localization")
    except Exception as exc:
        a.add("FAIL", "Unexpected audit error", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        checkpoint_end = sha_file(C.CHECKPOINT) if C.CHECKPOINT.is_file() else None
        if checkpoint_start and checkpoint_end and checkpoint_start == checkpoint_end:
            a.add("PASS", "Model weights preserved", "Checkpoint SHA256 unchanged throughout final audit")
        else:
            a.add("FAIL", "Model checkpoint integrity", "Missing or altered during audit")
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        audit_dir = C.OUTPUT / "18_final_audit" / stamp
        payload = write_report(a, audit_dir, args.check_only)
        print("\n" + "="*78)
        print("AUDIT FINISHED:", payload["technical_status"])
        print("REPORT:", audit_dir / "FINAL_AUDIT.html")
        print("ALSO:", audit_dir / "FINAL_AUDIT.md")
        print("CHECK COUNTS:", payload["checks"])
        print("NOTICE: A technically complete research prototype is NOT a clinically validated system.")
        print("="*78, flush=True)
    return 2 if any(e["status"] == "FAIL" for e in a.entries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
