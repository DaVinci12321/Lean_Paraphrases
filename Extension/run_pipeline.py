#!/usr/bin/env python3
"""
run_pipeline.py -- run the whole Lean Paraphrases extension end to end.

One command drives all four stages in order, with a single cost confirmation
up front instead of one per script:

    calibrate   score the paper's 50 hand-written pairs with both verifiers and
                print what each confidence threshold would accept, so the
                threshold used by the rest of the run is chosen from ground
                truth rather than guessed
    dataset     generate_paraphrase_dataset.py -- build the paraphrase dataset
    formalize   autoformalize_claude.py and autoformalize_gpt.py -- turn every
                paraphrase into a Lean 4 statement with `sorry`
    check       check_autoformalizations.py -- judge all four output files and
                write the report

Each stage runs as a subprocess, so its progress bars and logs come through
live and a failure in one stage cannot corrupt the others. Stages that have
already finished are detected and skipped; `--resume` continues a partial one.

Usage
-----
    python run_pipeline.py                       # the full 1000-problem run
    python run_pipeline.py --target 10           # small pilot, all four stages
    python run_pipeline.py --dry-run             # no API calls, no credentials
    python run_pipeline.py --resume              # pick up where it stopped
    python run_pipeline.py --only check          # re-run one stage
    python run_pipeline.py --skip-calibrate --yes  # unattended

After calibration the run pauses once to let you accept or change the
threshold, since seeing that table is the entire reason to calibrate. Pass
--yes to keep the configured threshold and run unattended.

State is kept in <workdir>/pipeline_state.json: which stages finished, when,
and with what settings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from lp_common import read_lines, require_keys

SCRIPT_DIR = Path(__file__).resolve().parent

STAGES = ("calibrate", "dataset", "formalize", "check")


# --------------------------------------------------------------------------
# Stage completion detection
# --------------------------------------------------------------------------

def line_count(path: Path) -> int:
    try:
        return len(read_lines(path))
    except FileNotFoundError:
        return -1


def dataset_status(workdir: Path, target: int) -> tuple[bool, bool, str]:
    """-> (complete, started, note)"""
    d = workdir / "dataset"
    files = ["rigorousparaphrases.txt", "nonrigorousparaphrases.txt",
             "formalstatements.txt"]
    started = (d / "records.jsonl").exists() or any((d / f).exists() for f in files)
    counts = {f: line_count(d / f) for f in files}
    if any(c < 0 for c in counts.values()):
        return False, started, "in progress" if started else "not started"
    if len(set(counts.values())) != 1:
        return False, True, f"files not line-aligned: {counts}"
    n = next(iter(counts.values()))
    if n < target:
        return False, True, f"{n}/{target} problems"
    return True, True, f"{n} problems"


def formalize_status(workdir: Path, model: str, n: int) -> tuple[bool, bool, str]:
    d = workdir / "formalized" / model
    started = any(d.glob("formalized_*.jsonl")) if d.exists() else False
    counts = {v: line_count(d / f"formalized_{v}.txt")
              for v in ("rigorous", "nonrigorous")}
    if any(c < 0 for c in counts.values()):
        return False, started, "in progress" if started else "not started"
    if set(counts.values()) != {n}:
        return False, True, f"{counts} lines, expected {n}"
    # A line can legitimately be empty (the model failed on that problem), so
    # line count is the completeness test, not content.
    return True, True, f"{n} problems"


def calibrate_status(workdir: Path) -> tuple[bool, bool, str]:
    path = workdir / "dataset" / "calibration.jsonl"
    if not path.exists():
        return False, False, "not started"
    try:
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip())
    except OSError:
        return False, True, "unreadable"
    return bool(n), True, (f"{n} pairs scored" if n else "empty")


def check_status(workdir: Path) -> tuple[bool, bool, str]:
    d = workdir / "evaluation"
    started = any(d.glob("*.jsonl")) if d.exists() else False
    report = d / "report.txt"
    if not report.exists():
        return False, started, "in progress" if started else "not started"
    return True, True, f"report at {report}"


# --------------------------------------------------------------------------
# Forcing a stage to re-run
# --------------------------------------------------------------------------

STAGE_OUTPUTS: dict[str, list[str]] = {
    "calibrate": ["dataset/calibration.jsonl"],
    "dataset": ["dataset/rigorousparaphrases.txt",
                "dataset/nonrigorousparaphrases.txt",
                "dataset/formalstatements.txt",
                "dataset/dataset.jsonl", "dataset/records.jsonl",
                "dataset/summary.json"],
    "formalize/claude": ["formalized/claude"],
    "formalize/gpt": ["formalized/gpt"],
    "check": ["evaluation"],
}


def clear_stage_outputs(workdir: Path, stage: str) -> None:
    """Delete a stage's output so it can be rebuilt from scratch.

    The child scripts refuse to overwrite existing output (that guard is what
    stops an accidental re-run from silently discarding a long job), so
    --force has to clear the way explicitly. Calibration output is never
    touched by the dataset stage, and vice versa.
    """
    import shutil
    for rel in STAGE_OUTPUTS.get(stage, []):
        path = workdir / rel
        if not path.exists():
            continue
        print(f"  --force: removing {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def require_resume_or_force(stage: str, note: str) -> None:
    print(f"\nerror: stage '{stage}' is half finished ({note}).\n"
          f"       Re-run with --resume to continue it, or --force to discard\n"
          f"       that work and start the stage over.", file=sys.stderr)


# --------------------------------------------------------------------------
# Running a stage
# --------------------------------------------------------------------------

def run_step(cmd: list[str], label: str, log_path: Path | None = None) -> int:
    """Run one child script. Streams its output unless a log path is given."""
    printable = " ".join(c if " " not in c else f'"{c}"' for c in cmd[1:])
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"  python {printable}")
    print(f"{'=' * 72}\n", flush=True)

    start = time.time()
    if log_path is None:
        rc = subprocess.run(cmd, cwd=SCRIPT_DIR).returncode
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=SCRIPT_DIR, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        print(f"  ({label} finished, log: {log_path})")

    elapsed = time.time() - start
    status = "ok" if rc == 0 else f"FAILED (exit {rc})"
    print(f"\n-- {label}: {status} in {elapsed:.0f}s", flush=True)
    return rc


def run_parallel(steps: list[tuple[list[str], str, Path]]) -> dict[str, int]:
    """Run several children at once, each logging to its own file.

    Worth it for the two autoformalizers: they hit different providers, so
    neither is competing with the other for rate limit.
    """
    print(f"\n{'=' * 72}")
    print(f"  running {len(steps)} steps in parallel")
    for _, label, log in steps:
        print(f"    {label:<36} -> {log}")
    print(f"{'=' * 72}\n", flush=True)

    start = time.time()
    procs = []
    for cmd, label, log in steps:
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("w", encoding="utf-8")
        procs.append((label, subprocess.Popen(
            cmd, cwd=SCRIPT_DIR, stdout=fh, stderr=subprocess.STDOUT), fh))

    results = {}
    try:
        for label, proc, fh in procs:
            results[label] = proc.wait()
            fh.close()
            print(f"-- {label}: "
                  f"{'ok' if results[label] == 0 else f'FAILED (exit {results[label]})'}",
                  flush=True)
    except KeyboardInterrupt:
        for _, proc, fh in procs:
            proc.terminate()
            fh.close()
        raise

    print(f"-- parallel block finished in {time.time() - start:.0f}s", flush=True)
    return results


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def load_state(path: Path) -> dict:
    if not path.exists():
        return {"stages": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"stages": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def mark(state: dict, path: Path, stage: str, **fields) -> None:
    state["stages"][stage] = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **fields,
    }
    save_state(path, state)


# --------------------------------------------------------------------------
# Threshold prompt
# --------------------------------------------------------------------------

def prompt_threshold(current: float) -> float | None:
    """After calibration, let the user accept or change the threshold.

    Returns the threshold to use, or None to abort.
    """
    print()
    print("The table above shows how many of the paper's 50 hand-written pairs")
    print("each threshold would accept. Those pairs are hand-checked ground")
    print("truth, so a threshold that rejects many of them is too strict.")
    while True:
        answer = input(
            f"Continue with --threshold {current:.2f}? "
            f"[Y / n / a new value like 0.80] "
        ).strip().lower()
        if answer in ("", "y", "yes"):
            return current
        if answer in ("n", "no", "q", "quit"):
            return None
        try:
            value = float(answer)
        except ValueError:
            print("  Not a number. Enter y, n, or a threshold between 0 and 1.")
            continue
        if not 0.0 < value <= 1.0:
            print("  A threshold must be between 0 and 1.")
            continue
        print(f"  using --threshold {value:.2f}")
        return value


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run the whole Lean Paraphrases extension pipeline: "
                    "calibrate, build the dataset, autoformalize with both "
                    "models, and judge the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", type=int, default=1000,
                   help="number of problems in the dataset")
    p.add_argument("--workdir", type=Path, default=SCRIPT_DIR,
                   help="root for dataset/, formalized/, evaluation/ and logs/")
    p.add_argument("--threshold", type=float, default=0.85,
                   help="confidence threshold, used by every judging stage")
    p.add_argument("--workers", type=int, default=6,
                   help="concurrency within each stage")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="paraphrase generation attempts per source problem")
    p.add_argument("--effort", default="medium",
                   choices=("low", "medium", "high", "xhigh", "max"),
                   help="Claude Opus 5 reasoning effort")
    p.add_argument("--dataset", default="SphereLab/FormalMATH-All",
                   help="HuggingFace source dataset")
    p.add_argument("--manual-dir", type=Path, default=None,
                   help="directory holding the paper's hand-written pairs "
                        "(default: auto-detected)")
    p.add_argument("--only", nargs="+", choices=STAGES, default=None,
                   metavar="STAGE",
                   help=f"run only these stages ({', '.join(STAGES)})")
    p.add_argument("--skip-calibrate", action="store_true",
                   help="skip calibration and use --threshold as given")
    p.add_argument("--sequential-formalize", action="store_true",
                   help="run the two autoformalizers one after the other "
                        "instead of together (slower, but streams progress)")
    p.add_argument("--force", action="store_true",
                   help="re-run stages even if their output already looks complete")
    p.add_argument("--resume", action="store_true",
                   help="resume partially finished stages instead of restarting")
    p.add_argument("--dry-run", action="store_true",
                   help="run every stage with stub models and no API calls")
    p.add_argument("--yes", action="store_true",
                   help="skip all confirmations, including the threshold prompt")
    args = p.parse_args(argv)

    stages = list(args.only) if args.only else list(STAGES)
    if args.skip_calibrate and "calibrate" in stages:
        stages.remove("calibrate")

    workdir = args.workdir.resolve()
    logs = workdir / "logs"
    state_path = workdir / "pipeline_state.json"
    state = load_state(state_path)
    py = sys.executable

    # -- preflight ---------------------------------------------------------
    print("Lean Paraphrases -- full pipeline")
    print("=" * 72)
    print(f"workdir   : {workdir}")
    print(f"target    : {args.target} problems")
    print(f"threshold : {args.threshold}")
    print(f"stages    : {', '.join(stages)}")
    print(f"mode      : {'DRY RUN (no API calls)' if args.dry_run else 'live'}")

    if not args.dry_run:
        missing = require_keys("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        if missing:
            print(f"\nerror: missing environment variable(s): "
                  f"{', '.join(missing)}", file=sys.stderr)
            return 2

    for mod in ("openai", "anthropic", "datasets"):
        try:
            __import__(mod)
        except ImportError:
            if args.dry_run and mod != "datasets":
                continue
            print(f"\nerror: missing dependency '{mod}'. Run: "
                  f"pip install -r requirements.txt", file=sys.stderr)
            return 2

    # -- what is already done ---------------------------------------------
    n = args.target
    done_calibrate, started_calibrate, calibrate_note = calibrate_status(workdir)
    done_dataset, started_dataset, dataset_note = dataset_status(workdir, args.target)
    if done_dataset:
        n = line_count(workdir / "dataset" / "rigorousparaphrases.txt")
    done_claude, started_claude, claude_note = formalize_status(workdir, "claude", n)
    done_gpt, started_gpt, gpt_note = formalize_status(workdir, "gpt", n)
    done_check, started_check, check_note = check_status(workdir)

    print("\nexisting output")
    print("-" * 72)
    print(f"  calibration         : {calibrate_note}")
    print(f"  dataset             : {dataset_note}")
    print(f"  formalized/claude   : {claude_note}")
    print(f"  formalized/gpt      : {gpt_note}")
    print(f"  evaluation          : {check_note}")

    # -- cost estimate ------------------------------------------------------
    if not args.dry_run:
        calls_lo = calls_hi = 0
        if "calibrate" in stages and not (done_calibrate and not args.force):
            calls_lo += 100
            calls_hi += 100                      # 2 judges x 50 manual pairs
        if "dataset" in stages and not (done_dataset and not args.force):
            calls_lo += 3 * args.target
            calls_hi += 3 * args.target * args.max_attempts
        if "formalize" in stages:
            for is_done in (done_claude, done_gpt):
                if not (is_done and not args.force):
                    calls_lo += 2 * args.target
                    calls_hi += 2 * args.target
        if "check" in stages and not (done_check and not args.force):
            calls_lo += 8 * args.target          # 2 judges x 4 configurations
            calls_hi += 8 * args.target

        print("\nestimated API calls")
        print("-" * 72)
        print(f"  {calls_lo:,} to {calls_hi:,} across the whole run. "
              f"This costs real money.")
        if not args.yes and calls_hi > 0:
            if input("\nProceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1

    common = ["--workers", str(args.workers), "--effort", args.effort, "--yes"]
    if args.dry_run:
        common.append("--dry-run")

    threshold = args.threshold
    start = time.time()

    try:
        # -- calibrate -----------------------------------------------------
        if "calibrate" in stages and done_calibrate and not args.force:
            print(f"\n-- stage: calibrate already done ({calibrate_note}), "
                  f"skipping. Use --force to re-score, or pass --threshold to "
                  f"change the bar without re-scoring.")
        elif "calibrate" in stages:
            if args.force:
                clear_stage_outputs(workdir, "calibrate")
            cmd = [py, "generate_paraphrase_dataset.py", "--calibrate",
                   "--threshold", str(threshold),
                   "--outdir", str(workdir / "dataset"), *common]
            if args.manual_dir:
                cmd += ["--manual-dir", str(args.manual_dir)]
            rc = run_step(cmd, "stage: calibrate")
            if rc != 0:
                print("\ncalibration failed -- stopping. Fix the problem above, "
                      "or re-run with --skip-calibrate.", file=sys.stderr)
                return rc
            mark(state, state_path, "calibrate", threshold=threshold)

            if not args.yes and not args.dry_run:
                chosen = prompt_threshold(threshold)
                if chosen is None:
                    print("aborted after calibration")
                    return 1
                threshold = chosen

        # -- dataset -------------------------------------------------------
        if "dataset" in stages:
            if done_dataset and not args.force:
                print(f"\n-- stage: dataset already complete ({dataset_note}), "
                      f"skipping. Use --force to rebuild.")
            elif started_dataset and not (args.resume or args.force):
                require_resume_or_force("dataset", dataset_note)
                return 2
            else:
                if args.force:
                    clear_stage_outputs(workdir, "dataset")
                cmd = [py, "generate_paraphrase_dataset.py",
                       "--target", str(args.target),
                       "--outdir", str(workdir / "dataset"),
                       "--dataset", args.dataset,
                       "--threshold", str(threshold),
                       "--max-attempts", str(args.max_attempts), *common]
                if args.manual_dir:
                    cmd += ["--manual-dir", str(args.manual_dir)]
                if args.resume:
                    cmd.append("--resume")
                rc = run_step(cmd, "stage: dataset")
                if rc != 0:
                    print("\ndataset stage failed -- stopping.", file=sys.stderr)
                    return rc
                mark(state, state_path, "dataset", target=args.target,
                     threshold=threshold)

            ok, _, note = dataset_status(workdir, 1)
            if not ok:
                print(f"\nerror: the dataset stage produced nothing usable "
                      f"({note}).", file=sys.stderr)
                return 1
            n = line_count(workdir / "dataset" / "rigorousparaphrases.txt")
            if n < args.target:
                print(f"\nwarning: dataset has {n} problems, fewer than the "
                      f"{args.target} requested. Continuing with {n}.",
                      file=sys.stderr)
            done_claude, started_claude, claude_note = formalize_status(
                workdir, "claude", n)
            done_gpt, started_gpt, gpt_note = formalize_status(workdir, "gpt", n)

        # -- formalize -----------------------------------------------------
        if "formalize" in stages:
            todo = []
            for model, is_done, was_started, note in (
                    ("claude", done_claude, started_claude, claude_note),
                    ("gpt", done_gpt, started_gpt, gpt_note)):
                if is_done and not args.force:
                    print(f"\n-- stage: formalize/{model} already complete "
                          f"({note}), skipping. Use --force to redo.")
                    continue
                if was_started and not (args.resume or args.force):
                    require_resume_or_force(f"formalize/{model}", note)
                    return 2
                if args.force:
                    clear_stage_outputs(workdir, f"formalize/{model}")
                cmd = [py, f"autoformalize_{model}.py",
                       "--dataset-dir", str(workdir / "dataset"),
                       "--outdir", str(workdir / "formalized" / model), *common]
                if args.resume:
                    cmd.append("--resume")
                todo.append((cmd, f"stage: formalize/{model}",
                             logs / f"formalize_{model}.log"))

            if todo:
                if args.sequential_formalize or len(todo) == 1:
                    for cmd, label, _ in todo:
                        rc = run_step(cmd, label)
                        if rc != 0:
                            print(f"\n{label} failed -- stopping.",
                                  file=sys.stderr)
                            return rc
                else:
                    # Different providers, so neither competes with the other
                    # for rate limit. Output goes to per-model logs to keep the
                    # two progress bars from interleaving.
                    results = run_parallel(todo)
                    if any(rc != 0 for rc in results.values()):
                        failed = [l for l, rc in results.items() if rc != 0]
                        print(f"\n{', '.join(failed)} failed -- see the logs in "
                              f"{logs}.", file=sys.stderr)
                        return 1
                mark(state, state_path, "formalize", problems=n)

        # -- check ---------------------------------------------------------
        if "check" in stages:
            if done_check and not args.force:
                print(f"\n-- stage: check already complete ({check_note}), "
                      f"skipping. Use --force to redo.")
            elif started_check and not (args.resume or args.force):
                require_resume_or_force("check", check_note)
                return 2
            else:
                if args.force:
                    clear_stage_outputs(workdir, "check")
                cmd = [py, "check_autoformalizations.py",
                       "--dataset-dir", str(workdir / "dataset"),
                       "--formalized-dir", str(workdir / "formalized"),
                       "--outdir", str(workdir / "evaluation"),
                       "--threshold", str(threshold), *common]
                if args.resume:
                    cmd.append("--resume")
                rc = run_step(cmd, "stage: check")
                if rc != 0:
                    print("\ncheck stage failed -- stopping.", file=sys.stderr)
                    return rc
                mark(state, state_path, "check", threshold=threshold)

    except KeyboardInterrupt:
        print("\n\ninterrupted. Every finished stage kept its output; re-run "
              "with --resume to continue.", file=sys.stderr)
        return 130

    # -- wrap up ------------------------------------------------------------
    elapsed = time.time() - start
    print(f"\n{'=' * 72}")
    print(f"  pipeline finished in {elapsed / 60:.1f} min")
    print(f"{'=' * 72}")
    print(f"  dataset     : {workdir / 'dataset'}")
    print(f"  formalized  : {workdir / 'formalized'}")
    print(f"  evaluation  : {workdir / 'evaluation'}")
    print(f"  state       : {state_path}")

    report = workdir / "evaluation" / "report.txt"
    if report.exists():
        print(f"\n{'-' * 72}")
        print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
