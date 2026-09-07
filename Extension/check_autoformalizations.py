#!/usr/bin/env python3
"""
check_autoformalizations.py -- stage 3 of the Lean Paraphrases extension.

Judges all four autoformalization files produced by stage 2:

    gpt_rigorous        formalized/gpt/formalized_rigorous.txt
    gpt_nonrigorous     formalized/gpt/formalized_nonrigorous.txt
    claude_rigorous     formalized/claude/formalized_rigorous.txt
    claude_nonrigorous  formalized/claude/formalized_nonrigorous.txt

For each problem in each file the judges see three things:

    1. the natural-language paraphrase the formalization was produced from
       (the rigorous one for a *_rigorous file, the non-rigorous one otherwise)
    2. the candidate Lean 4 formalization under judgement
    3. the reference Lean 4 statement from dataset/formalstatements.txt, as a
       worked example of how this problem can be formalized

The question asked is always the same: does the candidate state the same
mathematics as the natural-language problem? The reference is context, not an
answer key -- a candidate that differs from it but still matches the English is
correct.

Two judges vote, Claude Opus 5 and GPT 5.6 Sol. Agreement is what decides a
problem; once both have landed on the same verdict, one confident judge is
enough to accept it:

    EQUIVALENT      both say yes, at least one at confidence >= --threshold
    NOT_EQUIVALENT  both say no,  at least one at confidence >= --threshold
    UNDECIDED       the judges disagreed, or they agreed but neither reached
                    the threshold
    MISSING         stage 2 produced no statement for this line

Statuses are always recomputed from the stored judge responses when the report
is built, so `--report-only --threshold X` re-scores a finished run without
spending anything.; not judged

Output (in --outdir, default ./evaluation)
------------------------------------------
    gpt_rigorous.jsonl          per-problem verdicts, both judges, verbatim
    gpt_nonrigorous.jsonl
    claude_rigorous.jsonl
    claude_nonrigorous.jsonl
    report.txt                  statistics, plus the numbers of every
                                undecided problem, per configuration
    report.json                 the same, machine-readable

Usage
-----
    python check_autoformalizations.py
    python check_autoformalizations.py --limit 20 --outdir /tmp/eval
    python check_autoformalizations.py --resume
    python check_autoformalizations.py --dry-run --outdir /tmp/eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from lp_common import (
    ANTHROPIC_MODEL, OPENAI_MODEL, Clients, JsonlWriter, confidence,
    load_jsonl, read_lines, rebuild_lean, require_keys,
)

SCRIPT_DIR = Path(__file__).resolve().parent

JUDGE_ANTHROPIC = "claude_opus_5"
JUDGE_OPENAI = "gpt_5_6_sol"

# (config name, stage-2 model directory, which paraphrase file it came from)
CONFIGURATIONS = [
    ("gpt_rigorous", "gpt", "rigorous"),
    ("gpt_nonrigorous", "gpt", "nonrigorous"),
    ("claude_rigorous", "claude", "rigorous"),
    ("claude_nonrigorous", "claude", "nonrigorous"),
]

EQUIVALENT, NOT_EQUIVALENT, UNDECIDED, MISSING = (
    "EQUIVALENT", "NOT_EQUIVALENT", "UNDECIDED", "MISSING")

# --------------------------------------------------------------------------
# Judge prompt
# --------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are judging whether a Lean 4 formalization faithfully captures a \
natural-language mathematics problem.

You are given three things:

  1. THE PROBLEM -- a mathematics problem stated in English.
  2. THE CANDIDATE -- a Lean 4 formalization of that problem, produced by a
     language model, with `sorry` standing in for the proof.
  3. THE REFERENCE -- a Lean 4 formalization of the same underlying problem,
     taken from a curated dataset and known to compile.

Answer exactly one question: does THE CANDIDATE state the same mathematical \
content as THE PROBLEM?

How to use the reference. It is context, not an answer key. The candidate is \
under no obligation to resemble it. Different variable names, a different \
arrangement of the same hypotheses, a different Mathlib idiom, a different but \
equivalent encoding of the same object -- all fine, all still equivalent. Use \
the reference to see how this problem can reasonably be expressed in Lean and \
what domains and quantifier structure the problem intends. Where the candidate \
differs from the reference in a way that changes the mathematics, that is a \
signal worth checking, but the thing the candidate must match is THE PROBLEM, \
not the reference.

Judge NOT equivalent when the candidate:
  * changes, adds or drops a hypothesis -- including a variable's type or
    domain. ℕ, ℤ, ℚ and ℝ are all different; so are "positive", "non-negative"
    and "nonzero". Inventing a hypothesis the problem never stated is just as
    wrong as dropping one it did.
  * changes a quantifier, its scope, or the order of quantifiers.
  * states a different conclusion, or asserts a different final answer than the
    problem asserts.
  * flips a strict inequality to non-strict, or states divisibility backwards.
  * formalizes only part of the problem, or splits off a weaker claim.
  * is vacuous or trivially true as written -- for instance hypotheses that
    cannot be simultaneously satisfied, or a conclusion that holds by
    definition regardless of the hypotheses.
  * is not well-formed Lean 4 at the level of the statement.

Do NOT judge it non-equivalent for any of these:
  * using `sorry` as the proof. That is required here, not a defect.
  * omitting `import` or `open` lines.
  * naming, formatting, or Mathlib style choices.
  * being a longer, shorter, or differently structured but equivalent encoding.
  * being unproved, hard, or even false as a mathematical claim. You are
    judging faithfulness of the translation, not the truth of the theorem.

You are not a compiler. Judge the statement's meaning and its surface \
well-formedness, not whether it would elaborate against a particular Mathlib \
version.

Your confidence is your probability that your own verdict is correct, on a 0-1 \
scale. Use the full range and calibrate honestly: a translation you have \
checked clause by clause deserves a high confidence either way, and only \
genuine uncertainty belongs in the middle. Do not shade confidence downward \
merely to seem cautious -- a low confidence here means the problem gets thrown \
into the undecided pile, not that you were appropriately careful.\
"""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string",
            "description": "Walk the hypotheses, the variable domains, the "
                           "quantifiers and the conclusion of the candidate "
                           "against the English problem, one at a time.",
        },
        "differences": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One entry per difference in mathematical content "
                           "between the candidate and the problem. Empty if "
                           "there are none.",
        },
        "equivalent": {
            "type": "boolean",
            "description": "True if the candidate states the same mathematics "
                           "as the English problem.",
        },
        # No "minimum"/"maximum": neither provider's strict JSON-schema mode
        # accepts numeric bounds. Clamped in Python instead.
        "confidence": {
            "type": "number",
            "description": "Probability from 0.0 to 1.0 that your verdict in "
                           "the `equivalent` field is correct.",
        },
    },
    "required": ["analysis", "differences", "equivalent", "confidence"],
    "additionalProperties": False,
}


def judge_prompt(problem: str, candidate: str, reference: str) -> str:
    return "\n".join([
        "THE PROBLEM (natural language)",
        problem,
        "",
        "THE CANDIDATE (Lean 4 formalization under judgement)",
        rebuild_lean("", candidate),
        "",
        "THE REFERENCE (a known-good Lean 4 formalization of the same problem, "
        "for context only)",
        rebuild_lean("", reference),
        "",
        "Does the candidate state the same mathematics as the problem?",
    ])


# --------------------------------------------------------------------------
# Judging one problem
# --------------------------------------------------------------------------

def fake_verdict(index: int) -> dict:
    # Deterministic stub for --dry-run: mostly agreeing, with a disagreement
    # and a low-confidence case seeded in so the report paths get exercised.
    if index % 7 == 0:
        return {"analysis": "dry run", "differences": ["stub difference"],
                "equivalent": False, "confidence": 0.93}
    if index % 5 == 0:
        return {"analysis": "dry run", "differences": [],
                "equivalent": True, "confidence": 0.55}
    return {"analysis": "dry run", "differences": [],
            "equivalent": True, "confidence": 0.94}


def decide(verdicts: dict[str, dict], threshold: float) -> tuple[str, str]:
    """Combine the two judges into a status and a human-readable reason.

    Agreement is what decides a problem. Once both judges have landed on the
    same verdict, one of them being confident is enough to accept it -- the
    second judge concurring at 0.7 is corroboration, not doubt. Only a genuine
    split sends a problem to the undecided pile.

        judges disagree                     -> UNDECIDED, always
        judges agree, at least one >= T     -> that verdict
        judges agree, neither reaches T     -> UNDECIDED
    """
    errored = [n for n, v in verdicts.items() if "error" in v]
    if errored:
        return UNDECIDED, f"judge error: {', '.join(errored)}"
    if len(verdicts) != 2:
        return UNDECIDED, "a judge did not answer"

    names = list(verdicts)
    votes = {n: bool(verdicts[n].get("equivalent")) for n in names}
    confs = {n: confidence(verdicts[n], "confidence") for n in names}
    detail = ", ".join(f"{n}={'yes' if votes[n] else 'no'} {confs[n]:.2f}"
                       for n in names)

    if votes[names[0]] != votes[names[1]]:
        return UNDECIDED, f"judges disagreed ({detail})"

    if max(confs.values()) < threshold:
        return UNDECIDED, (f"judges agreed but neither reached "
                           f"{threshold:.2f} ({detail})")

    return (EQUIVALENT if votes[names[0]] else NOT_EQUIVALENT), detail


def rescore(results: dict[str, dict[int, dict]], threshold: float) -> int:
    """Recompute every stored verdict against the current rule and threshold.

    The raw judge responses are the expensive part and never change; `status`
    and `reason` are derived from them. Deriving them fresh here means
    --report-only re-scores an existing run for free, and a report can never
    disagree with the rule that produced it.
    """
    changed = 0
    for recs in results.values():
        for rec in recs.values():
            verdicts = rec.get("verdicts") or {}
            if not verdicts:
                continue  # MISSING: never judged, nothing to recompute
            status, reason = decide(verdicts, threshold)
            if status != rec.get("status"):
                changed += 1
            rec["status"], rec["reason"] = status, reason
    return changed


def judge_one(clients: Clients, index: int, problem: str, candidate: str,
              reference: str, threshold: float, dry_run: bool) -> dict:
    record: dict[str, Any] = {
        "index": index, "problem": problem, "candidate": candidate,
        "reference": reference, "verdicts": {}, "status": "", "reason": "",
    }

    if not candidate.strip():
        record["status"] = MISSING
        record["reason"] = "stage 2 produced no statement for this problem"
        return record
    if not problem.strip():
        record["status"] = MISSING
        record["reason"] = "no natural-language problem on this line"
        return record

    if dry_run:
        record["verdicts"] = {JUDGE_ANTHROPIC: fake_verdict(index),
                              JUDGE_OPENAI: fake_verdict(index + 1)}
    else:
        prompt = judge_prompt(problem, candidate, reference)
        verdicts: dict[str, dict] = {}

        def run(name: str, fn: Callable[[], dict]) -> None:
            try:
                verdicts[name] = fn()
            except Exception as exc:  # noqa: BLE001
                verdicts[name] = {"error": f"{type(exc).__name__}: {exc}"}

        # Both judges in parallel: the round trip dominates wall-clock.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for fut in [
                pool.submit(run, JUDGE_ANTHROPIC, lambda: clients.anthropic_json(
                    JUDGE_SYSTEM, prompt, JUDGE_SCHEMA, "verdict",
                    model=ANTHROPIC_MODEL)),
                pool.submit(run, JUDGE_OPENAI, lambda: clients.openai_json(
                    JUDGE_SYSTEM, prompt, JUDGE_SCHEMA, "verdict",
                    model=OPENAI_MODEL)),
            ]:
                fut.result()
        record["verdicts"] = verdicts

    record["status"], record["reason"] = decide(record["verdicts"], threshold)
    return record


# --------------------------------------------------------------------------
# Judging one configuration
# --------------------------------------------------------------------------

def check_configuration(clients: Clients, name: str, candidates: list[str],
                        problems: list[str], references: list[str],
                        outdir: Path, args) -> dict[int, dict]:
    jsonl_path = outdir / f"{name}.jsonl"

    done: dict[int, dict] = {}
    if args.resume:
        for rec in load_jsonl(jsonl_path):
            idx = rec.get("index", 0)
            if not 1 <= idx <= len(candidates):
                continue
            # A judge error is worth retrying; a real verdict is not.
            if rec.get("reason", "").startswith("judge error"):
                continue
            # Only reuse a verdict that was passed the same two texts. Re-running
            # an earlier stage changes what sits on line N, and a stale verdict
            # would then be attributed to a formalization it never saw.
            if (rec.get("candidate") != candidates[idx - 1]
                    or rec.get("problem") != problems[idx - 1]):
                continue
            done[idx] = rec
        print(f"  {name}: resuming with {len(done)} already judged")
    elif jsonl_path.exists():
        print(f"error: {jsonl_path} already exists. Pass --resume to continue "
              f"that run, or choose a different --outdir.", file=sys.stderr)
        raise SystemExit(2)

    todo = [i for i in range(1, len(candidates) + 1) if i not in done]
    writer = JsonlWriter(jsonl_path, append=args.resume)

    try:
        from tqdm import tqdm
        bar = tqdm(total=len(todo), unit="problem", desc=name)
    except ImportError:
        bar = None

    def work(i: int) -> dict:
        rec = judge_one(clients, i, problems[i - 1], candidates[i - 1],
                        references[i - 1], args.threshold, args.dry_run)
        writer.write(rec)
        if bar:
            bar.update(1)
        return rec

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rec in pool.map(work, todo):
                done[rec["index"]] = rec
    except KeyboardInterrupt:
        print("\ninterrupted -- reporting on what finished so far",
              file=sys.stderr)
    finally:
        if bar:
            bar.close()
        writer.close()

    return done


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def summarise(results: dict[str, dict[int, dict]], n: int,
              threshold: float) -> dict:
    per_config: dict[str, Any] = {}
    for name, recs in results.items():
        counts = {EQUIVALENT: 0, NOT_EQUIVALENT: 0, UNDECIDED: 0, MISSING: 0}
        undecided: list[dict] = []
        for i in range(1, n + 1):
            rec = recs.get(i)
            status = rec.get("status", UNDECIDED) if rec else UNDECIDED
            counts[status] = counts.get(status, 0) + 1
            if status == UNDECIDED:
                undecided.append({
                    "index": i,
                    "reason": rec.get("reason", "not judged") if rec
                              else "not judged",
                })
        decided = counts[EQUIVALENT] + counts[NOT_EQUIVALENT]
        per_config[name] = {
            "counts": counts,
            "decided": decided,
            "equivalence_rate_of_decided": (
                round(counts[EQUIVALENT] / decided, 4) if decided else None),
            "equivalence_rate_of_all": round(counts[EQUIVALENT] / n, 4) if n else None,
            "undecided_indices": [u["index"] for u in undecided],
            "undecided_detail": undecided,
        }

    # The paper's central question: does rewording alone change the verdict?
    sensitivity: dict[str, Any] = {}
    for model in ("gpt", "claude"):
        rig = results.get(f"{model}_rigorous", {})
        non = results.get(f"{model}_nonrigorous", {})
        flipped, both_wrong, both_right = [], [], []
        for i in range(1, n + 1):
            a = (rig.get(i) or {}).get("status")
            b = (non.get(i) or {}).get("status")
            if a not in (EQUIVALENT, NOT_EQUIVALENT) or b not in (
                    EQUIVALENT, NOT_EQUIVALENT):
                continue
            if a != b:
                flipped.append(i)
            elif a == NOT_EQUIVALENT:
                both_wrong.append(i)
            else:
                both_right.append(i)
        sensitivity[model] = {
            "comparable_problems": len(flipped) + len(both_wrong) + len(both_right),
            "verdict_flipped_between_paraphrases": len(flipped),
            "flipped_indices": flipped,
            "wrong_on_both_paraphrases": len(both_wrong),
            "correct_on_both_paraphrases": len(both_right),
        }

    # How often the two autoformalizing models fail on the same problem.
    cross: dict[str, Any] = {}
    for variant in ("rigorous", "nonrigorous"):
        g = results.get(f"gpt_{variant}", {})
        c = results.get(f"claude_{variant}", {})
        agree = both_wrong = 0
        for i in range(1, n + 1):
            a = (g.get(i) or {}).get("status")
            b = (c.get(i) or {}).get("status")
            if a not in (EQUIVALENT, NOT_EQUIVALENT) or b not in (
                    EQUIVALENT, NOT_EQUIVALENT):
                continue
            agree += a == b
            both_wrong += a == b == NOT_EQUIVALENT
        cross[variant] = {"models_agreed": agree, "both_models_wrong": both_wrong}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge_models": [ANTHROPIC_MODEL, OPENAI_MODEL],
        "confidence_threshold": threshold,
        "decision_rule": ("judges must agree; at least one of them must reach "
                          "the confidence threshold"),
        "problems": n,
        "per_configuration": per_config,
        "paraphrase_sensitivity": sensitivity,
        "cross_model": cross,
    }


def render_report(summary: dict) -> str:
    n = summary["problems"]
    out: list[str] = []
    add = out.append

    add("Lean Paraphrases -- autoformalization equivalence report")
    add("=" * 72)
    add(f"generated : {summary['generated_at']}")
    add(f"judges    : {', '.join(summary['judge_models'])}")
    add(f"threshold : {summary['confidence_threshold']:.2f} "
        f"(judges must agree; at least one must clear this confidence)")
    add(f"problems  : {n}")
    add("")

    add("RESULTS BY CONFIGURATION")
    add("-" * 72)
    add(f"{'configuration':<22}{'equiv':>8}{'not equiv':>11}"
        f"{'undecided':>11}{'missing':>9}{'equiv%':>9}")
    for name, cfg in summary["per_configuration"].items():
        c = cfg["counts"]
        rate = cfg["equivalence_rate_of_decided"]
        rate_s = f"{rate:.1%}" if rate is not None else "n/a"
        add(f"{name:<22}{c[EQUIVALENT]:>8}{c[NOT_EQUIVALENT]:>11}"
            f"{c[UNDECIDED]:>11}{c[MISSING]:>9}{rate_s:>9}")
    add("")
    add("equiv% is equivalent / (equivalent + not equivalent); undecided and")
    add("missing problems are excluded from it.")
    add("")

    add("EFFECT OF PARAPHRASING (same model, same problem, two wordings)")
    add("-" * 72)
    for model, s in summary["paraphrase_sensitivity"].items():
        total = s["comparable_problems"]
        flipped = s["verdict_flipped_between_paraphrases"]
        pct = f"{flipped / total:.1%}" if total else "n/a"
        add(f"{model}:")
        add(f"  problems decided for both paraphrases : {total}")
        add(f"  verdict changed with the wording      : {flipped} ({pct})")
        add(f"  wrong on both paraphrases             : {s['wrong_on_both_paraphrases']}")
        add(f"  correct on both paraphrases           : {s['correct_on_both_paraphrases']}")
        if s["flipped_indices"]:
            add(f"  problem numbers that flipped          : "
                f"{format_indices(s['flipped_indices'])}")
        add("")

    add("AGREEMENT BETWEEN THE TWO AUTOFORMALIZING MODELS")
    add("-" * 72)
    for variant, s in summary["cross_model"].items():
        add(f"{variant:<14} same verdict on {s['models_agreed']} problems, "
            f"both wrong on {s['both_models_wrong']}")
    add("")

    add("UNDECIDED PROBLEMS")
    add("-" * 72)
    add("The two judges disagreed, or agreed but neither reached the")
    add("confidence threshold. These need a human decision.")
    add("")
    for name, cfg in summary["per_configuration"].items():
        idxs = cfg["undecided_indices"]
        add(f"{name}: {len(idxs)} undecided")
        if idxs:
            add(f"  problem numbers: {format_indices(idxs)}")
            for item in cfg["undecided_detail"]:
                add(f"    #{item['index']:<6} {item['reason']}")
        add("")

    return "\n".join(out) + "\n"


def format_indices(indices: list[int]) -> str:
    """Compress a sorted index list into ranges: 1, 3-5, 9."""
    if not indices:
        return "(none)"
    parts, start, prev = [], indices[0], indices[0]
    for i in indices[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    return ", ".join(parts)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Judge all four autoformalization files for semantic "
                    "equivalence to their natural-language problems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-dir", type=Path, default=SCRIPT_DIR / "dataset",
                   help="directory holding the paraphrase and reference files")
    p.add_argument("--formalized-dir", type=Path,
                   default=SCRIPT_DIR / "formalized",
                   help="directory holding the per-model stage-2 output")
    p.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "evaluation",
                   help="where to write the verdicts and the report")
    p.add_argument("--threshold", type=float, default=0.80,
                   help="confidence at least one agreeing judge must clear "
                        "for a decision")
    p.add_argument("--limit", type=int, default=0,
                   help="only judge the first N problems (0 = all)")
    p.add_argument("--workers", type=int, default=6,
                   help="problems judged concurrently, per configuration")
    p.add_argument("--retries", type=int, default=5,
                   help="API retry attempts per call")
    p.add_argument("--effort", default="medium",
                   choices=("low", "medium", "high", "xhigh", "max"),
                   help="Claude Opus 5 reasoning effort")
    p.add_argument("--only", nargs="*", default=None,
                   metavar="CONFIG",
                   help="judge only these configurations, e.g. --only gpt_rigorous")
    p.add_argument("--resume", action="store_true",
                   help="continue a previous run, keeping existing verdicts")
    p.add_argument("--report-only", action="store_true",
                   help="rebuild the report from existing .jsonl files, no API "
                        "calls. Re-scores them at the current --threshold.")
    p.add_argument("--rewrite-status", action="store_true",
                   help="with --report-only, also write the recomputed status "
                        "and reason back into the .jsonl files (originals are "
                        "copied to .jsonl.bak first)")
    p.add_argument("--dry-run", action="store_true",
                   help="exercise the pipeline with stub judges and no API calls")
    p.add_argument("--yes", action="store_true",
                   help="skip the pre-run cost confirmation")
    args = p.parse_args(argv)

    if not args.dry_run and not args.report_only:
        missing = require_keys("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        if missing:
            print(f"error: missing environment variable(s): {', '.join(missing)}",
                  file=sys.stderr)
            return 2

    # -- load the inputs and check that everything is line-aligned ----------
    try:
        paraphrases = {
            "rigorous": read_lines(args.dataset_dir / "rigorousparaphrases.txt"),
            "nonrigorous": read_lines(
                args.dataset_dir / "nonrigorousparaphrases.txt"),
        }
        references = read_lines(args.dataset_dir / "formalstatements.txt")
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    configs = [c for c in CONFIGURATIONS
               if args.only is None or c[0] in args.only]
    if not configs:
        print(f"error: --only matched no configuration. Choose from: "
              f"{', '.join(c[0] for c in CONFIGURATIONS)}", file=sys.stderr)
        return 2

    candidates: dict[str, list[str]] = {}
    for name, model_dir, variant in configs:
        path = args.formalized_dir / model_dir / f"formalized_{variant}.txt"
        try:
            candidates[name] = read_lines(path)
        except FileNotFoundError as exc:
            print(f"error: {exc}\n       run autoformalize_{model_dir}.py first",
                  file=sys.stderr)
            return 2

    lengths = {"rigorous": len(paraphrases["rigorous"]),
               "nonrigorous": len(paraphrases["nonrigorous"]),
               "formalstatements": len(references),
               **{k: len(v) for k, v in candidates.items()}}
    if len(set(lengths.values())) != 1:
        print("error: input files are not line-aligned. Line counts:",
              file=sys.stderr)
        for k, v in lengths.items():
            print(f"       {k}: {v}", file=sys.stderr)
        return 2

    n = next(iter(lengths.values()))
    if args.limit:
        n = min(n, args.limit)
        paraphrases = {k: v[:n] for k, v in paraphrases.items()}
        references = references[:n]
        candidates = {k: v[:n] for k, v in candidates.items()}

    print(f"judges     : {ANTHROPIC_MODEL}, {OPENAI_MODEL}")
    print(f"threshold  : {args.threshold}")
    print(f"problems   : {n}")
    print(f"configs    : {', '.join(c[0] for c in configs)}")
    print(f"output     : {args.outdir}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # -- judge ---------------------------------------------------------------
    results: dict[str, dict[int, dict]] = {}

    if args.report_only:
        for name, _, _ in configs:
            results[name] = {r["index"]: r
                             for r in load_jsonl(args.outdir / f"{name}.jsonl")}
    else:
        judged = sum(1 for name, _, _ in configs
                     for i in range(n)
                     if candidates[name][i].strip())
        if not args.dry_run and not args.yes:
            print(f"\nAbout to make about {2 * judged} API calls "
                  f"(two judges on {judged} formalizations). This costs real "
                  f"money.")
            if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1

        clients = Clients(retries=args.retries, effort=args.effort)
        start = time.time()
        print()
        for name, _, variant in configs:
            results[name] = check_configuration(
                clients, name, candidates[name], paraphrases[variant],
                references, args.outdir, args)
        print(f"\njudging elapsed: {time.time() - start:.0f}s")

    # -- re-score --------------------------------------------------------
    # Statuses are derived from the stored judge responses rather than trusted
    # as written, so the report always reflects the rule and threshold in force
    # right now -- including when --report-only re-reads a finished run.
    changed = rescore(results, args.threshold)
    if changed:
        print(f"\nre-scored {changed} problem(s) against the current rule "
              f"(agreement + at least one judge >= {args.threshold:.2f})")

    if args.rewrite_status:
        if not args.report_only:
            print("note: --rewrite-status only applies with --report-only; "
                  "the .jsonl files were just written and are already current.")
        else:
            import shutil
            for name, _, _ in configs:
                path = args.outdir / f"{name}.jsonl"
                if not path.exists():
                    continue
                shutil.copy2(path, path.with_suffix(".jsonl.bak"))
                recs = results[name]
                with path.open("w", encoding="utf-8") as fh:
                    for i in sorted(recs):
                        fh.write(json.dumps(recs[i], ensure_ascii=False) + "\n")
                print(f"  rewrote {path} (backup at {path.with_suffix('.jsonl.bak')})")

    # -- report --------------------------------------------------------------
    summary = summarise(results, n, args.threshold)
    (args.outdir / "report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = render_report(summary)
    (args.outdir / "report.txt").write_text(report, encoding="utf-8")

    print()
    print(report)
    print(f"report written to {args.outdir / 'report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
