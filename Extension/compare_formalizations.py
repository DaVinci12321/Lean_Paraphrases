#!/usr/bin/env python3
"""
compare_formalizations.py -- stage 4 of the Lean Paraphrases extension.

For each model (gpt, claude) and each problem, compares the Lean 4 statement
produced from the RIGOROUS paraphrase against the one produced from the
NON-RIGOROUS paraphrase of the same problem, with both English paraphrases and
the curated reference statement in view.

The two paraphrases are mathematically equivalent by construction, so any
difference between the two formalizations comes from the wording, from model
nondeterminism, or from a mistake. Separating those three is the whole point of
the paper, and it is what this stage tries to do per problem:

    (a) every difference a reader of the mathematics would care about,
        categorised (variable domain, quantifier structure, hypothesis set,
        statement structure, encoding, numeric form, naming, answer form)
    (b) for each difference, whether it is attributable to specific wording in
        one paraphrase, with that wording quoted -- or whether it looks like
        arbitrary model variation
    (c) errors in each formalization, why they are wrong, and whether the
        paraphrase's wording likely caused them

Pairs whose two statements are byte-identical once the theorem name is
normalised away are recorded locally as "identical" without an API call.

The analyst is NOT shown the stage-3 equivalence verdicts. It reaches its own
error findings, and the report cross-tabulates the two so agreement between
them means something.

Output (in --outdir, default ./comparison)
-----------------------------------------
    gpt_comparison.jsonl        per-problem structured analysis
    claude_comparison.jsonl
    comparison_report.txt       aggregate statistics, all errors, notable cases
    comparison_details.txt      every pair, every difference, in full
    comparison_report.json      the same aggregates, machine-readable

Usage
-----
    python compare_formalizations.py --limit 10      # pilot
    python compare_formalizations.py                 # all 890 x 2
    python compare_formalizations.py --resume
    python compare_formalizations.py --report-only   # rebuild reports, no calls
    python compare_formalizations.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lp_common import (
    ANTHROPIC_MODEL, Clients, JsonlWriter, load_jsonl, read_lines,
    rebuild_lean, require_keys,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS = ("gpt", "claude")

CATEGORIES = [
    "variable_domain",       # N vs Z vs Q vs R, positivity, nonzero, finiteness
    "quantifier_structure",  # changed quantifier, order, or scope
    "hypothesis_set",        # hypotheses added, dropped or reformulated
    "statement_structure",   # iff vs implication, conjunction split, set vs predicate
    "definition_encoding",   # how an object is encoded: Finset vs Set, fn vs hypothesis
    "numeric_literal_form",  # 25/100 vs 1/4, decimal vs fraction
    "answer_representation", # how the asserted answer is expressed
    "naming_only",           # identifier choice echoing the wording
    "other",
]

# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

ANALYST_SYSTEM = """\
You are analysing how the wording of a natural-language mathematics problem \
affects the Lean 4 formalization a language model produces from it.

For ONE problem you are given:

  1. RIGOROUS PARAPHRASE      -- a symbol-heavy English statement
  2. NON-RIGOROUS PARAPHRASE  -- a word-heavy English statement of the SAME
                                 mathematics
  3. FORMALIZATION A          -- Lean 4, produced from the rigorous paraphrase
  4. FORMALIZATION B          -- Lean 4, produced from the non-rigorous one
  5. REFERENCE                -- a curated Lean 4 formalization of the
                                 underlying problem, known to compile

The two paraphrases are mathematically equivalent by construction. Every \
difference between A and B therefore traces to the wording, to arbitrary model \
variation, or to a mistake. Your job is to tell those apart.

(a) DIFFERENCES. List every difference between A and B that a reader of the \
mathematics would care about. Ignore pure alpha-renaming of bound variables, \
and ignore the theorem's own name -- both are noise. But DO report the choice \
of identifier names when the names echo the wording (say B uses `largest` and \
`smallest` where A used `a` and `b`): that is precisely the wording effect \
under study. Categorise it as naming_only with mathematically_significant \
false.

(b) ATTRIBUTION. For each difference, decide whether specific wording present \
in one paraphrase and absent from the other explains why the model went that \
way. If so, set attributable_to_paraphrase true and QUOTE the wording in \
paraphrase_evidence. If the difference looks like arbitrary variation with no \
wording trigger, set it false and say so. Do not invent an attribution to fill \
the field -- "no wording in either paraphrase accounts for this" is a useful \
and honest answer.

(c) ERRORS. An error is a formalization that fails to faithfully state the \
paraphrase it was produced from. Judge A against the RIGOROUS paraphrase and B \
against the NON-RIGOROUS one, using the reference for what the underlying \
problem means. Report the two sides separately. For each error, say why it is \
wrong, and whether the wording of its own paraphrase likely caused it.

Errors worth looking for, not an exhaustive list:
  * variable domain or type -- N, Z, Q and R are all different, and so are
    "positive", "non-negative" and "nonzero". An unqualified "number" in prose
    invites the wrong type; a worded description of a count invites N where the
    problem meant Z.
  * hypotheses the paraphrase never stated, or hypotheses it did state and the
    formalization dropped. Adding a standard-looking side condition the problem
    did not impose is a real error.
  * a changed quantifier, quantifier order, or scope.
  * an implication where the problem asserts a characterisation (iff), or the
    reverse; formalizing one direction of a two-way claim.
  * a claim strictly weaker than the problem's, or only part of the problem.
  * a statement that is vacuous or trivially true as written.
  * a wrong, dropped, or reshaped final answer.
  * a numeric form that changes the meaning -- integer division, truncation,
    a decimal that is not the stated fraction.

Do NOT call any of these an error: using `sorry` as the proof (that is \
required), omitting `import` or `open` lines, Mathlib naming or style choices, \
or an encoding that is longer or shorter but equivalent. You are not a \
compiler: judge meaning and surface well-formedness, not whether it would \
elaborate against a particular Mathlib version.

Be concrete and specific. A difference described as "different structure" is \
useless; "A states an iff over the solution set, B states only the forward \
implication" is what is wanted.\
"""

_DIFF_ITEM = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "description": {
            "type": "string",
            "description": "What differs, concretely, naming what A does and "
                           "what B does.",
        },
        "mathematically_significant": {
            "type": "boolean",
            "description": "True if the difference changes what is being "
                           "claimed; false for cosmetic or equivalent-encoding "
                           "differences.",
        },
        "attributable_to_paraphrase": {
            "type": "boolean",
            "description": "True only if wording present in one paraphrase and "
                           "absent from the other explains the difference.",
        },
        "paraphrase_evidence": {
            "type": "string",
            "description": "The wording that explains it, quoted. If not "
                           "attributable, say what makes it look like arbitrary "
                           "variation instead.",
        },
    },
    "required": ["category", "description", "mathematically_significant",
                 "attributable_to_paraphrase", "paraphrase_evidence"],
    "additionalProperties": False,
}

_ERROR_ITEM = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "description": {"type": "string",
                        "description": "The error, concretely."},
        "why_wrong": {
            "type": "string",
            "description": "Why this does not faithfully state its own "
                           "paraphrase.",
        },
        "caused_by_paraphrase": {
            "type": "boolean",
            "description": "True if the wording of its own paraphrase likely "
                           "led the model into this error.",
        },
        "paraphrase_cause": {
            "type": "string",
            "description": "The wording that likely led to it, quoted. Empty if "
                           "the error is not wording-driven.",
        },
    },
    "required": ["category", "description", "why_wrong",
                 "caused_by_paraphrase", "paraphrase_cause"],
    "additionalProperties": False,
}

COMPARISON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string",
            "description": "Work through A and B clause by clause against their "
                           "own paraphrases before filling in the lists below.",
        },
        "differences": {"type": "array", "items": _DIFF_ITEM},
        "rigorous_errors": {"type": "array", "items": _ERROR_ITEM},
        "nonrigorous_errors": {"type": "array", "items": _ERROR_ITEM},
        "any_semantic_difference": {
            "type": "boolean",
            "description": "True if A and B do not state the same mathematics.",
        },
        "paraphrase_effect_summary": {
            "type": "string",
            "description": "One or two sentences: what the rewording did to "
                           "this formalization, or that it did nothing.",
        },
    },
    "required": ["analysis", "differences", "rigorous_errors",
                 "nonrigorous_errors", "any_semantic_difference",
                 "paraphrase_effect_summary"],
    "additionalProperties": False,
}


def analyst_prompt(rig_nl: str, non_nl: str, rig_lean: str, non_lean: str,
                   reference: str) -> str:
    return "\n".join([
        "RIGOROUS PARAPHRASE", rig_nl, "",
        "NON-RIGOROUS PARAPHRASE", non_nl, "",
        "FORMALIZATION A (from the rigorous paraphrase)",
        rebuild_lean("", rig_lean), "",
        "FORMALIZATION B (from the non-rigorous paraphrase)",
        rebuild_lean("", non_lean), "",
        "REFERENCE (curated formalization of the underlying problem, context "
        "only)", rebuild_lean("", reference), "",
        "Analyse the differences, their attribution, and any errors.",
    ])


# --------------------------------------------------------------------------
# Local helpers
# --------------------------------------------------------------------------

def normalise(statement: str) -> str:
    """Drop the theorem's own name so a pure naming difference is not a diff."""
    return re.sub(r"^(theorem|lemma|example)\s+\S+", r"\1 T",
                  statement.strip()).strip()


def fake_analysis(index: int) -> dict:
    return {
        "analysis": "dry run",
        "differences": [{
            "category": "variable_domain",
            "description": f"stub difference for problem {index}",
            "mathematically_significant": index % 3 == 0,
            "attributable_to_paraphrase": index % 2 == 0,
            "paraphrase_evidence": "stub evidence",
        }],
        "rigorous_errors": ([] if index % 5 else [{
            "category": "variable_domain", "description": "stub error",
            "why_wrong": "stub", "caused_by_paraphrase": True,
            "paraphrase_cause": "stub wording"}]),
        "nonrigorous_errors": ([] if index % 4 else [{
            "category": "hypothesis_set", "description": "stub error",
            "why_wrong": "stub", "caused_by_paraphrase": False,
            "paraphrase_cause": ""}]),
        "any_semantic_difference": index % 3 == 0,
        "paraphrase_effect_summary": "dry run",
    }


def analyse_one(clients: Clients, model: str, index: int, rig_nl: str,
                non_nl: str, rig_lean: str, non_lean: str, reference: str,
                dry_run: bool) -> dict:
    rec: dict[str, Any] = {
        "index": index, "model": model,
        "rigorous_paraphrase": rig_nl, "nonrigorous_paraphrase": non_nl,
        "rigorous_formalization": rig_lean, "nonrigorous_formalization": non_lean,
        "reference": reference, "status": "", "analysis": {},
    }

    if not rig_lean.strip() or not non_lean.strip():
        rec["status"] = "missing_formalization"
        return rec

    if normalise(rig_lean) == normalise(non_lean):
        # Identical up to the theorem name: no API call needed, and recording it
        # as a real datum matters -- "the wording changed nothing" is a finding.
        rec["status"] = "identical"
        rec["analysis"] = {
            "analysis": "The two statements are identical once the theorem "
                        "name is normalised away.",
            "differences": [], "rigorous_errors": [], "nonrigorous_errors": [],
            "any_semantic_difference": False,
            "paraphrase_effect_summary": "No effect: the rewording produced "
                                         "the same Lean statement.",
        }
        return rec

    try:
        rec["analysis"] = (
            fake_analysis(index) if dry_run else clients.anthropic_json(
                ANALYST_SYSTEM,
                analyst_prompt(rig_nl, non_nl, rig_lean, non_lean, reference),
                COMPARISON_SCHEMA, "comparison", model=ANTHROPIC_MODEL))
        rec["status"] = "analysed"
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def run_model(clients: Clients, model: str, rig_nl: list[str],
              non_nl: list[str], references: list[str], outdir: Path,
              args) -> dict[int, dict]:
    rig_lean = read_lines(args.formalized_dir / model / "formalized_rigorous.txt")
    non_lean = read_lines(args.formalized_dir / model / "formalized_nonrigorous.txt")
    n = len(rig_nl)
    jsonl_path = outdir / f"{model}_comparison.jsonl"

    done: dict[int, dict] = {}
    if args.resume:
        for r in load_jsonl(jsonl_path):
            i = r.get("index", 0)
            if not 1 <= i <= n or r.get("status") == "error":
                continue
            # Only reuse an analysis of the same two statements.
            if (r.get("rigorous_formalization") != rig_lean[i - 1]
                    or r.get("nonrigorous_formalization") != non_lean[i - 1]):
                continue
            done[i] = r
        print(f"  {model}: resuming with {len(done)} already analysed")
    elif jsonl_path.exists():
        print(f"error: {jsonl_path} already exists. Pass --resume to continue, "
              f"or choose a different --outdir.", file=sys.stderr)
        raise SystemExit(2)

    todo = [i for i in range(1, n + 1) if i not in done]
    writer = JsonlWriter(jsonl_path, append=args.resume)
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(todo), unit="pair", desc=model)
    except ImportError:
        bar = None

    def work(i: int) -> dict:
        rec = analyse_one(clients, model, i, rig_nl[i - 1], non_nl[i - 1],
                          rig_lean[i - 1], non_lean[i - 1], references[i - 1],
                          args.dry_run)
        writer.write(rec)
        if bar:
            bar.update(1)
        return rec

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rec in pool.map(work, todo):
                done[rec["index"]] = rec
    except KeyboardInterrupt:
        print("\ninterrupted -- reporting on what finished", file=sys.stderr)
    finally:
        if bar:
            bar.close()
        writer.close()
    return done


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def summarise(results: dict[str, dict[int, dict]], verdicts: dict[str, dict],
              n: int) -> dict:
    out: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analyst_model": ANTHROPIC_MODEL,
        "problems": n,
        "per_model": {},
    }

    for model, recs in results.items():
        status = Counter(r.get("status", "not_attempted")
                         for r in recs.values())
        diff_cat = Counter()
        diff_cat_attrib = Counter()
        sig_attrib = sig_not_attrib = 0
        err_cat = {"rigorous": Counter(), "nonrigorous": Counter()}
        err_caused = {"rigorous": 0, "nonrigorous": 0}
        err_total = {"rigorous": 0, "nonrigorous": 0}
        problems_with_error = {"rigorous": set(), "nonrigorous": set()}
        semantic_diff, any_diff = [], []

        for i, rec in recs.items():
            a = rec.get("analysis") or {}
            diffs = a.get("differences") or []
            if diffs:
                any_diff.append(i)
            if a.get("any_semantic_difference"):
                semantic_diff.append(i)
            for d in diffs:
                cat = d.get("category", "other")
                diff_cat[cat] += 1
                if d.get("attributable_to_paraphrase"):
                    diff_cat_attrib[cat] += 1
                if d.get("mathematically_significant"):
                    if d.get("attributable_to_paraphrase"):
                        sig_attrib += 1
                    else:
                        sig_not_attrib += 1
            for side in ("rigorous", "nonrigorous"):
                errs = a.get(f"{side}_errors") or []
                if errs:
                    problems_with_error[side].add(i)
                for e in errs:
                    err_total[side] += 1
                    err_cat[side][e.get("category", "other")] += 1
                    if e.get("caused_by_paraphrase"):
                        err_caused[side] += 1

        total_diffs = sum(diff_cat.values())
        total_attrib = sum(diff_cat_attrib.values())

        # Does the analyst's error finding line up with stage 3's verdict?
        cross = Counter()
        for side in ("rigorous", "nonrigorous"):
            v = verdicts.get(f"{model}_{side}", {})
            for i in recs:
                judged_bad = v.get(i) == "NOT_EQUIVALENT"
                found_bad = i in problems_with_error[side]
                cross[f"{side}: {'judged wrong' if judged_bad else 'judged ok'}"
                      f" / {'analyst found error' if found_bad else 'analyst found none'}"] += 1

        out["per_model"][model] = {
            "status_counts": dict(status),
            "pairs_with_any_difference": len(any_diff),
            "pairs_identical": status.get("identical", 0),
            "pairs_with_semantic_difference": len(semantic_diff),
            "semantic_difference_indices": sorted(semantic_diff),
            "differences_total": total_diffs,
            "differences_attributable_to_paraphrase": total_attrib,
            "attribution_rate": (round(total_attrib / total_diffs, 4)
                                 if total_diffs else None),
            "differences_by_category": dict(diff_cat.most_common()),
            "attributable_by_category": dict(diff_cat_attrib.most_common()),
            "significant_differences_attributable": sig_attrib,
            "significant_differences_not_attributable": sig_not_attrib,
            "errors": {
                side: {
                    "problems_with_at_least_one_error":
                        len(problems_with_error[side]),
                    "error_indices": sorted(problems_with_error[side]),
                    "errors_total": err_total[side],
                    "errors_attributed_to_paraphrase": err_caused[side],
                    "by_category": dict(err_cat[side].most_common()),
                } for side in ("rigorous", "nonrigorous")
            },
            "analyst_vs_stage3": dict(sorted(cross.items())),
        }
    return out


def fmt_indices(idx: list[int]) -> str:
    if not idx:
        return "(none)"
    parts, start, prev = [], idx[0], idx[0]
    for i in idx[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    return ", ".join(parts)


def render_report(summary: dict, results: dict, verdicts: dict) -> str:
    L: list[str] = []
    add = L.append
    add("Lean Paraphrases -- rigorous vs non-rigorous formalization comparison")
    add("=" * 74)
    add(f"generated : {summary['generated_at']}")
    add(f"analyst   : {summary['analyst_model']}")
    add(f"problems  : {summary['problems']} per model")
    add("")
    add("For each problem the Lean statement produced from the rigorous")
    add("paraphrase is compared against the one produced from the non-rigorous")
    add("paraphrase, with both English statements and the curated reference in")
    add("view. The paraphrases are equivalent by construction, so a difference")
    add("is either wording-driven, arbitrary model variation, or a mistake.")
    add("")
    add("The analyst was not shown the stage-3 equivalence verdicts; the last")
    add("table in each model's section cross-tabulates the two independently.")
    add("")

    # Coverage first. Statistics computed over a fraction of the pairs are
    # worse than no statistics, because they look complete.
    incomplete = False
    for model, s in summary["per_model"].items():
        st = s["status_counts"]
        bad = st.get("error", 0) + st.get("not_attempted", 0)
        if bad:
            incomplete = True
    if incomplete:
        add("!" * 74)
        add("!!  WARNING: THIS REPORT IS INCOMPLETE")
        add("!" * 74)
        for model, s in summary["per_model"].items():
            st = s["status_counts"]
            add(f"!!  {model}: analysed={st.get('analysed', 0)} "
                f"identical={st.get('identical', 0)} "
                f"FAILED={st.get('error', 0)} "
                f"missing={st.get('missing_formalization', 0)} "
                f"not_attempted={st.get('not_attempted', 0)}")
        add("!!")
        add("!!  Every rate and count below is computed ONLY over the pairs")
        add("!!  that were actually analysed. Do not quote these numbers.")
        add("!!  Re-run with --resume once the failures are resolved.")
        add("!" * 74)
        add("")

    for model, s in summary["per_model"].items():
        add("=" * 74)
        add(f"MODEL: {model}")
        add("=" * 74)
        add("")
        add("COVERAGE")
        add("-" * 74)
        st = s["status_counts"]
        for k in ("analysed", "identical", "missing_formalization", "error",
                  "not_attempted"):
            if st.get(k):
                add(f"  {k:<24}{st[k]:>5}")
        usable = st.get("analysed", 0) + st.get("identical", 0)
        add(f"  {'usable for statistics':<24}{usable:>5} / {summary['problems']}")
        add("")
        add("HOW OFTEN THE WORDING CHANGED THE OUTPUT")
        add("-" * 74)
        n = summary["problems"]
        add(f"  identical Lean statement from both wordings : "
            f"{s['pairs_identical']:>4} / {n}")
        add(f"  at least one reportable difference          : "
            f"{s['pairs_with_any_difference']:>4} / {n}")
        add(f"  the two state different mathematics         : "
            f"{s['pairs_with_semantic_difference']:>4} / {n}")
        add("")
        add(f"  differences recorded in total               : {s['differences_total']}")
        rate = s["attribution_rate"]
        add(f"  attributable to the paraphrase wording      : "
            f"{s['differences_attributable_to_paraphrase']}"
            f"{f' ({rate:.1%})' if rate is not None else ''}")
        add(f"  mathematically significant AND attributable : "
            f"{s['significant_differences_attributable']}")
        add(f"  significant but NOT attributable (variation) : "
            f"{s['significant_differences_not_attributable']}")
        add("")
        add("DIFFERENCES BY CATEGORY")
        add("-" * 74)
        add(f"  {'category':<24}{'total':>8}{'attributable':>14}{'rate':>8}")
        for cat, tot in s["differences_by_category"].items():
            att = s["attributable_by_category"].get(cat, 0)
            add(f"  {cat:<24}{tot:>8}{att:>14}{att / tot:>7.0%}")
        add("")
        add("ERRORS")
        add("-" * 74)
        for side in ("rigorous", "nonrigorous"):
            e = s["errors"][side]
            add(f"  from the {side} paraphrase:")
            add(f"    problems with >= 1 error   : "
                f"{e['problems_with_at_least_one_error']}")
            add(f"    errors in total            : {e['errors_total']}")
            add(f"    attributed to the wording  : "
                f"{e['errors_attributed_to_paraphrase']}")
            for cat, c in e["by_category"].items():
                add(f"      {cat:<24}{c:>5}")
            add(f"    problem numbers            : {fmt_indices(e['error_indices'])}")
            add("")
        add("ANALYST FINDINGS vs STAGE-3 VERDICTS")
        add("-" * 74)
        for k, v in s["analyst_vs_stage3"].items():
            add(f"  {k:<62}{v:>5}")
        add("")

        # Every error, in full, because that is the actionable part.
        add("EVERY ERROR, IN FULL")
        add("-" * 74)
        for side in ("rigorous", "nonrigorous"):
            for i in s["errors"][side]["error_indices"]:
                rec = results[model][i]
                a = rec.get("analysis") or {}
                v = verdicts.get(f"{model}_{side}", {}).get(i, "?")
                add(f"#{i} [{side}]  (stage-3 verdict: {v})")
                para = (rec["rigorous_paraphrase"] if side == "rigorous"
                        else rec["nonrigorous_paraphrase"])
                lean = (rec["rigorous_formalization"] if side == "rigorous"
                        else rec["nonrigorous_formalization"])
                add(f"   paraphrase : {para[:400]}")
                add(f"   Lean       : {lean[:400]}")
                for e in a.get(f"{side}_errors") or []:
                    add(f"   ERROR [{e.get('category')}] {e.get('description')}")
                    add(f"     why  : {e.get('why_wrong')}")
                    if e.get("caused_by_paraphrase"):
                        add(f"     WORDING-DRIVEN: {e.get('paraphrase_cause')}")
                    else:
                        add(f"     not attributed to the wording")
                add("")
    return "\n".join(L) + "\n"


def render_details(results: dict, verdicts: dict) -> str:
    L: list[str] = []
    add = L.append
    add("Lean Paraphrases -- per-problem formalization differences")
    add("=" * 74)
    add("Every analysed pair, every difference. `ATTRIB` marks a difference the")
    add("analyst traced to specific wording; `SIG` marks one that changes the")
    add("mathematics.")
    add("")
    for model, recs in results.items():
        add("#" * 74)
        add(f"MODEL: {model}")
        add("#" * 74)
        for i in sorted(recs):
            rec = recs[i]
            a = rec.get("analysis") or {}
            diffs = a.get("differences") or []
            vr = verdicts.get(f"{model}_rigorous", {}).get(i, "?")
            vn = verdicts.get(f"{model}_nonrigorous", {}).get(i, "?")
            add("")
            add("-" * 74)
            add(f"#{i}  status={rec.get('status')}  "
                f"stage-3: rigorous={vr}, nonrigorous={vn}")
            add(f"  RIGOROUS NL    : {rec['rigorous_paraphrase'][:320]}")
            add(f"  NONRIGOROUS NL : {rec['nonrigorous_paraphrase'][:320]}")
            add(f"  LEAN A (rig)   : {rec['rigorous_formalization'][:320]}")
            add(f"  LEAN B (non)   : {rec['nonrigorous_formalization'][:320]}")
            add(f"  effect         : {a.get('paraphrase_effect_summary','')}")
            if not diffs:
                add("  differences    : none")
            for d in diffs:
                tags = []
                if d.get("attributable_to_paraphrase"):
                    tags.append("ATTRIB")
                if d.get("mathematically_significant"):
                    tags.append("SIG")
                add(f"  [{','.join(tags) or '-':<11}] {d.get('category')}: "
                    f"{d.get('description')}")
                if d.get("paraphrase_evidence"):
                    add(f"                evidence: {d.get('paraphrase_evidence')}")
            for side in ("rigorous", "nonrigorous"):
                for e in a.get(f"{side}_errors") or []:
                    add(f"  ERROR ({side}) [{e.get('category')}]: "
                        f"{e.get('description')}")
                    add(f"                why: {e.get('why_wrong')}")
                    if e.get("caused_by_paraphrase"):
                        add(f"                wording-driven: "
                            f"{e.get('paraphrase_cause')}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compare each model's rigorous vs non-rigorous Lean "
                    "formalizations and attribute the differences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-dir", type=Path, default=SCRIPT_DIR / "dataset")
    p.add_argument("--formalized-dir", type=Path,
                   default=SCRIPT_DIR / "formalized")
    p.add_argument("--evaluation-dir", type=Path,
                   default=SCRIPT_DIR / "evaluation")
    p.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "comparison")
    p.add_argument("--limit", type=int, default=0,
                   help="only analyse the first N problems (0 = all)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--effort", default="medium",
                   choices=("low", "medium", "high", "xhigh", "max"))
    p.add_argument("--only", nargs="*", choices=MODELS, default=None,
                   help="analyse only these models")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--report-only", action="store_true",
                   help="rebuild the reports from existing .jsonl, no API calls")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args(argv)

    if not (args.dry_run or args.report_only):
        missing = require_keys("ANTHROPIC_API_KEY")
        if missing:
            print(f"error: missing {', '.join(missing)}", file=sys.stderr)
            return 2

    try:
        rig_nl = read_lines(args.dataset_dir / "rigorousparaphrases.txt")
        non_nl = read_lines(args.dataset_dir / "nonrigorousparaphrases.txt")
        references = read_lines(args.dataset_dir / "formalstatements.txt")
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    models = list(args.only) if args.only else list(MODELS)
    sizes = {"rigorous": len(rig_nl), "nonrigorous": len(non_nl),
             "reference": len(references)}
    for m in models:
        for v in ("rigorous", "nonrigorous"):
            path = args.formalized_dir / m / f"formalized_{v}.txt"
            try:
                sizes[f"{m}/{v}"] = len(read_lines(path))
            except FileNotFoundError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
    if len(set(sizes.values())) != 1:
        print(f"error: files are not line-aligned: {sizes}", file=sys.stderr)
        return 2

    n = next(iter(sizes.values()))
    if args.limit:
        n = min(n, args.limit)
    rig_nl, non_nl, references = rig_nl[:n], non_nl[:n], references[:n]

    # Stage-3 verdicts, for the cross-tabulation only.
    verdicts: dict[str, dict[int, str]] = {}
    for m in models:
        for v in ("rigorous", "nonrigorous"):
            path = args.evaluation_dir / f"{m}_{v}.jsonl"
            verdicts[f"{m}_{v}"] = {
                r["index"]: r.get("status", "?")
                for r in load_jsonl(path)} if path.exists() else {}

    print(f"analyst   : {ANTHROPIC_MODEL} (effort {args.effort})")
    print(f"problems  : {n} per model")
    print(f"models    : {', '.join(models)}")
    print(f"output    : {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[int, dict]] = {}
    if args.report_only:
        for m in models:
            results[m] = {r["index"]: r for r in
                          load_jsonl(args.outdir / f"{m}_comparison.jsonl")
                          if r.get("index", 0) <= n}
    else:
        # Byte-identical pairs need no API call; count the rest.
        need = 0
        for m in models:
            a = read_lines(args.formalized_dir / m / "formalized_rigorous.txt")
            b = read_lines(args.formalized_dir / m / "formalized_nonrigorous.txt")
            need += sum(1 for i in range(n)
                        if a[i].strip() and b[i].strip()
                        and normalise(a[i]) != normalise(b[i]))
        if not args.dry_run and not args.yes:
            print(f"\n{need} pairs differ and need an API call "
                  f"({2 * n - need} are identical or missing and are handled "
                  f"locally for free). This costs real money.")
            if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1

        clients = Clients(retries=args.retries, effort=args.effort,
                          max_tokens=16000)
        start = time.time()
        print()
        for m in models:
            results[m] = run_model(clients, m, rig_nl, non_nl, references,
                                   args.outdir, args)
        print(f"\nanalysis elapsed: {time.time() - start:.0f}s")

    summary = summarise(results, verdicts, n)
    (args.outdir / "comparison_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = render_report(summary, results, verdicts)
    (args.outdir / "comparison_report.txt").write_text(report, encoding="utf-8")
    (args.outdir / "comparison_details.txt").write_text(
        render_details(results, verdicts), encoding="utf-8")

    print()
    print(report[:4000])
    print(f"...\nfull report  : {args.outdir / 'comparison_report.txt'}")
    print(f"per-problem  : {args.outdir / 'comparison_details.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
