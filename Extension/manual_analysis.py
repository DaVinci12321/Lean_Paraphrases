#!/usr/bin/env python3
"""
manual_analysis.py -- harness for analysing the formalization pairs in-session.

compare_formalizations.py farms each pair out to the API. This does the same job
without any API calls: it hands batches of pairs to whoever is reading (here,
Claude Code itself), then ingests the written analysis back into exactly the
schema compare_formalizations.py produces. That means the existing reporting
path works unchanged:

    python compare_formalizations.py --report-only

Batches are ordered by value, so stopping part-way still leaves the cases the
paper leans on already done:

    tier 1  a stage-3 verdict was NOT_EQUIVALENT, or the verdict flipped
            between the two wordings
    tier 2  the two Lean statements are structurally different (similarity
            below 0.80)
    tier 3  everything else that is not byte-identical

Usage
-----
    python manual_analysis.py status
    python manual_analysis.py next --size 25          # print the next batch
    python manual_analysis.py ingest batch.json       # record the analysis
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

from lp_common import load_jsonl, read_lines

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "dataset"
FORM = SCRIPT_DIR / "formalized"
EVAL = SCRIPT_DIR / "evaluation"
OUT = SCRIPT_DIR / "comparison"
MODELS = ("gpt", "claude")

CATEGORIES = [
    "variable_domain", "quantifier_structure", "hypothesis_set",
    "statement_structure", "definition_encoding", "numeric_literal_form",
    "answer_representation", "naming_only", "other",
]


def normalise(s: str) -> str:
    return re.sub(r"^(theorem|lemma|example)\s+\S+", r"\1 T", s.strip()).strip()


def load_all() -> dict:
    rig_nl = read_lines(DATA / "rigorousparaphrases.txt")
    non_nl = read_lines(DATA / "nonrigorousparaphrases.txt")
    ref = read_lines(DATA / "formalstatements.txt")
    lean = {(m, v): read_lines(FORM / m / f"formalized_{v}.txt")
            for m in MODELS for v in ("rigorous", "nonrigorous")}
    verdict = {}
    for m in MODELS:
        for v in ("rigorous", "nonrigorous"):
            path = EVAL / f"{m}_{v}.jsonl"
            verdict[(m, v)] = {r["index"]: r.get("status", "?")
                               for r in load_jsonl(path)} if path.exists() else {}
    return {"rig_nl": rig_nl, "non_nl": non_nl, "ref": ref, "lean": lean,
            "verdict": verdict, "n": len(rig_nl)}


def tier_of(d: dict, m: str, i: int) -> int:
    a, b = d["lean"][(m, "rigorous")][i - 1], d["lean"][(m, "nonrigorous")][i - 1]
    vr, vn = d["verdict"][(m, "rigorous")].get(i), d["verdict"][(m, "nonrigorous")].get(i)
    if "NOT_EQUIVALENT" in (vr, vn) or (vr and vn and vr != vn):
        return 1
    return 2 if SequenceMatcher(None, normalise(a), normalise(b)).ratio() < 0.80 else 3


def pending(d: dict) -> list[tuple[int, str, int]]:
    """-> [(tier, model, index)] still needing analysis, best first."""
    have = {m: {r["index"] for r in load_jsonl(OUT / f"{m}_comparison.jsonl")
                if r.get("status") in ("analysed", "identical",
                                       "missing_formalization")}
            for m in MODELS}
    todo = []
    for m in MODELS:
        a, b = d["lean"][(m, "rigorous")], d["lean"][(m, "nonrigorous")]
        for i in range(1, d["n"] + 1):
            if i in have[m]:
                continue
            if not a[i - 1].strip() or not b[i - 1].strip():
                continue
            if normalise(a[i - 1]) == normalise(b[i - 1]):
                continue
            todo.append((tier_of(d, m, i), m, i))
    todo.sort(key=lambda t: (t[0], t[1], t[2]))
    return todo


def cmd_status(d: dict) -> int:
    todo = pending(d)
    from collections import Counter
    print(f"pairs still to analyse : {len(todo)}")
    print(f"  by tier   : {dict(sorted(Counter(t for t, _, _ in todo).items()))}")
    print(f"  by model  : {dict(Counter(m for _, m, _ in todo))}")
    for m in MODELS:
        recs = load_jsonl(OUT / f"{m}_comparison.jsonl")
        c = Counter(r.get("status") for r in recs)
        print(f"  {m}: {dict(c)}")
    return 0


def cmd_next(d: dict, size: int) -> int:
    todo = pending(d)[:size]
    if not todo:
        print("nothing left to analyse")
        return 0
    print(f"### BATCH: {len(todo)} pairs (tiers "
          f"{sorted({t for t, _, _ in todo})})\n")
    for tier, m, i in todo:
        vr = d["verdict"][(m, "rigorous")].get(i, "?")
        vn = d["verdict"][(m, "nonrigorous")].get(i, "?")
        print("=" * 78)
        print(f"KEY {m}:{i}   tier={tier}   stage3: rig={vr} non={vn}")
        print(f"-- RIGOROUS NL --\n{d['rig_nl'][i - 1]}")
        print(f"-- NONRIGOROUS NL --\n{d['non_nl'][i - 1]}")
        print(f"-- LEAN A (from rigorous) --\n{d['lean'][(m, 'rigorous')][i - 1]}")
        print(f"-- LEAN B (from nonrigorous) --\n{d['lean'][(m, 'nonrigorous')][i - 1]}")
        print(f"-- REFERENCE --\n{d['ref'][i - 1]}")
    print("=" * 78)
    print(f"\nkeys in this batch: {[f'{m}:{i}' for _, m, i in todo]}")
    return 0


REQUIRED_DIFF = {"category", "description", "mathematically_significant",
                 "attributable_to_paraphrase", "paraphrase_evidence"}
REQUIRED_ERR = {"category", "description", "why_wrong",
                "caused_by_paraphrase", "paraphrase_cause"}


def cmd_ingest(d: dict, path: Path, replace: bool = False) -> int:
    """Ingest {"gpt:12": {...analysis...}, ...} into the per-model jsonl.

    With `replace`, any record already stored for an incoming (model, index)
    is dropped first, so a corrected analysis supersedes the earlier one
    instead of being appended alongside it.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_model: dict[str, list[dict]] = {m: [] for m in MODELS}
    problems = []

    for key, a in payload.items():
        try:
            m, idx = key.split(":")
            i = int(idx)
        except ValueError:
            problems.append(f"{key}: malformed key, want 'model:index'")
            continue
        if m not in MODELS or not 1 <= i <= d["n"]:
            problems.append(f"{key}: unknown model or index out of range")
            continue
        missing = {"differences", "rigorous_errors", "nonrigorous_errors",
                   "any_semantic_difference", "paraphrase_effect_summary"
                   } - set(a)
        if missing:
            problems.append(f"{key}: missing fields {sorted(missing)}")
            continue
        for dd in a["differences"]:
            if REQUIRED_DIFF - set(dd):
                problems.append(f"{key}: difference missing "
                                f"{sorted(REQUIRED_DIFF - set(dd))}")
            if dd.get("category") not in CATEGORIES:
                problems.append(f"{key}: bad category {dd.get('category')!r}")
        for side in ("rigorous_errors", "nonrigorous_errors"):
            for ee in a[side]:
                if REQUIRED_ERR - set(ee):
                    problems.append(f"{key}: {side} missing "
                                    f"{sorted(REQUIRED_ERR - set(ee))}")
                if ee.get("category") not in CATEGORIES:
                    problems.append(f"{key}: bad category {ee.get('category')!r}")
        a.setdefault("analysis", "analysed in-session by Claude Code")
        by_model[m].append({
            "index": i, "model": m,
            "rigorous_paraphrase": d["rig_nl"][i - 1],
            "nonrigorous_paraphrase": d["non_nl"][i - 1],
            "rigorous_formalization": d["lean"][(m, "rigorous")][i - 1],
            "nonrigorous_formalization": d["lean"][(m, "nonrigorous")][i - 1],
            "reference": d["ref"][i - 1],
            "status": "analysed", "analyst": "claude-code-in-session",
            "analysis": a,
        })

    if problems:
        print("REJECTED -- nothing was written:", file=sys.stderr)
        for p in problems[:25]:
            print(f"  {p}", file=sys.stderr)
        return 2

    for m, recs in by_model.items():
        if not recs:
            continue
        out = OUT / f"{m}_comparison.jsonl"
        dropped = 0
        if replace and out.exists():
            incoming = {r["index"] for r in recs}
            kept = [ln for ln in out.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and json.loads(ln).get("index") not in incoming]
            dropped = sum(1 for ln in out.read_text(encoding="utf-8").splitlines()
                          if ln.strip()) - len(kept)
            out.write_text("".join(ln + "\n" for ln in kept), encoding="utf-8")
        with out.open("a", encoding="utf-8") as fh:
            for r in sorted(recs, key=lambda r: r["index"]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        note = f" (replaced {dropped})" if dropped else ""
        print(f"  {m}: +{len(recs)} records{note}")
    print(f"remaining: {len(pending(d))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    nx = sub.add_parser("next")
    nx.add_argument("--size", type=int, default=25)
    ing = sub.add_parser("ingest")
    ing.add_argument("path", type=Path)
    ing.add_argument("--replace", action="store_true",
                     help="supersede any stored analysis for the same pairs")
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    d = load_all()
    if args.cmd == "status":
        return cmd_status(d)
    if args.cmd == "next":
        return cmd_next(d, args.size)
    return cmd_ingest(d, args.path, args.replace)


if __name__ == "__main__":
    raise SystemExit(main())
