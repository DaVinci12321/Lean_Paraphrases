#!/usr/bin/env python3
"""
prune_disagreements.py -- drop every problem the judges split on.

A problem is removed when ANY of the four configurations (gpt_rigorous,
gpt_nonrigorous, claude_rigorous, claude_nonrigorous) ended in a judge
disagreement. The surviving set is one where all four configurations reached a
definite verdict, which is what the paper's per-problem comparisons need: a
problem that is undecided in one configuration cannot be compared across
wordings or across models.

Everything downstream is filtered and renumbered together so the line-aligned
contract still holds:

    dataset/rigorousparaphrases.txt      dataset/dataset.jsonl
    dataset/nonrigorousparaphrases.txt   dataset/records.jsonl
    dataset/formalstatements.txt         dataset/summary.json
    formalized/{claude,gpt}/formalized_{rigorous,nonrigorous}.{txt,jsonl}
    evaluation/{configuration}.jsonl

Nothing is thrown away. Before any file is touched, the whole tree is copied to
removed_problems/original_backup/, and every removed problem is written out in
full -- both paraphrases, the reference statement, all four formalizations, and
all eight judge responses with their analyses -- to removed_problems/.

The report is not regenerated here; run this afterwards:

    python check_autoformalizations.py --report-only

Usage
-----
    python prune_disagreements.py --dry-run    # show what would go
    python prune_disagreements.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from lp_common import load_jsonl, read_lines, write_lines

SCRIPT_DIR = Path(__file__).resolve().parent

CONFIGURATIONS = [
    ("gpt_rigorous", "gpt", "rigorous"),
    ("gpt_nonrigorous", "gpt", "nonrigorous"),
    ("claude_rigorous", "claude", "rigorous"),
    ("claude_nonrigorous", "claude", "nonrigorous"),
]
ALIGNED_TXT = ["rigorousparaphrases.txt", "nonrigorousparaphrases.txt",
               "formalstatements.txt"]


def disagreed(rec: dict) -> bool:
    return (rec.get("status") == "UNDECIDED"
            and str(rec.get("reason", "")).startswith("judges disagreed"))


def renumber(records: list[dict], keep: list[int]) -> list[dict]:
    """Filter records by original index and renumber them 1..len(keep)."""
    new_of = {old: i for i, old in enumerate(keep, 1)}
    out = []
    for rec in sorted(records, key=lambda r: r.get("index", 0)):
        old = rec.get("index")
        if old not in new_of:
            continue
        rec = dict(rec)
        rec["original_index"] = old
        rec["index"] = new_of[old]
        out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Remove every problem any configuration's judges "
                    "disagreed on, from the dataset, the formalizations and "
                    "the evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--workdir", type=Path, default=SCRIPT_DIR,
                   help="root holding dataset/, formalized/ and evaluation/")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be removed and change nothing")
    p.add_argument("--force", action="store_true",
                   help="run even though removed_problems/ already exists")
    args = p.parse_args(argv)

    wd = args.workdir.resolve()
    out = wd / "removed_problems"
    if out.exists() and not (args.dry_run or args.force):
        print(f"error: {out} already exists -- this looks like it has already "
              f"been pruned.\n       Pass --force to prune again (the backup "
              f"inside will be overwritten).", file=sys.stderr)
        return 2

    # -- load ---------------------------------------------------------------
    try:
        aligned = {f: read_lines(wd / "dataset" / f) for f in ALIGNED_TXT}
        dataset = load_jsonl(wd / "dataset" / "dataset.jsonl")
        evaluation = {c: load_jsonl(wd / "evaluation" / f"{c}.jsonl")
                      for c, _, _ in CONFIGURATIONS}
        formal_txt, formal_jsonl = {}, {}
        for _, model, variant in CONFIGURATIONS:
            d = wd / "formalized" / model
            formal_txt[(model, variant)] = read_lines(
                d / f"formalized_{variant}.txt")
            formal_jsonl[(model, variant)] = load_jsonl(
                d / f"formalized_{variant}.jsonl")
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    n = len(aligned[ALIGNED_TXT[0]])
    sizes = {f: len(v) for f, v in aligned.items()}
    sizes |= {f"{m}/{v}": len(x) for (m, v), x in formal_txt.items()}
    if len(set(sizes.values())) != 1:
        print(f"error: files are not line-aligned: {sizes}", file=sys.stderr)
        return 2

    # -- decide what goes ---------------------------------------------------
    by_config = {c: {r["index"] for r in recs if disagreed(r)}
                 for c, recs in evaluation.items()}
    removed = sorted(set().union(*by_config.values()))
    keep = [i for i in range(1, n + 1) if i not in set(removed)]

    print(f"problems              : {n}")
    for c, _, _ in CONFIGURATIONS:
        print(f"  {c:<22}{len(by_config[c]):>4} disagreements")
    print(f"union to remove       : {len(removed)}")
    print(f"remaining             : {len(keep)}")

    if args.dry_run:
        print(f"\nwould remove: {removed}")
        print("\n(dry run -- nothing changed)")
        return 0

    # -- back up before touching anything -----------------------------------
    backup = out / "original_backup"
    if backup.exists():
        shutil.rmtree(backup)
    backup.mkdir(parents=True)
    for sub in ("dataset", "formalized", "evaluation"):
        if (wd / sub).exists():
            shutil.copytree(wd / sub, backup / sub)
    print(f"\nbacked up dataset/, formalized/ and evaluation/ to {backup}")

    # -- write the removed problems out in full -----------------------------
    ds_by_index = {r["index"]: r for r in dataset}
    ev_by_index = {c: {r["index"]: r for r in recs}
                   for c, recs in evaluation.items()}

    detail = []
    for idx in removed:
        d = ds_by_index.get(idx, {})
        rec = {
            "original_index": idx,
            "removed_because_disagreement_in": [
                c for c, _, _ in CONFIGURATIONS if idx in by_config[c]],
            "disagreement_count": sum(idx in by_config[c]
                                      for c, _, _ in CONFIGURATIONS),
            "source_id": d.get("source_id", ""),
            "source_origin": d.get("source_origin", ""),
            "theorem_name": d.get("theorem_name", ""),
            "domain": d.get("domain", ""),
            "source_problem": d.get("source_problem", ""),
            "rigorous": d.get("rigorous", ""),
            "nonrigorous": d.get("nonrigorous", ""),
            "reference_formal": d.get("formal_flat", ""),
            "reference_formal_verbatim": d.get("formal_statement", ""),
            "configurations": {},
        }
        # Record all four configurations, not just the ones that split -- the
        # agreeing ones are the context you need to judge the split by hand.
        for c, _, _ in CONFIGURATIONS:
            e = ev_by_index[c].get(idx, {})
            rec["configurations"][c] = {
                "disagreed": idx in by_config[c],
                "status": e.get("status", ""),
                "reason": e.get("reason", ""),
                "candidate_formalization": e.get("candidate", ""),
                "judges": e.get("verdicts", {}),
            }
        detail.append(rec)

    out.mkdir(parents=True, exist_ok=True)
    with (out / "removed_problems.jsonl").open("w", encoding="utf-8") as fh:
        for rec in detail:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Human-readable companion.
    lines = [
        "Problems removed for judge disagreement",
        "=" * 72,
        f"generated : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"removed   : {len(removed)} of {n} problems",
        f"remaining : {len(keep)}",
        "",
        "A problem is here because at least one of the four configurations "
        "ended",
        "with the two judges on opposite sides. Indices below are the ORIGINAL "
        "1-based",
        "line numbers, before renumbering.",
        "",
    ]
    for rec in detail:
        lines += [
            "-" * 72,
            f"#{rec['original_index']}  ({rec['disagreement_count']} of 4 "
            f"configurations split: {', '.join(rec['removed_because_disagreement_in'])})",
            f"  origin    : {rec['source_origin']}  {rec['theorem_name']}",
            f"  rigorous  : {rec['rigorous'][:300]}",
            f"  nonrigor. : {rec['nonrigorous'][:300]}",
            f"  reference : {rec['reference_formal'][:300]}",
        ]
        for c, cfg in rec["configurations"].items():
            flag = "SPLIT" if cfg["disagreed"] else "     "
            lines.append(f"  [{flag}] {c}: {cfg['status']}")
            lines.append(f"          candidate: {cfg['candidate_formalization'][:220]}")
            for jname, v in (cfg["judges"] or {}).items():
                if "error" in v:
                    lines.append(f"          {jname}: ERROR {v['error']}")
                    continue
                lines.append(
                    f"          {jname}: "
                    f"{'equivalent' if v.get('equivalent') else 'NOT equivalent'}"
                    f" (confidence {v.get('confidence')})")
                for diff in (v.get("differences") or [])[:4]:
                    lines.append(f"              - {diff}")
        lines.append("")
    (out / "removed_problems.txt").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")

    (out / "summary.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "removed if any of the four configurations ended in a judge "
                "disagreement",
        "problems_before": n,
        "problems_removed": len(removed),
        "problems_after": len(keep),
        "disagreements_per_configuration": {
            c: len(by_config[c]) for c, _, _ in CONFIGURATIONS},
        "removed_indices_original": removed,
        "removed_by_configuration_count": {
            str(k): sum(1 for i in removed
                        if sum(i in by_config[c] for c, _, _ in CONFIGURATIONS) == k)
            for k in (1, 2, 3, 4)},
    }, indent=2) + "\n", encoding="utf-8")

    (out / "index_map.json").write_text(json.dumps({
        "note": "maps the ORIGINAL 1-based line number to the new one; "
                "removed problems map to null",
        "map": {str(i): (keep.index(i) + 1 if i in set(keep) else None)
                for i in range(1, n + 1)},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(removed)} removed problems to {out}")

    # -- filter and renumber -------------------------------------------------
    keep0 = [i - 1 for i in keep]

    for f, vals in aligned.items():
        write_lines(wd / "dataset" / f, [vals[i] for i in keep0])
    for (model, variant), vals in formal_txt.items():
        write_lines(wd / "formalized" / model / f"formalized_{variant}.txt",
                    [vals[i] for i in keep0])

    def dump(path: Path, recs: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(wd / "dataset" / "dataset.jsonl", renumber(dataset, keep))
    for (model, variant), recs in formal_jsonl.items():
        dump(wd / "formalized" / model / f"formalized_{variant}.jsonl",
             renumber(recs, keep))
    for c, _, _ in CONFIGURATIONS:
        dump(wd / "evaluation" / f"{c}.jsonl", renumber(evaluation[c], keep))

    # records.jsonl is the stage-1 construction log, keyed by source_id rather
    # than by line number. Match it up through dataset.jsonl and drop the
    # removed problems' entries, keeping every rejected-source record intact.
    records_path = wd / "dataset" / "records.jsonl"
    if records_path.exists():
        removed_ids = {ds_by_index[i].get("source_id") for i in removed
                       if i in ds_by_index}
        records = load_jsonl(records_path)
        kept_records = [r for r in records
                        if not (r.get("accepted")
                                and r.get("source_id") in removed_ids)]
        dump(records_path, kept_records)
        dump(out / "removed_records.jsonl",
             [r for r in records
              if r.get("accepted") and r.get("source_id") in removed_ids])
        print(f"records.jsonl: {len(records)} -> {len(kept_records)} entries")

    # summary.json describes what stage 1 did, which pruning does not change.
    # Add the pruning outcome alongside rather than rewriting history.
    summary_path = wd / "dataset" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["pruning"] = {
            "rule": "removed if any of the four configurations ended in a "
                    "judge disagreement",
            "problems_before_pruning": n,
            "problems_removed": len(removed),
            "problems_after_pruning": len(keep),
            "detail": "removed_problems/",
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n",
                                encoding="utf-8")

    # Stale artefacts that still refer to the old numbering.
    for name in ("report.txt", "report.json", "report_old_rule.txt",
                 "report_old_rule.json"):
        src = wd / "evaluation" / name
        if src.exists():
            src.rename(out / f"prepruning_{name}")
    for bak in (wd / "evaluation").glob("*.jsonl.bak"):
        bak.rename(out / f"prepruning_{bak.name}")

    print(f"\nfiltered every file to {len(keep)} problems, renumbered 1..{len(keep)}")
    print("stale reports moved into removed_problems/ (they used the old "
          "numbering)")
    print("\nnext: python check_autoformalizations.py --report-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
