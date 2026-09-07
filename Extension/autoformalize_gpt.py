#!/usr/bin/env python3
"""
autoformalize_gpt.py -- stage 2 of the Lean Paraphrases extension.

Autoformalizes every paraphrase produced by generate_paraphrase_dataset.py into
a Lean 4 statement with `sorry` as the proof, using OpenAI's GPT 5.6 Sol.

This script and its sibling (autoformalize_claude.py) are identical apart from
the MODEL CONFIGURATION block below. Running both gives the four files the
paper's comparison needs:

    formalized/gpt/formalized_rigorous.txt
    formalized/gpt/formalized_nonrigorous.txt
    formalized/claude/formalized_rigorous.txt
    formalized/claude/formalized_nonrigorous.txt

Input (from --dataset-dir, default ./dataset)
---------------------------------------------
    rigorousparaphrases.txt     one rigorous paraphrase per line
    nonrigorousparaphrases.txt  one non-rigorous paraphrase per line

Output (in --outdir, default ./formalized/gpt)
----------------------------------------------------------
    formalized_rigorous.txt      one Lean 4 statement per line
    formalized_nonrigorous.txt   one Lean 4 statement per line
    formalized_rigorous.jsonl    verbatim model output plus metadata
    formalized_nonrigorous.jsonl verbatim model output plus metadata

Line N of every output file corresponds to line N of every input file, so the
reference statements in dataset/formalstatements.txt line up too. When the
model fails on a problem the .txt line is left EMPTY rather than shifted --
alignment is the contract the later stages depend on, and the .jsonl records
what went wrong.

Each paraphrase is formalized independently, in a fresh single-turn request
with no other context, matching the paper's one-shot setup: no retries on
content, no self-correction, no pass@k.

Usage
-----
    python autoformalize_gpt.py                        # formalize the whole dataset
    python autoformalize_gpt.py --limit 10             # pilot on the first 10 problems
    python autoformalize_gpt.py --resume               # continue an interrupted run
    python autoformalize_gpt.py --dry-run --outdir /tmp/x
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lp_common import (
    ANTHROPIC_MODEL, OPENAI_MODEL, Clients, JsonlWriter, flatten_lean,
    lean_preamble, load_jsonl, read_lines, require_keys, split_lean,
    write_lines,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# MODEL CONFIGURATION
# This block is the only difference between autoformalize_claude.py and
# autoformalize_gpt.py. Everything below it is identical in both files.
# ---------------------------------------------------------------------------
MODEL_LABEL = "gpt"
MODEL_NAME = OPENAI_MODEL             # "gpt-5.6-sol"
MODEL_DISPLAY = "OpenAI's GPT 5.6 Sol"
REQUIRED_ENV = "OPENAI_API_KEY"


def call_model(clients: Clients, prompt: str) -> str:
    return clients.openai_text(prompt, model=MODEL_NAME)
# ---------------------------------------------------------------------------

# The prompt from Section 4 of the paper, used verbatim.
PAPER_PROMPT = (
    "Formalize this natural language problem into Lean 4 language. Only create "
    "the prompt, and use 'sorry' as the proof. Your response should only "
    "include the final formalized Lean 4 statement."
)

# The line-aligned .txt format needs one collapsible declaration per problem.
# This is appended to the paper's prompt unless --paper-prompt-only is passed.
FORMAT_CONSTRAINT = (
    "Output a single self-contained `theorem` declaration. Do not define "
    "auxiliary functions or constants before it; inline any definition the "
    "statement needs as a hypothesis instead. You may include `import Mathlib` "
    "and `open` lines. Do not include any explanation."
)


def build_prompt(paraphrase: str, paper_prompt_only: bool) -> str:
    head = PAPER_PROMPT if paper_prompt_only else f"{PAPER_PROMPT}\n\n{FORMAT_CONSTRAINT}"
    return f"{head}\n{paraphrase}"


def formalize_one(clients: Clients, index: int, paraphrase: str,
                  paper_prompt_only: bool, dry_run: bool) -> dict:
    """Formalize a single paraphrase. Never raises -- a failure is recorded as
    a record with an empty `statement`, which becomes an empty output line."""
    record = {"index": index, "model": MODEL_NAME, "paraphrase": paraphrase,
              "raw": "", "statement": "", "preamble": "", "status": "",
              "error": ""}

    if not paraphrase.strip():
        record["status"] = "empty_input"
        return record

    if dry_run:
        record["raw"] = (f"import Mathlib\n\ntheorem dry_{index} (n : ℕ) : "
                         f"n + 0 = n := by sorry")
    else:
        try:
            record["raw"] = call_model(clients, build_prompt(
                paraphrase, paper_prompt_only))
        except Exception as exc:  # noqa: BLE001
            record["status"] = "api_error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            return record

    statement = flatten_lean(record["raw"])
    if not statement:
        # The model answered but produced no theorem/lemma we could isolate.
        record["status"] = "unparseable"
        return record

    record["statement"] = statement
    record["preamble"] = lean_preamble(record["raw"])
    record["has_extra_decls"] = len(split_lean(record["raw"])[1].split(
        "theorem ")) > 2
    record["status"] = "ok"
    return record


def run_variant(clients: Clients, variant: str, paraphrases: list[str],
                outdir: Path, args) -> dict:
    """Formalize every paraphrase of one variant ('rigorous' or 'nonrigorous')."""
    jsonl_path = outdir / f"formalized_{variant}.jsonl"
    txt_path = outdir / f"formalized_{variant}.txt"

    done: dict[int, dict] = {}
    if args.resume:
        for rec in load_jsonl(jsonl_path):
            idx = rec.get("index", 0)
            # Only completed work counts; a previous failure is worth retrying.
            if rec.get("status") != "ok" or not 1 <= idx <= len(paraphrases):
                continue
            # And only if it was produced from the paraphrase now on that line.
            # Re-running stage 1 renumbers the dataset, and silently reusing a
            # verdict from the old line N would corrupt the alignment.
            if rec.get("paraphrase") != paraphrases[idx - 1]:
                continue
            done[idx] = rec
        print(f"  {variant}: resuming with {len(done)} already formalized")
    elif jsonl_path.exists():
        print(f"error: {jsonl_path} already exists. Pass --resume to continue "
              f"that run, or choose a different --outdir.", file=sys.stderr)
        raise SystemExit(2)

    todo = [i for i in range(1, len(paraphrases) + 1) if i not in done]
    writer = JsonlWriter(jsonl_path, append=args.resume)

    try:
        from tqdm import tqdm
        bar = tqdm(total=len(todo), unit="problem", desc=f"{MODEL_LABEL}/{variant}")
    except ImportError:
        bar = None

    def work(i: int) -> dict:
        rec = formalize_one(clients, i, paraphrases[i - 1],
                            args.paper_prompt_only, args.dry_run)
        writer.write(rec)
        if bar:
            bar.update(1)
        return rec

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rec in pool.map(work, todo):
                done[rec["index"]] = rec
    except KeyboardInterrupt:
        print("\ninterrupted -- writing what finished so far", file=sys.stderr)
    finally:
        if bar:
            bar.close()
        writer.close()

    # Line N of the .txt is problem N, always. A problem the model failed on
    # gets an empty line rather than being skipped.
    lines = [done.get(i, {}).get("statement", "")
             for i in range(1, len(paraphrases) + 1)]
    write_lines(txt_path, lines)

    stats: dict[str, int] = {}
    for i in range(1, len(paraphrases) + 1):
        key = done.get(i, {}).get("status", "not_attempted")
        stats[key] = stats.get(key, 0) + 1
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=f"Autoformalize the paraphrase dataset into Lean 4 with "
                    f"{MODEL_NAME}.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-dir", type=Path, default=SCRIPT_DIR / "dataset",
                   help="directory holding the paraphrase .txt files")
    p.add_argument("--outdir", type=Path,
                   default=SCRIPT_DIR / "formalized" / MODEL_LABEL,
                   help="where to write the formalized files")
    p.add_argument("--limit", type=int, default=0,
                   help="only formalize the first N problems (0 = all)")
    p.add_argument("--workers", type=int, default=6,
                   help="problems formalized concurrently")
    p.add_argument("--retries", type=int, default=5,
                   help="API retry attempts per call")
    p.add_argument("--effort", default="medium",
                   choices=("low", "medium", "high", "xhigh", "max"),
                   help="reasoning effort (Claude only; ignored by the GPT script)")
    p.add_argument("--paper-prompt-only", action="store_true",
                   help="use the paper's prompt verbatim, without the "
                        "single-theorem formatting constraint. Faithful to the "
                        "original methodology, but multi-declaration answers "
                        "cannot be collapsed onto one line and will be recorded "
                        "as unparseable.")
    p.add_argument("--resume", action="store_true",
                   help="continue a previous run, keeping successful results")
    p.add_argument("--dry-run", action="store_true",
                   help="exercise the pipeline with a stub model and no API calls")
    p.add_argument("--yes", action="store_true",
                   help="skip the pre-run cost confirmation")
    args = p.parse_args(argv)

    if not args.dry_run:
        missing = require_keys(REQUIRED_ENV)
        if missing:
            print(f"error: missing environment variable(s): {', '.join(missing)}",
                  file=sys.stderr)
            return 2

    rigorous = read_lines(args.dataset_dir / "rigorousparaphrases.txt")
    nonrigorous = read_lines(args.dataset_dir / "nonrigorousparaphrases.txt")
    if len(rigorous) != len(nonrigorous):
        print(f"error: input files are not line-aligned -- "
              f"rigorous has {len(rigorous)} lines, nonrigorous has "
              f"{len(nonrigorous)}.", file=sys.stderr)
        return 2
    if args.limit:
        rigorous, nonrigorous = rigorous[:args.limit], nonrigorous[:args.limit]

    n = len(rigorous)
    print(f"model    : {MODEL_NAME}")
    print(f"input    : {args.dataset_dir} ({n} problems)")
    print(f"output   : {args.outdir}")

    if not args.dry_run and not args.yes:
        print(f"\nAbout to make {2 * n} API calls ({n} rigorous + {n} "
              f"non-rigorous). This costs real money.")
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    clients = Clients(retries=args.retries, effort=args.effort, max_tokens=8000)

    start = time.time()
    print()
    stats = {v: run_variant(clients, v, src, args.outdir, args)
             for v, src in (("rigorous", rigorous),
                            ("nonrigorous", nonrigorous))}

    print()
    for variant, s in stats.items():
        ok = s.get("ok", 0)
        print(f"{variant:<12}: {ok}/{n} formalized  {s}")
    print(f"elapsed     : {time.time() - start:.0f}s")
    print(f"output      : {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
