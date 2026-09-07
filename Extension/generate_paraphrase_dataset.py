#!/usr/bin/env python3
"""
generate_paraphrase_dataset.py -- stage 1 of the Lean Paraphrases extension.

The original study ("Lean Paraphrases: Analyzing the Effect of Wording on
Autoformalization Output") used 50 hand-written problems, each with two
paraphrases:

    * a RIGOROUS paraphrase  -- symbol-heavy, mathematically explicit
    * a NON-RIGOROUS one     -- word-heavy, as few symbols as possible

Its stated limitation is dataset size: manual construction and manual checking
capped it at 50. This script lifts that cap to 1000.

Pipeline
--------
    1. SOURCE     Draw problems from SphereLab/FormalMATH-All. Every row pairs
                  a natural-language statement with a Lean 4 formalization that
                  is known to compile, so each problem arrives with a reference
                  formalization already attached -- the same setup the paper
                  used manually.

    2. GENERATE   Claude Opus 5 writes both paraphrases in one structured call,
                  primed with the house style plus few-shot exemplars taken
                  from the paper's own 50-problem manual dataset.

    3. VERIFY     Two independent judges -- Claude Opus 5 and GPT 5.6 Sol --
                  each score four claims with a 0-1 confidence:
                     a. rigorous paraphrase == source problem
                     b. non-rigorous paraphrase == source problem
                     c. the two paraphrases == each other
                     d. the rigor contrast is real (symbolic vs. worded)
                  A pair is accepted only when EVERY claim from BOTH judges is
                  true AND every confidence clears --threshold.

    4. REPAIR     Rejected pairs go back to the generator with the judges'
                  objections, up to --max-attempts times, then are dropped.

Output (in --outdir, default ./dataset)
---------------------------------------
    rigorousparaphrases.txt     one rigorous paraphrase per line
    nonrigorousparaphrases.txt  one non-rigorous paraphrase per line
    formalstatements.txt        one reference Lean 4 statement per line
    dataset.jsonl               all three plus provenance, verbatim
    records.jsonl               every attempt and every verdict
    summary.json                acceptance statistics

The three .txt files are line-aligned: line N of each is the same problem.
Downstream stages rely on that, so nothing is ever written to one without the
other two.

Install
-------
    pip install -r requirements.txt

Credentials
-----------
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...

Usage
-----
    # 1. Calibrate. Scores the paper's 50 hand-written pairs with the same two
    #    verifiers and prints what each threshold would have accepted. Those
    #    pairs are the closest thing to ground truth, so this is how you pick
    #    --threshold without guessing.
    python generate_paraphrase_dataset.py --calibrate

    # 2. Pilot before committing to the full run
    python generate_paraphrase_dataset.py --target 10 --outdir pilot

    # 3. Full run
    python generate_paraphrase_dataset.py --target 1000 --workers 8

    # Resume an interrupted run (re-reads records.jsonl, skips finished work)
    python generate_paraphrase_dataset.py --target 1000 --resume

    # Exercise the pipeline with no API calls and no credentials
    python generate_paraphrase_dataset.py --target 5 --dry-run --outdir /tmp/dry

Notes on the quality gate
-------------------------
The threshold is a precision knob, not a yield knob: a bad pair silently
corrupts the experiment, while a rejected one costs only API calls. That said,
FormalMATH-All has 5560 rows, so the usable pool is finite -- if a 1000-problem
run runs out of source problems, lower --threshold (after checking --calibrate)
or raise --max-len rather than assuming the dataset is at fault.

Cost is roughly 3 API calls per accepted problem (1 generation + 2
verifications), plus 3 more per repair attempt.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Iterator

from lp_common import (
    ANTHROPIC_MODEL, OPENAI_MODEL, Clients, JsonlWriter, confidence,
    flatten_lean, has_helper_decls, lean_preamble, load_jsonl, one_line,
    read_lines, require_keys, sha, write_lines,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# Claude writes the paraphrases; Claude and GPT both check them.
GENERATOR_MODEL = ANTHROPIC_MODEL
JUDGE_ANTHROPIC_MODEL = ANTHROPIC_MODEL
JUDGE_OPENAI_MODEL = OPENAI_MODEL

DEFAULT_DATASET = "SphereLab/FormalMATH-All"

# --------------------------------------------------------------------------
# House style (distilled from Section 3 of the paper)
# --------------------------------------------------------------------------

STYLE_GUIDE = """\
You are building a research dataset of paraphrase PAIRS for a study on how the \
wording of a natural-language mathematics problem affects a language model's \
Lean 4 autoformalization of it.

For one source problem you write exactly two English statements of THE SAME \
mathematical content. They must be logically identical: same hypotheses, same \
quantifiers, same variable domains, same conclusion, same asserted answer. The \
ONLY thing allowed to differ is how the mathematics is worded.

PARAPHRASE 1 -- RIGOROUS
  * Symbol-heavy and mathematically explicit. Use LaTeX inline math ($...$).
  * State domains and hypotheses symbolically: "Let $a,b,c$ be positive real
    numbers", "for all $x,y \\in \\mathbb{R}$", "$p$ prime", "$n \\in \\mathbb{N}$".
  * Write the relation as a formula, not a description:
    "$\\sum_{k=1}^{n} \\frac{1}{k}$", "$p^2 \\mid \\binom{2p}{p}-2$",
    "$x \\equiv 0 \\pmod 2$".

PARAPHRASE 2 -- NON-RIGOROUS
  * Word-heavy. Goal: as FEW symbols and numerals as you can manage while
    staying unambiguous. Ideally the only math notation left is the bare
    variable names, or none at all.
  * Replace formulas with standard mathematical English: "the cyclic sum of",
    "the central binomial coefficient", "the complex conjugate", "the n-th
    harmonic number", "has a remainder of 1 when divided by", "is even",
    "the sum of the squares of", "the product of three consecutive integers".
  * Prefer named notions over their expansions ("the complement of an angle",
    "a geometric sequence", "a perfect square", "a lattice point").
  * Do NOT drop a hypothesis just because it is awkward to word. If the
    rigorous version says the numbers are positive reals, the worded version
    must still say they are positive.

BOTH PARAPHRASES
  * One single line each. No line breaks, no bullet points, no numbering.
  * A statement to be proved, not a question to be explored.
  * If the source problem has a definite answer, append the answer as an
    assertion in the paper's house format, e.g. "Prove that the answer is
    $630$." or "Show that the answer is $\\frac{24}{25}$." Both paraphrases
    must assert the SAME answer.
  * If the source is already a proof problem, no answer clause is needed.
  * No solution, no proof, no hints, no commentary -- statement only.
  * Self-contained: no reference to figures, diagrams, tables, answer choices,
    or "the previous part".
  * Plain English prose. Never emit Lean syntax, Mathlib lemma names, or
    identifiers like `Finset.Icc` or `Nat.Prime` -- these are natural-language
    statements that a model will later be asked to formalize from scratch.

REJECT the source problem (set "usable": false) if it depends on a figure, is
multiple-choice, is not self-contained, is ill-posed, or cannot be worded
without symbols in any reasonable way.\
"""

FORMAL_HINT_NOTE = """\
A reference Lean 4 formalization of the source problem is provided below. Use \
it ONLY to resolve ambiguity in the natural-language statement -- which set a \
variable ranges over, how a quantifier is scoped, whether an inequality is \
strict. Both paraphrases must agree with the domains and quantifiers it uses. \
Do not copy its syntax, its identifiers, or its variable names into your \
paraphrases.\
"""

# Fallback exemplars, used when the manual dataset files are not next to the
# script. Taken verbatim from the paper's 50-problem set.
FALLBACK_EXEMPLARS: list[tuple[str, str]] = [
    (
        "For $a, b, c, d>0, abcd=1$ prove that "
        "$\\frac{1}{1+(1+a)^2}+\\frac{1}{1+(1+b)^2}+\\frac{1}{1+(1+c)^2}"
        "+\\frac{1}{1+(1+d)^2}\\le\\frac{4}{5}$",
        "For positive numbers $a, b, c, d$ whose product is 1, prove that the "
        "cyclic sum of $\\frac{1}{1+(1+a)^2}$ is at most $\\frac{4}{5}$.",
    ),
    (
        "Find the sum of all positive integers $N=10a+b$, where $1\\le a\\le 9$ "
        "and $0\\le b\\le 9$, and where $a\\mid N$ and $b\\mid N$. Show that the "
        "answer is $630$.",
        "Find the sum of all positive two-digit integers that are divisible by "
        "each of their digits. Show that the answer is $630$.",
    ),
    (
        "For every p prime number show that $ p^2 \\mid \\binom{2p}{p}-2 $",
        "For any prime number $p$, show that $p^2$ always divides the number two "
        "less than the $p$-th central binomial coefficient.",
    ),
]

# Indices (1-based) into the manual dataset used as few-shot exemplars. Fixed
# rather than random so the prompt prefix stays byte-stable and cacheable.
EXEMPLAR_INDICES = [5, 6, 20, 25, 36, 46]

# --------------------------------------------------------------------------
# JSON schemas for structured output
# --------------------------------------------------------------------------
# The free-text analysis field comes first in every schema so the model reasons
# before it commits to booleans and confidences.

GENERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string",
            "description": "Brief plan: the hypotheses, the conclusion, the "
                           "answer if any, and which worded devices will carry "
                           "the non-rigorous version.",
        },
        "usable": {
            "type": "boolean",
            "description": "False if the source problem cannot be turned into a "
                           "clean self-contained paraphrase pair.",
        },
        "reject_reason": {
            "type": "string",
            "description": "Why it is unusable; empty string if usable.",
        },
        "answer": {
            "type": "string",
            "description": "The definite answer in LaTeX if the problem has one, "
                           "else empty string.",
        },
        "rigorous": {
            "type": "string",
            "description": "The rigorous paraphrase, one line. Empty if unusable.",
        },
        "nonrigorous": {
            "type": "string",
            "description": "The non-rigorous paraphrase, one line. Empty if unusable.",
        },
        "worded_devices": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Worded devices used in the non-rigorous version, e.g. "
                           "['cyclic sum', 'perfect square'].",
        },
    },
    "required": [
        "analysis", "usable", "reject_reason", "answer",
        "rigorous", "nonrigorous", "worded_devices",
    ],
    "additionalProperties": False,
}

_CONF = "Probability from 0.0 to 1.0 that the corresponding claim is true."

VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string",
            "description": "Compare hypotheses, variable domains, quantifiers, "
                           "the conclusion and the asserted answer across all "
                           "three statements. Name any discrepancy explicitly.",
        },
        "discrepancies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One entry per difference in MATHEMATICAL CONTENT "
                           "that makes one of the four claims below false. Do "
                           "not list stylistic observations, wording "
                           "preferences, or things you decided were fine. "
                           "Empty if there are none.",
        },
        # No "minimum"/"maximum" keywords: neither provider's strict
        # JSON-schema mode accepts numeric bounds. Values are clamped in Python.
        "rigorous_equivalent": {"type": "boolean"},
        "rigorous_confidence": {"type": "number", "description": _CONF},
        "nonrigorous_equivalent": {"type": "boolean"},
        "nonrigorous_confidence": {"type": "number", "description": _CONF},
        "paraphrases_mutually_equivalent": {"type": "boolean"},
        "mutual_confidence": {"type": "number", "description": _CONF},
        "rigor_contrast_ok": {"type": "boolean"},
        "rigor_contrast_confidence": {"type": "number", "description": _CONF},
    },
    "required": [
        "analysis", "discrepancies",
        "rigorous_equivalent", "rigorous_confidence",
        "nonrigorous_equivalent", "nonrigorous_confidence",
        "paraphrases_mutually_equivalent", "mutual_confidence",
        "rigor_contrast_ok", "rigor_contrast_confidence",
    ],
    "additionalProperties": False,
}

CLAIMS = [
    ("rigorous_equivalent", "rigorous_confidence"),
    ("nonrigorous_equivalent", "nonrigorous_confidence"),
    ("paraphrases_mutually_equivalent", "mutual_confidence"),
    ("rigor_contrast_ok", "rigor_contrast_confidence"),
]

VERIFIER_SYSTEM = """\
You are an adversarial checker for a mathematics paraphrase dataset. You are \
shown an ORIGINAL problem and two candidate paraphrases of it: one written to \
be rigorous and symbolic, one written to be wordy and non-symbolic.

Your job is to find reasons to REJECT the pair. Be strict and literal. Do not \
read a statement charitably, do not silently supply a missing hypothesis, and \
do not assume the writer meant the obvious thing.

Judge four claims independently:

  1. rigorous_equivalent -- the rigorous paraphrase asserts exactly the same
     mathematical content as the original problem.
  2. nonrigorous_equivalent -- likewise for the non-rigorous paraphrase.
  3. paraphrases_mutually_equivalent -- the two paraphrases assert exactly the
     same content as each other.
  4. rigor_contrast_ok -- paraphrase A really is the more symbolic of the two
     and paraphrase B really does carry the mathematics in words.

Treat as NON-equivalent, every time:
  * a changed, added or dropped hypothesis, including variable domain --
    "real" vs "positive real" vs "integer" vs "natural number" are all
    different, and an unqualified "number" is not the same as "positive real";
  * a changed quantifier or a changed order of quantifiers;
  * a changed, dropped or newly invented final answer;
  * strict vs non-strict inequality; "at most" vs "less than";
  * a different indexing range, a different modulus, a different exponent;
  * "divides" vs "is divisible by" stated backwards;
  * a statement that is merely implied by the original rather than equal to it.

What rigor_contrast_ok means, precisely:
  * Paraphrase A should state domains and relations symbolically.
  * Paraphrase B should carry the STRUCTURE of the statement in words --
    aggregation, quantifiers, hypotheses and the comparison should be worded:
    "the cyclic sum of", "for every prime", "is at most", "has a remainder of
    1 when divided by", "the sum of the squares of".
  * Paraphrase B MAY keep a core LaTeX expression that has no reasonable
    worded form. Retaining one summand, one fraction or the bare variable
    names is expected and fine. What is not fine is reprinting the entire
    statement in symbols.
  * Judge this claim FALSE only when B is not meaningfully less symbolic than
    A -- the two are near-identical, or B simply repeats A's full formula with
    a few connecting words.

DO NOT reject a pair for any of the following. None of them is a defect:
  * Paraphrase A closely resembling the original problem. Source problems are
    often already rigorous, and A is under no obligation to differ from the
    original. You are judging equivalence and rigor contrast, not novelty.
  * Paraphrase B naming a standard mathematical object instead of expanding it
    -- "cyclic sum", "the central binomial coefficient", "the n-th harmonic
    number", "the complex conjugate", "a perfect square", "a lattice point",
    "an arithmetic progression", "the complement of an angle". This is precise
    mathematical English and counts as equivalent whenever the naming is
    unambiguous in context, even though the expansion is left implicit. Leaving
    an expansion implicit in this way is the entire point of paraphrase B.
  * Different variable names, different word order, or a different but
    equivalent phrasing of the same relation.
  * The paraphrases containing no solution, proof or method. They are problem
    statements only.

Put an entry in "discrepancies" only when it is a difference in mathematical \
content that makes one of the four claims false. It is a list of defects, not a \
list of observations: if you looked at something and concluded it was \
acceptable, it does not belong there.

Each confidence is your probability that the corresponding claim is true, on a \
0-1 scale. Use the full range and calibrate honestly: a claim you have checked \
term by term and found to hold deserves a high confidence, a claim you have \
found to fail deserves a low one, and only genuine uncertainty belongs in the \
middle. Do not shade a confidence downward merely to seem cautious.\
"""


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def norm_for_dedupe(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())[:400]


def symbol_density(text: str) -> float:
    """Fraction of characters that are math notation or digits. A cheap sanity
    check on the rigor contrast; the judges do the real work."""
    if not text:
        return 0.0
    hits = sum(1 for ch in text if ch.isdigit() or ch in "$\\^_{}=<>+*/|")
    return hits / len(text)


# --------------------------------------------------------------------------
# Few-shot exemplars from the paper's manual dataset
# --------------------------------------------------------------------------

def find_manual_dir(explicit: Path | None = None) -> Path | None:
    """Locate the paper's hand-built 50-problem dataset.

    Looked for next to this script first, then one level up -- the manual files
    live alongside the paper, which may or may not be the same directory the
    pipeline runs from.
    """
    candidates = [explicit] if explicit else [SCRIPT_DIR, SCRIPT_DIR.parent]
    for d in candidates:
        if d and (d / "rigorousparaphrases.txt").exists() \
                and (d / "nonrigorousparaphrases.txt").exists():
            return d
    return None


def read_manual_pairs(directory: Path | None) -> list[tuple[str, str]]:
    """Read the hand-built 50-problem dataset from `directory`."""
    if directory is None:
        return []
    rig_path = directory / "rigorousparaphrases.txt"
    non_path = directory / "nonrigorousparaphrases.txt"
    if not (rig_path.exists() and non_path.exists()):
        return []
    # The files are line-aligned: line N of each is the same problem.
    rig = [x for x in read_lines(rig_path) if x]
    non = [x for x in read_lines(non_path) if x]
    return list(zip(rig, non))


def build_exemplar_block(pairs: list[tuple[str, str]]) -> str:
    chosen = [pairs[i - 1] for i in EXEMPLAR_INDICES
              if 0 < i <= len(pairs)] if pairs else []
    if not chosen:
        chosen = FALLBACK_EXEMPLARS
    sections = [
        f"Example {n}\n"
        f"  Rigorous:     {one_line(rig)}\n"
        f"  Non-rigorous: {one_line(non)}"
        for n, (rig, non) in enumerate(chosen, 1)
    ]
    return (
        "Reference pairs from the hand-built portion of this dataset. Match "
        "their register, their level of detail, and their answer-clause "
        "formatting.\n\n" + "\n\n".join(sections)
    )


# --------------------------------------------------------------------------
# Source dataset
# --------------------------------------------------------------------------

FIGURE_RE = re.compile(
    r"\b(figure|diagram|as shown|shown below|the picture|the graph below|"
    r"table below|image|attached)\b", re.I,
)
CHOICE_RE = re.compile(
    r"(\\textbf\{\(?[A-E]\)?\}|\(\s*[A-E]\s*\)\s*\S+.*\(\s*[B-E]\s*\))", re.S,
)
OPTION_LABEL_RE = re.compile(r"(?:^|[\s$])([A-E])\s*[:.]\s", re.M)
PART_RE = re.compile(r"\b(part \(?[ab]\)?|previous part|see above)\b", re.I)
# Enumerated sub-questions -- a multi-part problem has no single statement to
# paraphrase, and its reference formalization covers only one of the parts.
SUBQ_PATTERNS = (
    re.compile(r"\(\s*1\s*\).{5,}\(\s*2\s*\)", re.S),
    re.compile(r"(?:^|\s)1[.)]\s.{5,}(?:^|\s)2[.)]\s", re.S),
    re.compile(r"[①-⑩].{2,}[①-⑩]", re.S),
)


def is_candidate(problem: str, formal: str,
                 min_len: int, max_len: int) -> tuple[bool, str]:
    """Cheap local pre-filter. Anything that survives is worth an API call."""
    if not problem or not formal:
        return False, "empty"
    if not (min_len <= len(problem) <= max_len):
        return False, "length"
    if FIGURE_RE.search(problem):
        return False, "figure"
    if (CHOICE_RE.search(problem)
            or len(set(OPTION_LABEL_RE.findall(problem))) >= 3):
        return False, "multiple_choice"
    if PART_RE.search(problem) or any(r.search(problem) for r in SUBQ_PATTERNS):
        return False, "multi_part"
    if sum(ch.isascii() for ch in problem) / len(problem) < 0.95:
        return False, "non_english"
    # A formalization with helper definitions cannot be collapsed onto one line
    # without breaking it, and the .txt files are line-aligned by contract.
    if has_helper_decls(formal):
        return False, "formal_not_single_theorem"
    if not flatten_lean(formal):
        return False, "formal_unparseable"
    return True, ""


def iter_source_problems(dataset: str, split: str, min_len: int, max_len: int,
                         seed: int, require_compiles: bool) -> Iterator[dict]:
    """Load the source dataset, shuffle it and yield filtered problems.

    FormalMATH-All is ~5.5k rows, so it loads whole rather than streaming --
    that gives a real shuffle and an exact pool size instead of a buffer
    approximation.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split).shuffle(seed=seed)

    seen: set[str] = set()
    seq = 0
    for row in ds:
        problem = (row.get("refined_statement") or "").strip()
        formal = (row.get("autoformalization") or "").strip()
        if require_compiles and row.get("compiler_feedback_bool") is False:
            continue
        ok, _ = is_candidate(problem, formal, min_len, max_len)
        if not ok:
            continue
        key = norm_for_dedupe(problem)
        if key in seen:
            continue
        seen.add(key)
        seq += 1
        yield {
            "seq": seq,
            "source_id": sha(problem),
            "source_dataset": dataset,
            "source_origin": row.get("source", ""),
            "theorem_name": row.get("theorem_names", ""),
            "domain": row.get("domain", ""),
            "problem": problem,
            "formal_statement": formal,
            "formal_flat": flatten_lean(formal),
            "formal_preamble": lean_preamble(formal),
        }


def iter_fake_problems(count: int) -> Iterator[dict]:
    """Offline stand-in used by --dry-run."""
    for i in range(1, count + 1):
        problem = (f"Let $a$ and $b$ be positive real numbers with $a+b={i}$. "
                   f"Prove that $ab \\le \\frac{{{i}^2}}{{4}}$.")
        formal = (f"import Mathlib\n\ntheorem dry_{i} (a b : ℝ) (ha : 0 < a) "
                  f"(hb : 0 < b) (h : a + b = {i}) : a * b ≤ {i}^2 / 4 := by")
        yield {
            "seq": i, "source_id": sha(problem), "source_dataset": "dry-run",
            "source_origin": "synthetic", "theorem_name": f"dry_{i}",
            "domain": "", "problem": problem, "formal_statement": formal,
            "formal_flat": flatten_lean(formal),
            "formal_preamble": lean_preamble(formal),
        }


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

def generation_prompt(source: dict, exemplars: str, feedback: list[str],
                      formal_hint: bool) -> str:
    parts = [exemplars, "", "SOURCE PROBLEM", source["problem"]]
    if formal_hint and source.get("formal_statement"):
        parts += ["", FORMAL_HINT_NOTE, "", "REFERENCE LEAN 4 FORMALIZATION",
                  source["formal_statement"]]
    if feedback:
        parts += [
            "",
            "A previous attempt at this pair was REJECTED by the verifiers for "
            "the reasons below. Write a new pair that fixes every one of them. "
            "If the problem simply cannot be paraphrased cleanly, set "
            '"usable": false.',
            *(f"  - {f}" for f in feedback),
        ]
    parts += ["", "Write the two paraphrases now."]
    return "\n".join(parts)


def verification_prompt(source: dict, rigorous: str, nonrigorous: str,
                        formal_hint: bool) -> str:
    parts = ["ORIGINAL PROBLEM", source["problem"]]
    if formal_hint and source.get("formal_statement"):
        parts += [
            "",
            "REFERENCE LEAN 4 FORMALIZATION OF THE ORIGINAL (authoritative for "
            "variable domains and quantifier structure; both paraphrases must "
            "agree with it)",
            source["formal_statement"],
        ]
    parts += [
        "",
        "CANDIDATE PARAPHRASE A (intended to be rigorous / symbolic)",
        rigorous,
        "",
        "CANDIDATE PARAPHRASE B (intended to be non-rigorous / worded)",
        nonrigorous,
        "",
        "Evaluate the four claims.",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Dry-run stubs
# --------------------------------------------------------------------------

def fake_generation(source: dict) -> dict:
    return {
        "analysis": "dry run", "usable": True, "reject_reason": "", "answer": "",
        "rigorous": one_line(source["problem"]),
        "nonrigorous": "Prove that for two positive numbers with a fixed sum, "
                       "their product is at most the square of half that sum.",
        "worded_devices": ["fixed sum"],
    }


def fake_verification() -> dict:
    return {
        "analysis": "dry run", "discrepancies": [],
        "rigorous_equivalent": True, "rigorous_confidence": 0.97,
        "nonrigorous_equivalent": True, "nonrigorous_confidence": 0.95,
        "paraphrases_mutually_equivalent": True, "mutual_confidence": 0.96,
        "rigor_contrast_ok": True, "rigor_contrast_confidence": 0.93,
    }


# --------------------------------------------------------------------------
# Generation / verification / acceptance
# --------------------------------------------------------------------------

def generate_pair(clients: Clients, source: dict, exemplars: str,
                  feedback: list[str], formal_hint: bool, dry: bool) -> dict:
    if dry:
        return fake_generation(source)
    return clients.anthropic_json(
        STYLE_GUIDE,
        generation_prompt(source, exemplars, feedback, formal_hint),
        GENERATION_SCHEMA, "paraphrase_pair", model=GENERATOR_MODEL,
    )


def verify_pair(clients: Clients, source: dict, rigorous: str, nonrigorous: str,
                formal_hint: bool, dry: bool) -> dict[str, dict]:
    """Run both judges. A judge that errors out is recorded, not swallowed --
    a missing verdict means the pair cannot be accepted."""
    if dry:
        return {"claude_opus_5": fake_verification(),
                "gpt_5_6_sol": fake_verification()}

    prompt = verification_prompt(source, rigorous, nonrigorous, formal_hint)
    verdicts: dict[str, dict] = {}

    def run(name: str, fn: Callable[[], dict]) -> None:
        try:
            verdicts[name] = fn()
        except Exception as exc:  # noqa: BLE001
            verdicts[name] = {"error": f"{type(exc).__name__}: {exc}"}

    # Two judges in parallel: the round trip dominates wall-clock here.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for fut in [
            pool.submit(run, "claude_opus_5", lambda: clients.anthropic_json(
                VERIFIER_SYSTEM, prompt, VERIFICATION_SCHEMA, "verdict",
                model=JUDGE_ANTHROPIC_MODEL)),
            pool.submit(run, "gpt_5_6_sol", lambda: clients.openai_json(
                VERIFIER_SYSTEM, prompt, VERIFICATION_SCHEMA, "verdict",
                model=JUDGE_OPENAI_MODEL)),
        ]:
            fut.result()
    return verdicts


def judge_objections(verdict: dict, threshold: float,
                     discrepancies_block: bool) -> list[str]:
    """Reasons this single judge rejects the pair.

    The gate is the judge's four booleans and four confidences. A judge that
    names a concrete discrepancy while still marking every claim true has
    contradicted itself; by default that also blocks, since for dataset
    construction a false accept is more costly than a false reject. Pass
    --allow-flagged-discrepancies to demote those to advisory.
    """
    if "error" in verdict:
        return [f"verifier failed to respond: {verdict['error']}"]
    out: list[str] = []
    for flag, conf_key in CLAIMS:
        ok = bool(verdict.get(flag))
        conf = confidence(verdict, conf_key)
        if not ok:
            out.append(f"{flag} judged FALSE (confidence {conf:.2f})")
        elif conf < threshold:
            out.append(f"{flag} true but confidence {conf:.2f} < {threshold:.2f}")
    if discrepancies_block:
        out += [f"discrepancy: {d}" for d in verdict.get("discrepancies", []) if d]
    return out


def evaluate(verdicts: dict[str, dict], threshold: float, mode: str,
             discrepancies_block: bool = True
             ) -> tuple[bool, list[str], dict[str, float]]:
    """Combine the judges. Returns (accepted, objections, min-confidence-per-judge).

    `objections` doubles as the repair feedback sent back to the generator, so
    it lists discrepancies even when they are not blocking.
    """
    per_judge_ok: dict[str, bool] = {}
    per_judge_min: dict[str, float] = {}
    objections: list[str] = []

    for name, verdict in verdicts.items():
        reasons = judge_objections(verdict, threshold, discrepancies_block)
        per_judge_ok[name] = not reasons
        if not discrepancies_block:
            reasons = reasons + [f"discrepancy (advisory): {d}"
                                 for d in verdict.get("discrepancies", []) if d]
        per_judge_min[name] = (
            0.0 if "error" in verdict
            else min(confidence(verdict, c) for _, c in CLAIMS))
        objections += [f"[{name}] {r}" for r in reasons]

    if mode == "both":
        accepted = len(per_judge_ok) == 2 and all(per_judge_ok.values())
    elif mode == "any":
        accepted = any(per_judge_ok.values())
    else:  # "mean" -- both judges must answer, every boolean must be true,
           # and the mean of their weakest confidences must clear the bar.
        both_answered = len(verdicts) == 2 and all(
            "error" not in v for v in verdicts.values())
        bools_ok = both_answered and all(
            bool(v.get(flag)) for v in verdicts.values() for flag, _ in CLAIMS)
        mean_min = (sum(per_judge_min.values()) / len(per_judge_min)
                    if per_judge_min else 0.0)
        accepted = bools_ok and mean_min >= threshold

    return accepted, objections, per_judge_min


def process_source(clients: Clients, source: dict, exemplars: str, args,
                   discrepancies_block: bool) -> dict:
    """Generate -> verify -> repair loop for one source problem."""
    record: dict[str, Any] = {
        "seq": source["seq"],
        "source_id": source["source_id"],
        "source_dataset": source["source_dataset"],
        "source_origin": source["source_origin"],
        "theorem_name": source["theorem_name"],
        "domain": source["domain"],
        "source_problem": source["problem"],
        "formal_statement": source["formal_statement"],
        "formal_flat": source["formal_flat"],
        "formal_preamble": source["formal_preamble"],
        "accepted": False, "status": "", "attempts": [],
        "rigorous": "", "nonrigorous": "",
    }
    feedback: list[str] = []

    for attempt in range(1, args.max_attempts + 1):
        entry: dict[str, Any] = {"attempt": attempt, "feedback_in": list(feedback)}
        try:
            gen = generate_pair(clients, source, exemplars, feedback,
                                not args.no_formal_hint, args.dry_run)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"generation failed: {type(exc).__name__}: {exc}"
            record["attempts"].append(entry)
            record["status"] = "generation_error"
            return record

        if not gen.get("usable", False):
            entry["usable"] = False
            entry["reject_reason"] = gen.get("reject_reason", "")
            record["attempts"].append(entry)
            record["status"] = "generator_rejected_source"
            return record

        rigorous = one_line(gen.get("rigorous", ""))
        nonrigorous = one_line(gen.get("nonrigorous", ""))
        entry.update({
            "rigorous": rigorous, "nonrigorous": nonrigorous,
            "worded_devices": gen.get("worded_devices", []),
            "generator_answer": gen.get("answer", ""),
            "symbol_density": {
                "rigorous": round(symbol_density(rigorous), 4),
                "nonrigorous": round(symbol_density(nonrigorous), 4),
            },
        })

        if not rigorous or not nonrigorous:
            entry["error"] = "generator returned an empty paraphrase"
            record["attempts"].append(entry)
            record["status"] = "empty_generation"
            return record

        verdicts = verify_pair(clients, source, rigorous, nonrigorous,
                               not args.no_formal_hint, args.dry_run)
        accepted, objections, minima = evaluate(
            verdicts, args.threshold, args.judge_mode, discrepancies_block)
        entry.update({"verdicts": verdicts, "min_confidence": minima,
                      "objections": objections, "accepted": accepted})
        record["attempts"].append(entry)

        if accepted:
            record.update({
                "accepted": True, "status": "accepted",
                "rigorous": rigorous, "nonrigorous": nonrigorous,
                "accepted_on_attempt": attempt, "min_confidence": minima,
            })
            return record

        feedback = objections[:8]  # keep the repair prompt focused

    record["status"] = "rejected_after_max_attempts"
    return record


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_dataset_files(records: list[dict], outdir: Path, target: int) -> int:
    """Write the three line-aligned .txt files plus the verbatim .jsonl.

    Ordering is by source-stream position, not completion order, so the same
    inputs always produce the same file regardless of --workers.
    """
    accepted = sorted(
        (r for r in records if r.get("accepted")),
        key=lambda r: (r.get("seq", 0), r.get("source_id", "")),
    )[:target]

    write_lines(outdir / "rigorousparaphrases.txt",
                (r["rigorous"] for r in accepted))
    write_lines(outdir / "nonrigorousparaphrases.txt",
                (r["nonrigorous"] for r in accepted))
    write_lines(outdir / "formalstatements.txt",
                (r["formal_flat"] for r in accepted))

    with (outdir / "dataset.jsonl").open("w", encoding="utf-8") as fh:
        for i, r in enumerate(accepted, 1):
            fh.write(json.dumps({
                "index": i,
                "source_id": r["source_id"],
                "source_origin": r["source_origin"],
                "theorem_name": r["theorem_name"],
                "domain": r["domain"],
                "source_problem": r["source_problem"],
                "rigorous": r["rigorous"],
                "nonrigorous": r["nonrigorous"],
                "formal_flat": r["formal_flat"],
                "formal_preamble": r["formal_preamble"],
                "formal_statement": r["formal_statement"],
                "min_confidence": r.get("min_confidence", {}),
                "accepted_on_attempt": r.get("accepted_on_attempt"),
            }, ensure_ascii=False) + "\n")
    return len(accepted)


def write_summary(records: list[dict], outdir: Path, args) -> dict:
    statuses: dict[str, int] = {}
    for rec in records:
        key = rec.get("status", "unknown")
        statuses[key] = statuses.get(key, 0) + 1

    accepted = [r for r in records if r.get("accepted")]
    on_attempt: dict[str, int] = {}
    origins: dict[str, int] = {}
    for rec in accepted:
        k = str(rec.get("accepted_on_attempt", "?"))
        on_attempt[k] = on_attempt.get(k, 0) + 1
        o = rec.get("source_origin", "?")
        origins[o] = origins.get(o, 0) + 1

    disagreements = 0
    for rec in records:
        for att in rec.get("attempts", []):
            mins = att.get("min_confidence") or {}
            if len(mins) == 2:
                a, b = list(mins.values())
                if (a >= args.threshold) != (b >= args.threshold):
                    disagreements += 1

    summary = {
        "generator_model": GENERATOR_MODEL,
        "verifier_models": [JUDGE_ANTHROPIC_MODEL, JUDGE_OPENAI_MODEL],
        "confidence_threshold": args.threshold,
        "judge_mode": args.judge_mode,
        "max_attempts": args.max_attempts,
        "formal_hint_shown_to_models": not args.no_formal_hint,
        "source_dataset": args.dataset,
        "source_split": args.split,
        "problems_processed": len(records),
        "problems_accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / len(records), 4) if records else 0.0,
        "status_counts": statuses,
        "accepted_on_attempt": on_attempt,
        "accepted_by_source_origin": origins,
        "judge_threshold_disagreements": disagreements,
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


# --------------------------------------------------------------------------
# Calibration against the paper's hand-written pairs
# --------------------------------------------------------------------------

def run_calibration(clients: Clients, args) -> int:
    """Score the paper's 50 manual pairs with the same verifiers the pipeline
    uses, and report what each threshold would have accepted.

    Those pairs were written and checked by hand for the paper, so they are the
    closest thing to ground truth available. If the verifiers reject a large
    share of them the bar is too high, and the full run will burn API calls
    rediscovering that.

    There is no separate "original" for a manual pair, so the rigorous version
    stands in as the original. That makes claim 1 trivially true; claims 2, 3
    and 4 are the informative ones.
    """
    blocks = not args.allow_flagged_discrepancies
    manual_dir = find_manual_dir(args.manual_dir)
    pairs = read_manual_pairs(manual_dir)
    if not pairs:
        print("error: could not find the hand-written dataset "
              "(rigorousparaphrases.txt + nonrigorousparaphrases.txt). Looked "
              f"in {SCRIPT_DIR} and {SCRIPT_DIR.parent}; pass --manual-dir to "
              "point at it.", file=sys.stderr)
        return 2
    print(f"manual dataset: {manual_dir}")
    if args.calibrate_limit:
        pairs = pairs[:args.calibrate_limit]
    print(f"calibrating on {len(pairs)} hand-written pairs "
          f"({JUDGE_ANTHROPIC_MODEL} + {JUDGE_OPENAI_MODEL})\n")

    def score(item):
        idx, (rigorous, nonrigorous) = item
        source = {"problem": rigorous, "formal_statement": ""}
        verdicts = verify_pair(clients, source, rigorous, nonrigorous,
                               False, args.dry_run)
        _, objections, minima = evaluate(verdicts, args.threshold,
                                         args.judge_mode, blocks)
        return {"index": idx, "rigorous": rigorous, "nonrigorous": nonrigorous,
                "verdicts": verdicts, "min_confidence": minima,
                "objections": objections}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = sorted(pool.map(score, enumerate(pairs, 1)),
                         key=lambda r: r["index"])

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "calibration.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{'threshold':>10}  {'accepted':>8}  {'rate':>7}")
    for t in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        n = sum(1 for r in results
                if evaluate(r["verdicts"], t, args.judge_mode, blocks)[0])
        print(f"{t:>10.2f}  {n:>8}  {n / len(results):>6.1%}")

    print(f"\nrejected at --threshold {args.threshold:.2f}:")
    rejected = [r for r in results
                if not evaluate(r["verdicts"], args.threshold,
                                args.judge_mode, blocks)[0]]
    for r in rejected:
        print(f"  #{r['index']}: {r['nonrigorous'][:90]}")
        for o in r["objections"][:3]:
            print(f"      {o}")
    if not rejected:
        print("  (none)")
    print(f"\nper-pair detail written to {out}")
    return 0


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build a verified rigorous / non-rigorous paraphrase "
                    "dataset with reference Lean 4 statements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", type=int, default=1000,
                   help="number of accepted problems to produce")
    p.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "dataset",
                   help="output directory (the manual dataset is never overwritten)")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="HuggingFace dataset id to draw source problems from")
    p.add_argument("--split", default="train")
    p.add_argument("--threshold", type=float, default=0.85,
                   help="minimum verifier confidence required on every claim")
    p.add_argument("--judge-mode", choices=("both", "any", "mean"), default="both",
                   help="both = every judge must clear the bar (strictest)")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="generation attempts per source problem, including repairs")
    p.add_argument("--workers", type=int, default=6,
                   help="source problems processed concurrently")
    p.add_argument("--retries", type=int, default=5,
                   help="API retry attempts per call")
    p.add_argument("--effort", default="medium",
                   choices=("low", "medium", "high", "xhigh", "max"),
                   help="Claude Opus 5 reasoning effort")
    p.add_argument("--min-len", type=int, default=60,
                   help="minimum source problem length in characters")
    p.add_argument("--max-len", type=int, default=900,
                   help="maximum source problem length in characters")
    p.add_argument("--no-formal-hint", action="store_true",
                   help="hide the reference Lean statement from the generator "
                        "and verifiers (ablation)")
    p.add_argument("--allow-noncompiling", action="store_true",
                   help="keep source rows whose reference formalization does "
                        "not compile")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume", action="store_true",
                   help="continue a previous run using its records.jsonl")
    p.add_argument("--allow-flagged-discrepancies", action="store_true",
                   help="do not reject on a verifier's free-text discrepancy "
                        "list; gate on the booleans and confidences only")
    p.add_argument("--manual-dir", type=Path, default=None,
                   help="directory holding the paper's hand-written "
                        "rigorousparaphrases.txt / nonrigorousparaphrases.txt, "
                        "used for few-shot exemplars and --calibrate "
                        "(default: this directory, then its parent)")
    p.add_argument("--calibrate", action="store_true",
                   help="score the paper's hand-written pairs with the verifiers "
                        "and report acceptance by threshold, then exit")
    p.add_argument("--calibrate-limit", type=int, default=0,
                   help="only calibrate on the first N manual pairs (0 = all)")
    p.add_argument("--dry-run", action="store_true",
                   help="exercise the pipeline with stub models and no API calls")
    p.add_argument("--yes", action="store_true",
                   help="skip the pre-run cost confirmation")
    args = p.parse_args(argv)

    random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    records_path = args.outdir / "records.jsonl"

    if not args.dry_run:
        missing = require_keys("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        if missing:
            print(f"error: missing environment variable(s): {', '.join(missing)}",
                  file=sys.stderr)
            return 2

    clients = Clients(retries=args.retries, effort=args.effort)

    if args.calibrate:
        return run_calibration(clients, args)

    prior_records: list[dict] = []
    prior_ids: set[str] = set()
    if args.resume:
        prior_records = load_jsonl(records_path)
        prior_ids = {r.get("source_id", "") for r in prior_records}
        done = sum(1 for r in prior_records if r.get("accepted"))
        print(f"resuming: {len(prior_records)} records on disk, {done} accepted")
    elif records_path.exists():
        print(f"error: {records_path} already exists. Pass --resume to continue "
              f"that run, or choose a different --outdir.", file=sys.stderr)
        return 2

    remaining = args.target - sum(1 for r in prior_records if r.get("accepted"))
    if remaining <= 0:
        print("target already met; rewriting output files from records.jsonl")
        n = write_dataset_files(prior_records, args.outdir, args.target)
        write_summary(prior_records, args.outdir, args)
        print(f"wrote {n} problems to {args.outdir}")
        return 0

    if not args.dry_run and not args.yes:
        print(f"About to make roughly {remaining * 3}-{remaining * 3 * args.max_attempts} "
              f"API calls ({GENERATOR_MODEL} generation + {JUDGE_ANTHROPIC_MODEL} "
              f"and {JUDGE_OPENAI_MODEL} verification per problem, times up to "
              f"{args.max_attempts} attempts). This costs real money.")
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1

    blocks = not args.allow_flagged_discrepancies
    manual_dir = find_manual_dir(args.manual_dir)
    manual_pairs = read_manual_pairs(manual_dir)
    if manual_pairs:
        print(f"few-shot exemplars from the hand-written dataset in {manual_dir}")
    else:
        print("warning: hand-written dataset not found; falling back to the "
              "three built-in exemplars. Pass --manual-dir to point at it.",
              file=sys.stderr)
    exemplars = build_exemplar_block(manual_pairs)
    writer = JsonlWriter(records_path)
    records = list(prior_records)

    if args.dry_run:
        source_iter: Iterator[dict] = iter_fake_problems(remaining * 2)
    else:
        source_iter = iter_source_problems(
            args.dataset, args.split, args.min_len, args.max_len, args.seed,
            not args.allow_noncompiling)
    source_iter = (s for s in source_iter if s["source_id"] not in prior_ids)

    try:
        from tqdm import tqdm
        bar = tqdm(total=args.target, initial=args.target - remaining,
                   unit="problem", desc="accepted")
    except ImportError:
        bar = None

    accepted_count = args.target - remaining
    exhausted = False
    start = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            in_flight: dict = {}
            while accepted_count < args.target:
                # Keep the pool fed, but never start more work than could
                # still be needed.
                while (not exhausted
                       and len(in_flight) < args.workers * 2
                       and accepted_count + len(in_flight) < args.target):
                    source = next(source_iter, None)
                    if source is None:
                        exhausted = True
                        break
                    fut = pool.submit(process_source, clients, source,
                                      exemplars, args, blocks)
                    in_flight[fut] = source

                if not in_flight:
                    break

                done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for fut in done:
                    source = in_flight.pop(fut)
                    try:
                        record = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        record = {
                            "seq": source["seq"], "source_id": source["source_id"],
                            "source_problem": source["problem"],
                            "accepted": False, "status": "worker_error",
                            "error": f"{type(exc).__name__}: {exc}", "attempts": [],
                        }
                    records.append(record)
                    writer.write(record)
                    if record.get("accepted"):
                        accepted_count += 1
                        if bar:
                            bar.update(1)

                # Periodic flush so a kill -9 still leaves usable .txt files.
                if len(records) % 25 == 0:
                    write_dataset_files(records, args.outdir, args.target)

            if exhausted and accepted_count < args.target:
                print(f"\nwarning: source dataset exhausted with "
                      f"{accepted_count}/{args.target} accepted. Lower "
                      f"--threshold or raise --max-len to widen the pool.",
                      file=sys.stderr)
    except KeyboardInterrupt:
        print("\ninterrupted -- writing what has been accepted so far",
              file=sys.stderr)
    finally:
        if bar:
            bar.close()
        writer.close()

    n = write_dataset_files(records, args.outdir, args.target)
    summary = write_summary(records, args.outdir, args)

    print()
    print(f"accepted            : {n}/{args.target}")
    print(f"source problems seen: {summary['problems_processed']}")
    print(f"acceptance rate     : {summary['acceptance_rate']:.1%}")
    print(f"status breakdown    : {summary['status_counts']}")
    print(f"elapsed             : {time.time() - start:.0f}s")
    print(f"output              : {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
