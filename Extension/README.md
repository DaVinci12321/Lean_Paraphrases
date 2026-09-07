# Lean Paraphrases — automated extension

Scales the manual 50-problem study ("Lean Paraphrases: Analyzing the Effect of
Wording on Autoformalization Output") to 1000 problems, and automates the
manual checking that capped the original.

One command runs everything:

```bash
python run_pipeline.py --target 1000
```

Under it are four stages, each also runnable on its own:

```
stage 1   generate_paraphrase_dataset.py   build 1000 paraphrase pairs
stage 2   autoformalize_claude.py          formalize them with Claude Opus 5
          autoformalize_gpt.py             formalize them with GPT 5.6 Sol
stage 3   check_autoformalizations.py      judge all four output files
```

`lp_common.py` holds the shared plumbing (API clients, retries, Lean
flattening, line-aligned file IO). The two stage-2 scripts are byte-identical
apart from one model configuration block.

Stage 1 uses the paper's 50 hand-written pairs for few-shot exemplars and for
`--calibrate`. It looks for `rigorousparaphrases.txt` and
`nonrigorousparaphrases.txt` next to the script, then one level up (where they
currently live, alongside the paper). `--manual-dir` overrides the search.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Running it

```bash
python run_pipeline.py --target 1000
```

That runs calibration, builds the dataset, autoformalizes with both models, and
judges the results — asking for confirmation once, up front, with a total call
estimate, rather than one prompt per script.

After calibration it pauses once to show how many of the paper's 50
hand-written pairs each threshold would accept, and lets you keep `--threshold`
or type a new one. That table is the whole reason to calibrate, so the pause is
where it earns its keep. `--yes` skips it and runs unattended.

| flag | effect |
|---|---|
| `--dry-run` | stub models, no API calls, no credentials — exercises every stage |
| `--target 10` | small pilot through all four stages |
| `--resume` | continue a half-finished stage |
| `--force` | discard a stage's output and rebuild it |
| `--only check` | run one stage (`calibrate`, `dataset`, `formalize`, `check`) |
| `--skip-calibrate --yes` | fully unattended |
| `--sequential-formalize` | run the two autoformalizers one at a time |

Completed stages are detected and skipped, so re-running the command is cheap
and safe. A *half*-finished stage stops the run with a message telling you to
choose `--resume` or `--force` — it will neither silently discard a long job
nor silently reuse stale output. Progress is recorded in
`pipeline_state.json`.

By default the two autoformalizers run at the same time; they hit different
providers, so neither competes with the other for rate limit. Their output goes
to `logs/formalize_claude.log` and `logs/formalize_gpt.log` so the two progress
bars don't interleave.

Each stage also runs standalone, with the same `--dry-run` / `--resume` /
`--limit` flags:

```bash
python generate_paraphrase_dataset.py --calibrate
python generate_paraphrase_dataset.py --target 1000 --workers 8
python autoformalize_claude.py
python autoformalize_gpt.py
python check_autoformalizations.py
```

## What gets produced

```
dataset/
  rigorousparaphrases.txt      symbol-heavy paraphrase, one per line
  nonrigorousparaphrases.txt   word-heavy paraphrase, one per line
  formalstatements.txt         reference Lean 4 statement, one per line
  dataset.jsonl                all of the above plus provenance, verbatim
  records.jsonl                every generation attempt and every verdict
  calibration.jsonl            per-pair calibration scores
  summary.json                 acceptance statistics

formalized/claude/
  formalized_rigorous.txt      Claude's Lean 4 output, one per line
  formalized_nonrigorous.txt
  formalized_*.jsonl           verbatim model output plus metadata
formalized/gpt/                the same four files from GPT

evaluation/
  gpt_rigorous.jsonl           per-problem verdicts from both judges
  gpt_nonrigorous.jsonl
  claude_rigorous.jsonl
  claude_nonrigorous.jsonl
  report.txt                   statistics + every undecided problem number
  report.json                  the same, machine-readable

logs/                          stage-2 output when the two run in parallel
pipeline_state.json            which stages finished, when, with what settings
```

**Every `.txt` file is line-aligned.** Line N of any two of them is the same
problem, so comparing a formalization against its source is `sed -n 'Np'`. When
a stage fails on a problem the line is left empty rather than shifted, and the
`.jsonl` sidecar records why.

## How the quality gates work

**Stage 1.** Claude Opus 5 writes both paraphrases from a
`SphereLab/FormalMATH-All` problem. Claude Opus 5 and GPT 5.6 Sol then each
score four claims with a 0–1 confidence — rigorous ≡ source, non-rigorous ≡
source, the two ≡ each other, and whether the rigor contrast is real. A pair is
kept only when every claim from *both* judges is true and every confidence
clears `--threshold` (default 0.85). Rejected pairs go back to the generator
with the judges' specific objections, up to three attempts.

Because every FormalMATH row ships a Lean statement known to compile, the
generator and the verifiers both see it and treat it as authoritative for
variable domains and quantifier structure. `--no-formal-hint` turns that off
for an ablation.

**Stage 3.** Each judge sees three things: the natural-language paraphrase, the
candidate formalization, and the reference Lean statement as a worked example.
The reference is explicitly context, not an answer key — a candidate that
differs from it but still matches the English counts as equivalent. Both judges
must agree *and* both clear the threshold:

| outcome | meaning |
|---|---|
| `EQUIVALENT` | both said yes, both confident |
| `NOT_EQUIVALENT` | both said no, both confident |
| `UNDECIDED` | judges disagreed, or one was below threshold — needs a human |
| `MISSING` | stage 2 produced nothing for this line |

`report.txt` lists the undecided problem numbers per configuration, with the
reason for each, and reports the paper's central statistic: how often a model's
verdict flips between the two wordings of the *same* problem.

## Choices worth knowing about

- **One line per problem forces flattening.** Lean statements are collapsed to
  a single whitespace-normalised line ending in `sorry`, and the `import` /
  `open` preamble is dropped into the `.jsonl`. Prepend `import Mathlib` to
  compile a line. Source rows whose formalization defines helper functions
  before the theorem can't be flattened safely, so they are filtered out at
  stage 1 (647 of 5560 rows).
- **The stage-2 prompt is the paper's, plus one constraint.** The original
  prompt is used verbatim, followed by an instruction to emit a single
  self-contained `theorem` — without it, multi-declaration answers can't be
  line-aligned. `--paper-prompt-only` drops the addition for a faithful
  replication; expect some answers to come back unparseable.
- **The threshold is a precision knob.** A bad pair silently corrupts the
  experiment; a rejected one only costs API calls. But the FormalMATH pool is
  finite — 4404 rows survive filtering, so a 1000-problem target needs roughly
  a 23% acceptance rate. If a run exhausts the pool, lower `--threshold` (after
  looking at the calibration table) or raise `--max-len`.
- **Cost.** Calibration is ~100 calls. Stage 1 is ~3 per accepted problem plus
  3 per repair. Stage 2 is 2 per problem per model. Stage 3 is 8 per problem.
  For 1000 problems the whole pipeline is roughly **17,000 calls** (~9,300
  Claude, ~7,700 GPT); `run_pipeline.py` prints its own estimate before
  spending anything.
- **Reasoning effort defaults to `medium`.** Adaptive thinking is billed as
  output, and at 1000 problems output tokens are ~87% of the bill, so effort is
  the dominant cost lever — far more than anything on the input side. At
  `medium` the Claude half of a full run lands near **$250**; `--effort high`
  roughly doubles that. Measured input cost is only ~$70 either way. Raise
  effort per stage if the judging looks sloppy: `--effort high` is worth
  considering for `check_autoformalizations.py`, which is the stage where a
  wrong verdict silently corrupts the results.
