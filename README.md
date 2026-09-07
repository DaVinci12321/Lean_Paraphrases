# Lean Paraphrases

Code and data for **"Lean Paraphrases: Analyzing the Effect of Wording on Autoformalization Output"** — a study of how paraphrasing a natural-language math problem (holding its meaning fixed, varying only how rigorous/symbolic vs. wordy the phrasing is) changes what a Lean 4 autoformalization model produces.

This repository contains two studies:

| | `Manual/` | `Extension/` |
|---|---|---|
| Scale | 50 problems × 2 paraphrases = 100 | 1000 problems × 2 paraphrases = 2000 (890 problems after quality pruning) |
| Paraphrase pairs | Hand-written | LLM-generated, dual-LLM-judge verified |
| Source problems | Lean Workbook, Constructive Bench (Enumerate-Conjecture-Prove), Prover Bench (DeepSeek-Prover) | FormalMATH |
| Correctness checking | Manual (by hand) | Automated, dual-LLM judge |
| Models formalizing | GPT 5.6 Sol, Claude Opus 5 | GPT 5.6 Sol, Claude Opus 5 |

`Extension/` automates and scales up exactly what `Manual/` did by hand, using the 50 manual pairs as few-shot exemplars and replacing manual correctness-checking with a calibrated, threshold-gated dual-LLM judge. See the paper for the full experimental design, methodology, and results.

## Repository layout

```
Manual/
  leanparaphrasesdataset.txt        100 problems, one per line: lines 1-50 rigorous, 51-100 non-rigorous
  rigorousparaphrases.txt           the 50 rigorous paraphrases alone
  nonrigorousparaphrases.txt        the 50 non-rigorous paraphrases alone
  leanparaphrasescode_gpt.py        runs the 100 problems through GPT 5.6 Sol
  leanparaphrasescode_anthropic.py  runs the 100 problems through Claude Opus 5

Extension/
  run_pipeline.py                   single entry point: generate -> formalize -> judge
  generate_paraphrase_dataset.py    stage 1: build paraphrase pairs from FormalMATH
  autoformalize_gpt.py              stage 2: formalize with GPT 5.6 Sol
  autoformalize_claude.py           stage 2: formalize with Claude Opus 5
  check_autoformalizations.py       stage 3: dual-LLM equivalence judging
  prune_disagreements.py            removes problems where the judges disagreed on any configuration
  compare_formalizations.py         diffs each model's rigorous vs. non-rigorous output, attributes differences to wording (via API calls)
  manual_analysis.py                same comparison, without API calls: feeds batches of pairs to whoever is running it and ingests the analysis into the same schema
  lp_common.py                      shared API/client/retry/file-IO plumbing
  dataset/                          generated paraphrase pairs + reference Lean statements
  formalized/{gpt,claude}/          model outputs
  evaluation/                       per-problem equivalence verdicts + summary report
  comparison/                       per-problem wording-attributed differences (used for the paper's worked examples)
  removed_problems/                 the problems pruned for judge disagreement, and why
  README.md                         full pipeline documentation (flags, cost estimates, quality-gate details)
```

`Extension/README.md` has the complete documentation for that pipeline (every flag, what each stage produces, cost estimates, and how the quality gates work). This top-level file is the map; that one is the manual.

## Reproducing the manual study

```bash
cd Manual
pip install openai anthropic
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
python leanparaphrasescode_gpt.py        > gpt_output.txt
python leanparaphrasescode_anthropic.py  > claude_output.txt
```

Each script reads `leanparaphrasesdataset.txt` line by line (100 lines: 50 rigorous, then 50 non-rigorous) and prints the prompt plus the model's Lean 4 formalization for each. In the paper, these outputs were then checked by hand against the source problem statements — there is no automated grader in `Manual/`.

## Reproducing the automated extension

```bash
cd Extension
pip install -r requirements.txt
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
python run_pipeline.py --target 1000
```

This single command runs all three stages (generate paraphrase pairs, formalize with both models, judge equivalence with both models) and reports the same headline statistics as the paper: per-configuration equivalence rate, how often a model's verdict flips between the two wordings of the same problem, and cross-model agreement. See `Extension/README.md` for `--dry-run` (no API calls, exercises the whole pipeline with stub models), `--target 10` (a small pilot), and every other flag.

`compare_formalizations.py` and `manual_analysis.py` reproduce the paper's worked examples (e.g. the functional-equation problems in the Results section) by diffing a model's rigorous-paraphrase formalization against its non-rigorous one and attributing each difference to a specific piece of wording.

**Cost note:** a full `--target 1000` run is roughly 17,000 API calls split across two providers. See `Extension/README.md` for a per-stage cost breakdown. `--dry-run` and `--target 10` are the cheap ways to sanity-check the pipeline before committing to a full run.

## Data sources and credit

The manual dataset's 50 source problems, and the automated extension's source pool, were drawn from existing formal-informal problem-pair datasets. Please credit and respect the license of whichever you use downstream:

- **Lean Workbook** — Ying et al., *Lean Workbook: A Large-scale Lean Problem Set Formalized from Natural Language Math Problems*, arXiv:2406.03847
- **Enumerate-Conjecture-Prove / Constructive Bench** — Sun et al., arXiv:2505.18492
- **DeepSeek-Prover / Prover Bench** — Xin et al., arXiv:2405.14333
- **FormalMATH** — Yu et al., *FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models*, arXiv:2505.02735 (`SphereLab/FormalMATH-All` on Hugging Face)

The paraphrases, autoformalizations, and judge verdicts generated by this project are original to this work.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The copyright line currently reads "Anonymous Author(s)" to match the paper's anonymized submission — update it with the real author name(s) once the review period ends. The released datasets (paraphrase pairs, model outputs, judge verdicts) are derivative of the sources listed above; if you redistribute them, you are responsible for complying with each source's own license in addition to this repository's.

## Citation

Citation details (and a link to the paper) will be added once the review process concludes. In the meantime, please refer to this repository by name and commit hash if you build on it.

## Questions / issues

Please open an issue on this repository.
