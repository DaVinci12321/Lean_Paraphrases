"""
lp_common.py -- shared plumbing for the Lean Paraphrases extension pipeline.

Four scripts sit on top of this module:

    generate_paraphrase_dataset.py   build the paraphrase dataset
    autoformalize_claude.py          autoformalize it with Claude Opus 5
    autoformalize_gpt.py             autoformalize it with GPT 5.6 Sol
    check_autoformalizations.py      judge all four autoformalization files

Everything model-facing, retry-related or file-format-related lives here so the
two autoformalization scripts differ only in their model configuration block.

Line-aligned file convention
----------------------------
Every .txt file this pipeline produces holds exactly one problem per line, and
line N of any two of them refers to the same problem. That makes the files
trivially indexable and comparable, which is the whole point, but it means Lean
blocks have to be collapsed onto a single line. Each script therefore also
writes a .jsonl sidecar holding the verbatim multi-line model output; the .txt
is the index, the .jsonl is the record of truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

ANTHROPIC_MODEL = "claude-opus-5"
OPENAI_MODEL = "gpt-5.6-sol"

# Non-retryable HTTP statuses: retrying a malformed request just wastes money.
NON_RETRYABLE = {400, 401, 403, 404, 422}


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

def one_line(text: str) -> str:
    """Collapse text to a single whitespace-normalised line."""
    return re.sub(r"\s+", " ", (text or "").replace("\\n", " ")).strip()


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


FENCE_RE = re.compile(r"```(?:lean4?|Lean4?)?\s*\n?(.*?)```", re.S)


def strip_code_fence(text: str) -> str:
    """Pull the code out of a ```lean ...``` block if the model wrapped it."""
    if not text:
        return ""
    m = FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


PREAMBLE_RE = re.compile(r"^\s*(import|open|set_option|variable|universe)\b")
DECL_RE = re.compile(r"(?m)^\s*(theorem|lemma|example)\b")
HELPER_DECL_RE = re.compile(
    r"(?m)^\s*(noncomputable\s+def|def|abbrev|instance|structure|inductive)\b")


def split_lean(block: str) -> tuple[list[str], str]:
    """Split a Lean block into its preamble lines and the declaration body.

    Returns ([] , "") when there is no theorem/lemma/example to be found.
    """
    block = strip_code_fence(block)
    m = DECL_RE.search(block)
    if not m:
        return [], ""
    preamble = [ln.strip() for ln in block[:m.start()].splitlines()
                if ln.strip() and PREAMBLE_RE.match(ln)]
    return preamble, block[m.start():].strip()


def ensure_sorry(statement: str) -> str:
    """Make sure a flattened statement ends in a `sorry` placeholder proof."""
    s = statement.rstrip()
    if re.search(r"\bsorry\s*$", s):
        return s
    if re.search(r":=\s*by\s*$", s):
        return s + " sorry"
    if s.endswith(":="):
        return s + " by sorry"
    return s + " := by sorry"


def strip_lean_comments(body: str) -> str:
    """Remove Lean comments before a multi-line block is collapsed to one line.

    A `--` line comment runs to the end of its line, so joining the lines first
    would let it swallow the rest of the declaration. Block comments are
    dropped for the same reason.
    """
    body = re.sub(r"/-.*?-/", " ", body, flags=re.DOTALL)
    return "\n".join(re.sub(r"--.*$", "", ln) for ln in body.splitlines())


_LET_KW_RE = re.compile(r"^\s*\(*\s*(let|have)\s")


def close_let_bindings(body: str) -> str:
    """Insert the `;` that a newline used to stand in for after a `let` binding.

    Lean 4 lets a term-level `let`/`have` binding end at a newline, so
    collapsing a declaration onto one line silently glues the binding to the
    term it scopes over: `let t := e` followed by `body` becomes `let t := e
    body`, which parses as an application. Restoring the explicit separator
    keeps the flattened statement parseable. A binding is taken to end at the
    first following line indented no deeper than the binding keyword itself,
    which is the same rule Lean's whitespace sensitivity uses.
    """
    lines = body.splitlines()
    out: list[str] = []
    pending: list[int] = []
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        while pending and indent <= pending[-1]:
            pending.pop()
            if out and not out[-1].rstrip().endswith((";", ",")):
                out[-1] = out[-1].rstrip() + ";"
        m = _LET_KW_RE.match(line)
        if m:
            pending.append(m.start(1))
        out.append(line)
    return "\n".join(out)


def flatten_lean(block: str) -> str:
    """Collapse a Lean block to one line: the declaration, ending in `sorry`.

    The `import` / `open` preamble is dropped -- it is recorded in the .jsonl
    sidecar instead. Returns "" when no declaration could be found, which the
    callers treat as a failed formalization rather than silently writing a
    blank line.
    """
    _, body = split_lean(block)
    if not body:
        return ""
    return ensure_sorry(one_line(close_let_bindings(strip_lean_comments(body))))


def has_helper_decls(block: str) -> bool:
    """True when the block defines something before the theorem.

    Such a block cannot be collapsed onto one line without breaking it, since
    `def f : N -> N | 0 => 1 | n => n` style definitions are newline-delimited.
    """
    _, body = split_lean(block)
    if not body:
        return True
    head = strip_code_fence(block)
    return bool(HELPER_DECL_RE.search(head)) or len(DECL_RE.findall(head)) != 1


def lean_preamble(block: str) -> str:
    return " | ".join(split_lean(block)[0])


def rebuild_lean(preamble: str, statement: str) -> str:
    """Reassemble a readable multi-line Lean block for a prompt."""
    lines = [p.strip() for p in (preamble or "import Mathlib").split("|") if p.strip()]
    return "\n".join(lines) + "\n\n" + statement


# --------------------------------------------------------------------------
# Line-aligned file IO
# --------------------------------------------------------------------------

def read_lines(path: Path) -> list[str]:
    """Read a line-aligned .txt file. Blank lines are preserved as empty
    entries so line N stays problem N even when a stage failed on it."""
    if not path.exists():
        raise FileNotFoundError(f"missing input file: {path}")
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [ln.strip() for ln in lines]


def write_lines(path: Path, lines: Iterable[str]) -> int:
    items = [one_line(x) for x in lines]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")
    return len(items)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a torn final line from a hard kill
    return out


class JsonlWriter:
    """Thread-safe append-as-you-go writer, so an interrupted run keeps its
    completed work and can be resumed."""

    def __init__(self, path: Path, append: bool = True):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = path.open("a" if append else "w", encoding="utf-8")

    def write(self, record: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# API clients
# --------------------------------------------------------------------------

def with_retries(fn: Callable[[], Any], attempts: int, label: str) -> Any:
    """Exponential backoff with jitter. 4xx client errors are not retried."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - each SDK raises its own types
            status = getattr(exc, "status_code", None)
            if status in NON_RETRYABLE or i == attempts - 1:
                raise
            delay = min(60.0, 2.0 * (2 ** i)) * (0.5 + random.random())
            print(f"  [retry {i + 1}/{attempts - 1}] {label}: {exc} "
                  f"-- sleeping {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("unreachable")


class Clients:
    """Lazily constructed API clients, shared across worker threads.

    Both SDK clients are safe to share; every call made through them here is
    stateless.
    """

    def __init__(self, retries: int = 5, effort: str = "medium",
                 max_tokens: int = 8000):
        self.retries = retries
        self.effort = effort
        self.max_tokens = max_tokens
        self._openai = None
        self._anthropic = None
        self._lock = threading.Lock()

    @property
    def openai(self):
        with self._lock:
            if self._openai is None:
                from openai import OpenAI
                self._openai = OpenAI()
            return self._openai

    @property
    def anthropic(self):
        with self._lock:
            if self._anthropic is None:
                from anthropic import Anthropic
                self._anthropic = Anthropic()
            return self._anthropic

    # -- structured (JSON schema) calls -----------------------------------

    def openai_json(self, system: str, user: str, schema: dict,
                    name: str, model: str = OPENAI_MODEL) -> dict:
        def call():
            resp = self.openai.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": name, "strict": True,
                                    "schema": schema},
                },
            )
            return json.loads(resp.choices[0].message.content)

        return with_retries(call, self.retries, f"openai/{model}/{name}")

    def anthropic_json(self, system: str, user: str, schema: dict,
                       name: str, model: str = ANTHROPIC_MODEL) -> dict:
        def call():
            resp = self.anthropic.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    # No "minimum"/"maximum" anywhere in these schemas: neither
                    # provider's strict JSON-schema mode accepts numeric bounds.
                    "format": {"type": "json_schema", "schema": schema},
                },
                messages=[{"role": "user", "content": user}],
            )
            return json.loads(next(b.text for b in resp.content
                                   if b.type == "text"))

        return with_retries(call, self.retries, f"anthropic/{model}/{name}")

    # -- plain text calls (used for autoformalization) ---------------------

    def openai_text(self, prompt: str, model: str = OPENAI_MODEL) -> str:
        def call():
            resp = self.openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

        return with_retries(call, self.retries, f"openai/{model}/text")

    def anthropic_text(self, prompt: str, model: str = ANTHROPIC_MODEL) -> str:
        def call():
            resp = self.anthropic.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        return with_retries(call, self.retries, f"anthropic/{model}/text")


# --------------------------------------------------------------------------
# Verdict helpers
# --------------------------------------------------------------------------

def confidence(verdict: dict, key: str) -> float:
    """Read a confidence, clamped to [0, 1]. A missing or unparseable value is
    treated as zero confidence, i.e. as maximum doubt."""
    try:
        return max(0.0, min(1.0, float(verdict.get(key))))
    except (TypeError, ValueError):
        return 0.0


def require_keys(*names: str) -> list[str]:
    """Return the missing credential environment variables."""
    return [n for n in names if not os.environ.get(n)]
