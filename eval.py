"""
eval.py
=======
Evaluates the triage classifier against hand-labeled ground truth.

Loads:
  ground_truth.json  — list of {message_id, raw_text, category, priority, needs_human}

For each entry, runs classify_message() and compares against the label.

Reports:
  - % exact category match
  - % exact priority match
  - % needs_human agreement
  - A mismatch table (category / priority / needs_human columns) for any row
    where at least one field disagrees, showing expected vs. actual side-by-side.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from classifier import classify_message
from models import TriageDecision

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth.json"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class GroundTruthEntry:
    message_id: str
    raw_text: str
    category: str
    priority: str
    needs_human: bool


@dataclass
class EvalResult:
    entry: GroundTruthEntry
    prediction: TriageDecision
    latency: float          # seconds

    # Computed match flags
    category_match: bool = field(init=False)
    priority_match: bool = field(init=False)
    needs_human_match: bool = field(init=False)
    has_mismatch: bool = field(init=False)

    def __post_init__(self):
        self.category_match     = self.entry.category    == self.prediction.category
        self.priority_match     = self.entry.priority    == self.prediction.priority
        self.needs_human_match  = self.entry.needs_human == self.prediction.needs_human
        self.has_mismatch       = not (
            self.category_match and self.priority_match and self.needs_human_match
        )


# ---------------------------------------------------------------------------
# Table rendering helpers
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_BOLD   = "\033[1m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"

def _cell(value: str, match: bool) -> str:
    """Colour a cell value green (match) or red (mismatch)."""
    colour = _GREEN if match else _RED
    return f"{colour}{value}{_RESET}"

def _tick(match: bool) -> str:
    return f"{_GREEN}[OK]{_RESET}" if match else f"{_RED}[X]{_RESET}"

def _pct(num: int, den: int) -> str:
    if den == 0:
        return "N/A"
    return f"{(num / den) * 100:.1f}%"


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------
def load_ground_truth(path: Path) -> list[GroundTruthEntry]:
    raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for item in raw:
        try:
            entries.append(GroundTruthEntry(
                message_id  = item["message_id"],
                raw_text    = item["raw_text"],
                category    = item["category"],
                priority    = item["priority"],
                needs_human = bool(item["needs_human"]),
            ))
        except KeyError as exc:
            print(f"[WARN] Skipping malformed entry {item.get('message_id', '?')}: {exc}")
    return entries


def run_eval(entries: list[GroundTruthEntry]) -> list[EvalResult]:
    results: list[EvalResult] = []
    total = len(entries)

    print(f"\nRunning classifier on {total} messages...\n")
    for idx, entry in enumerate(entries, start=1):
        t0 = time.perf_counter()
        prediction = classify_message(entry.raw_text)
        latency = time.perf_counter() - t0

        result = EvalResult(entry=entry, prediction=prediction, latency=latency)
        results.append(result)

        status = "MISMATCH" if result.has_mismatch else "OK"
        status_col = f"{_RED}[MISMATCH]{_RESET}" if result.has_mismatch else f"{_GREEN}[OK]{_RESET}"
        print(f"  [{idx:>2}/{total}] {entry.message_id}  ->  {status_col}  ({latency:.3f}s)")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(results: list[EvalResult]) -> None:
    n = len(results)
    cat_ok   = sum(1 for r in results if r.category_match)
    pri_ok   = sum(1 for r in results if r.priority_match)
    nh_ok    = sum(1 for r in results if r.needs_human_match)
    mismatches = [r for r in results if r.has_mismatch]

    # ---- Metric summary ----
    print(f"\n{_BOLD}{'='*60}{_RESET}")
    print(f"{_BOLD}  EVALUATION SUMMARY  ({n} messages){_RESET}")
    print(f"{_BOLD}{'='*60}{_RESET}")

    metrics = [
        ("Category match",    cat_ok, n),
        ("Priority match",    pri_ok, n),
        ("needs_human agree", nh_ok,  n),
    ]
    for label, ok, total in metrics:
        bar_filled = int((ok / total) * 20) if total else 0
        bar        = f"[{'#' * bar_filled}{'-' * (20 - bar_filled)}]"
        colour     = _GREEN if ok == total else (_YELLOW if ok >= total * 0.7 else _RED)
        print(f"  {label:<22}  {colour}{bar}{_RESET}  {_pct(ok, total):>6}  ({ok}/{total})")

    print(f"{_BOLD}{'='*60}{_RESET}")

    # ---- Mismatch table ----
    if not mismatches:
        print(f"\n{_GREEN}  All predictions match ground truth!{_RESET}\n")
        return

    print(f"\n{_BOLD}  MISMATCH DETAILS  ({len(mismatches)} of {n} messages){_RESET}\n")

    # Column widths
    col = {
        "id":  12,
        "field": 12,
        "expected": 24,
        "actual": 24,
    }
    header = (
        f"{'ID':<{col['id']}}  "
        f"{'FIELD':<{col['field']}}  "
        f"{'EXPECTED':<{col['expected']}}  "
        f"{'ACTUAL':<{col['actual']}}"
    )
    divider = "-" * len(header)

    print(f"  {_BOLD}{header}{_RESET}")
    print(f"  {divider}")

    for r in mismatches:
        mid = r.entry.message_id
        rows_printed = 0

        checks = [
            ("category",    str(r.entry.category),   str(r.prediction.category),  r.category_match),
            ("priority",    str(r.entry.priority),    str(r.prediction.priority),  r.priority_match),
            ("needs_human", str(r.entry.needs_human), str(r.prediction.needs_human), r.needs_human_match),
        ]
        for field_name, expected, actual, match in checks:
            if match:
                continue   # Only show mismatching fields
            id_col   = mid if rows_printed == 0 else ""
            exp_cell = f"{_RED}{expected}{_RESET}"
            act_cell = f"{_RED}{actual}{_RESET}"
            print(
                f"  {id_col:<{col['id']}}  "
                f"{field_name:<{col['field']}}  "
                f"{exp_cell:<{col['expected'] + len(_RED) + len(_RESET)}}  "
                f"{act_cell}"
            )
            rows_printed += 1

        # Separator between different message blocks
        print(f"  {'.' * len(divider)}")

    print()


def print_full_table(results: list[EvalResult]) -> None:
    """Prints the per-message full comparison table (all rows, all fields)."""
    print(f"\n{_BOLD}  FULL COMPARISON TABLE{_RESET}\n")

    header = (
        f"{'ID':<12}  "
        f"{'CAT EXP':<22}  "
        f"{'CAT ACT':<22}  "
        f"{'PRI E':>5}  "
        f"{'PRI A':>5}  "
        f"{'NH E':>5}  "
        f"{'NH A':>5}  "
        f"{'LAT':>7}"
    )
    print(f"  {_BOLD}{header}{_RESET}")
    print(f"  {'-' * len(header)}")

    for r in results:
        cat_e  = r.entry.category
        cat_a  = r.prediction.category
        pri_e  = r.entry.priority
        pri_a  = r.prediction.priority
        nh_e   = str(r.entry.needs_human)[0]    # T / F
        nh_a   = str(r.prediction.needs_human)[0]

        print(
            f"  {r.entry.message_id:<12}  "
            f"{_cell(f'{cat_e:<22}', r.category_match)}  "
            f"{_cell(f'{cat_a:<22}', r.category_match)}  "
            f"{_cell(f'{pri_e:>5}', r.priority_match)}  "
            f"{_cell(f'{pri_a:>5}', r.priority_match)}  "
            f"{_cell(f'{nh_e:>5}', r.needs_human_match)}  "
            f"{_cell(f'{nh_a:>5}', r.needs_human_match)}  "
            f"{r.latency:>6.3f}s"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    if not GROUND_TRUTH_FILE.exists():
        print(f"[ERROR] Ground truth file not found: {GROUND_TRUTH_FILE}", file=sys.stderr)
        sys.exit(1)

    entries = load_ground_truth(GROUND_TRUTH_FILE)
    if not entries:
        print("[ERROR] No valid entries found in ground_truth.json", file=sys.stderr)
        sys.exit(1)

    results = run_eval(entries)

    print_full_table(results)
    print_summary(results)


if __name__ == "__main__":
    main()
