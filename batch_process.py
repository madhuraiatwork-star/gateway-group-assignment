"""
batch_process.py
================
Loads messages from messages.json, classifies each one using classify_message_with_meta(),
and writes all results to results.json.

Features:
  - Exponential backoff retry on Groq rate-limit errors (HTTP 429 / RateLimitError)
  - Configurable inter-request delay to stay within rate limits proactively
  - Per-call latency tracking (wall-clock seconds)
  - Per-call token usage tracking (prompt, completion, total)
  - Cumulative session totals reported on completion
  - Structured logging: message_id | category | priority | needs_human on every row
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from groq import Groq, RateLimitError

from classifier import classify_message_with_meta, _make_fallback_decision, _empty_usage
from models import TriageInput, TriageDecision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MESSAGES_FILE = Path(__file__).parent / "messages.json"
RESULTS_FILE  = Path(__file__).parent / "results.json"

# Seconds to wait between requests to be polite to Groq's rate limits
INTER_REQUEST_DELAY_S: float = 0.5

# Retry settings for HTTP 429 / RateLimitError
MAX_RETRIES: int = 4
INITIAL_BACKOFF_S: float = 2.0   # first sleep after a 429
BACKOFF_MULTIPLIER: float = 2.0  # doubles each attempt

# ---------------------------------------------------------------------------
# Pricing  (source: groq.com/pricing, llama-3.1-8b-instant, as of 2025-07)
# ---------------------------------------------------------------------------
MODEL_NAME = "llama-3.1-8b-instant"
# Dollars per million tokens
_PRICE_INPUT_PER_M:  float = 0.05   # $0.05 / 1M input tokens
_PRICE_OUTPUT_PER_M: float = 0.08   # $0.08 / 1M output tokens

# ---------------------------------------------------------------------------
# Logging setup  — console + file
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "batch_process.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("batch_process")


# ---------------------------------------------------------------------------
# Session report
# ---------------------------------------------------------------------------
def print_session_report(
    n_messages: int,
    session_tokens: dict,
    session_latency: float,
) -> None:
    """
    Prints a formatted session summary showing:
      - Total and average token counts (prompt, completion, total)
      - Total and average wall-clock latency
      - Estimated cost broken down by input/output tokens
    """
    n = max(n_messages, 1)  # avoid division by zero

    prompt_tok     = session_tokens["prompt_tokens"]
    completion_tok = session_tokens["completion_tokens"]
    total_tok      = session_tokens["total_tokens"]

    avg_prompt     = prompt_tok     / n
    avg_completion = completion_tok / n
    avg_total      = total_tok      / n
    avg_latency    = session_latency / n

    cost_input  = (prompt_tok     / 1_000_000) * _PRICE_INPUT_PER_M
    cost_output = (completion_tok / 1_000_000) * _PRICE_OUTPUT_PER_M
    cost_total  = cost_input + cost_output

    sep = "=" * 68
    log.info(sep)
    log.info("  SESSION REPORT  |  model: %s", MODEL_NAME)
    log.info(sep)
    log.info("  Messages processed : %d", n_messages)
    log.info("  Total wall-clock   : %.3f s", session_latency)
    log.info("  Avg latency/msg    : %.3f s", avg_latency)
    log.info("-" * 68)
    log.info("  Token usage (totals)")
    log.info("    Prompt tokens    : %d", prompt_tok)
    log.info("    Completion tokens: %d", completion_tok)
    log.info("    Total tokens     : %d", total_tok)
    log.info("  Token usage (per-message averages)")
    log.info("    Avg prompt       : %.1f tok", avg_prompt)
    log.info("    Avg completion   : %.1f tok", avg_completion)
    log.info("    Avg total        : %.1f tok", avg_total)
    log.info("-" * 68)
    log.info("  Estimated cost  (source: groq.com/pricing)")
    log.info("    Rate - input     : $%.4f / 1M tokens", _PRICE_INPUT_PER_M)
    log.info("    Rate - output    : $%.4f / 1M tokens", _PRICE_OUTPUT_PER_M)
    log.info("    Cost - input     : $%.6f  (%d tokens)", cost_input,  prompt_tok)
    log.info("    Cost - output    : $%.6f  (%d tokens)", cost_output, completion_tok)
    log.info("    TOTAL COST      : $%.6f", cost_total)
    log.info("    Projected /1K msgs: $%.4f", (cost_total / n_messages) * 1000 if n_messages else 0)
    log.info(sep)


# ---------------------------------------------------------------------------
# Core classify-with-retry
# ---------------------------------------------------------------------------
def classify_with_retry(
    raw_text: str,
    message_id: str,
    client: Groq,
) -> tuple[TriageDecision, dict, float]:
    """
    Calls classify_message_with_meta() with exponential-backoff retry on
    RateLimitError.  Returns (decision, usage_dict, latency_seconds).
    On permanent failure returns the fallback decision.
    """
    backoff = INITIAL_BACKOFF_S

    for attempt in range(1, MAX_RETRIES + 1):
        t_start = time.perf_counter()
        try:
            decision, usage = classify_message_with_meta(raw_text, client=client)
            latency = time.perf_counter() - t_start
            return decision, usage, latency

        except RateLimitError:
            latency = time.perf_counter() - t_start
            if attempt == MAX_RETRIES:
                log.warning(
                    "[%s] Rate limit hit on attempt %d/%d — giving up, using fallback.",
                    message_id, attempt, MAX_RETRIES,
                )
                return _make_fallback_decision(), _empty_usage(), latency

            log.warning(
                "[%s] Rate limit hit (attempt %d/%d). Sleeping %.1fs before retry...",
                message_id, attempt, MAX_RETRIES, backoff,
            )
            time.sleep(backoff)
            backoff *= BACKOFF_MULTIPLIER

        except Exception as exc:
            latency = time.perf_counter() - t_start
            log.error("[%s] Unexpected error: %s. Using fallback.", message_id, exc)
            return _make_fallback_decision(), _empty_usage(), latency

    # Should never reach here
    return _make_fallback_decision(), _empty_usage(), 0.0


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------
def run_batch() -> None:
    # Load input
    if not MESSAGES_FILE.exists():
        log.error("Input file not found: %s", MESSAGES_FILE)
        raise FileNotFoundError(MESSAGES_FILE)

    raw_messages: list[dict] = json.loads(MESSAGES_FILE.read_text(encoding="utf-8"))
    log.info("Loaded %d messages from %s", len(raw_messages), MESSAGES_FILE.name)

    # Validate input with Pydantic
    inputs: list[TriageInput] = []
    for raw in raw_messages:
        try:
            inputs.append(TriageInput(**raw))
        except Exception as exc:
            log.warning("Skipping invalid message entry %s: %s", raw, exc)

    if not inputs:
        log.error("No valid messages to process.")
        return

    # Shared Groq client (one connection, reused across calls)
    client = Groq()

    results: list[dict] = []

    # Session-level accumulators
    session_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    session_latency = 0.0

    log.info("=" * 72)
    log.info("%-12s | %-22s | %-8s | %-12s | %s",
             "message_id", "category", "priority", "needs_human", "confidence")
    log.info("-" * 72)

    for idx, t_input in enumerate(inputs, start=1):
        decision, usage, latency = classify_with_retry(
            raw_text=t_input.raw_text,
            message_id=t_input.message_id,
            client=client,
        )

        # Accumulate session stats
        for key in session_tokens:
            session_tokens[key] += usage[key]
        session_latency += latency

        # Structured log line
        log.info(
            "%-12s | %-22s | %-8s | %-12s | %.2f  (lat=%.3fs tok=%d)",
            t_input.message_id,
            decision.category[:22],
            decision.priority,
            str(decision.needs_human),
            decision.confidence,
            latency,
            usage["total_tokens"],
        )

        # Build result record
        result_record = {
            "message_id": t_input.message_id,
            "raw_text": t_input.raw_text,
            "decision": decision.model_dump(),
            "meta": {
                "latency_seconds": round(latency, 4),
                "tokens": usage,
            },
        }
        results.append(result_record)

        # Inter-request delay (skip after last message)
        if idx < len(inputs):
            time.sleep(INTER_REQUEST_DELAY_S)

    log.info("=" * 72)
    print_session_report(
        n_messages=len(results),
        session_tokens=session_tokens,
        session_latency=session_latency,
    )

    # Write results
    RESULTS_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Results written to %s", RESULTS_FILE)


if __name__ == "__main__":
    run_batch()
