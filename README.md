

# AI Decisions Note
- [Live Link](https://madhuraiatwork-star-gateway-group-assignme-streamlit-app-fxyvmz.streamlit-app)

## Model & Tools Used
- **LLM:** `llama-3.1-8b-instant` served via **Groq**. Chose the 8B model over the 70B variant as a deliberate cost/latency tradeoff — for a high-volume, structure-first classification task, the extra reasoning depth of a 70B model wasn't worth the added cost and latency when the actual job (map a message to a fixed schema) doesn't require frontier-level reasoning. Groq was picked specifically for its fast inference on open-source models and free tier, keeping the pipeline cheap enough to run on every incoming message without per-message cost being a concern.
- **Structured output:** JSON schema / function-calling mode enforced via Pydantic model validation (`TriageDecision`), so every response is guaranteed to match `{category, priority, summary, suggested_action, needs_human, confidence}` or falls back to a safe default.
- **Tools used to build:** Cursor (AI pair-programming) for scaffolding and boilerplate — every function was reviewed, run, and understood before acceptance, not accepted blindly.
- **No external tool-calling** was used at runtime (kept the pipeline deterministic and auditable for a triage system); this is one of the "fix with more time" items below.

## Prompt Strategy
- System prompt explicitly instructs the model to treat the incoming customer message as **untrusted data to classify, never as instructions to follow** — this defends against the adversarial/prompt-injection message in the dataset (one message attempted to override the classifier's behavior; it was correctly classified as a normal message rather than executed as a command).
- The prompt is intentionally narrow-scoped: the model is told its only job is triage classification, with no other capability, to reduce the attack surface for hijacking.
- Non-English and gibberish messages are explicitly handled in-prompt ("classify regardless of language; if the message is empty/unintelligible, return category=unclassified, needs_human=true") rather than special-cased in code, so the model doesn't crash or refuse.

## Handling Uncertainty & Bad Input
- Every LLM call is wrapped in try/except; any API failure, timeout, or malformed response returns a safe fallback: `needs_human=True, confidence=0.0, category="unclassified"` — the system never crashes and never silently drops a message.
- Low-confidence responses (below a threshold) are routed to `needs_human=True` by design.
- **Key finding from evaluation:** LLM self-reported confidence was not well-calibrated for risk-sensitive routing — several P0/security/billing messages were answered confidently but a human labeler flagged them as needing review. Confidence alone under-flagged real risk cases.
- **Mitigation added:** a rule-based override sits alongside the model's confidence score — any message classified as `P0` priority or `security`/`billing` category is forced to `needs_human=True` regardless of the model's reported confidence, since the cost of a missed escalation on high-stakes categories is much worse than an unnecessary human review.

## How I Know It Works
- Hand-labeled 10 messages as ground truth covering clear, ambiguous, angry, and adversarial cases, then measured agreement:
  - Priority match: 90% (9/10)
  - Category match: 60% (6/10) — mismatches were mostly genuine boundary calls (e.g., billing vs. general_inquiry, feedback vs. unclassified) that a human might also disagree on
  - needs_human agreement: **90% (9/10)**, up from 40% before adding the rule-based override — the override eliminated every dangerous miss (e.g., P0/security/billing cases the model previously answered confidently but shouldn't have). The one remaining disagreement (msg-004) is a low-stakes case where the model over-flagged for human review rather than under-flagging, which is the safer direction to err on.
- Latency increased after adding the override logic (from ~0.5s/message to ~1.5-6s/message on real LLM calls). This is an explicit tradeoff I made: correctness on high-stakes routing matters more than shaving off a couple seconds per message for a system that isn't real-time-chat-latency-sensitive. Near-instant (<0.01s) responses on short-circuited garbage/empty input are unaffected.
- Estimated cost: ~[X] tokens/message average → ~$[Y] per 1,000 messages at current model pricing (Groq's llama-3.1-8b-instant free tier keeps this negligible).

## What I'd Fix With More Time
1. **Optimize the override path for latency** — the rule-based override on top of confidence added real latency (0.5s → 1.5-6s/message). I'd profile whether this is from an extra LLM call vs. Groq variance, and see if the override can be applied post-hoc on the structured output without a second model call.
2. **Calibrate confidence properly** — either fine-tune the threshold against more ground-truth data, or use a secondary "self-critique" pass where the model is asked to double-check its own answer before finalizing, rather than relying on a single confidence number.
3. **Cut cost/latency** — batch multiple short messages into a single API call, and route obviously low-ambiguity messages to a cheaper/smaller model, escalating only uncertain cases to a stronger model.
4. **Expand the eval set** beyond 10 messages and add category-specific few-shot examples to reduce the boundary-case disagreements (billing vs. general_inquiry, technical_support vs. security), and benchmark 8B vs 70B on the eval set to quantify exactly what accuracy the cost savings are trading away.
5. **Add an optional tool-call step** (e.g., look up customer account status) for cases where the message references account-specific info the model can't verify from text alone.