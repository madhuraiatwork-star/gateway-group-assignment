from models import TriageInput, TriageDecision
from pydantic import ValidationError

def test_models():
    # 1. Test TriageInput
    input_data = TriageInput(message_id="msg-123", raw_text="Help, I have an issue.")
    print("TriageInput validation successful:\n", input_data, "\n")

    # 2. Test valid TriageDecision
    valid_decision = TriageDecision(
        category="billing",
        priority="P1",
        summary="User is unable to process payment. Please assist.",
        suggested_action="Escalate to billing team.",
        needs_human=True,
        confidence=0.95
    )
    print("Valid TriageDecision validation successful:\n", valid_decision, "\n")

    # 3. Test invalid confidence (> 1.0)
    try:
        TriageDecision(
            category="billing",
            priority="P1",
            summary="User is unable to process payment. Please assist.",
            suggested_action="Escalate to billing team.",
            needs_human=True,
            confidence=1.5
        )
        print("ERROR: Confidence > 1.0 did not raise ValidationError")
    except ValidationError as e:
        print("Expected ValidationError raised for confidence > 1.0:\n", e, "\n")

    # 4. Test invalid summary (> 2 sentences)
    try:
        TriageDecision(
            category="billing",
            priority="P1",
            summary="This is sentence one. This is sentence two. This is sentence three.",
            suggested_action="Escalate to billing team.",
            needs_human=True,
            confidence=0.8
        )
        print("ERROR: Summary with > 2 sentences did not raise ValidationError")
    except ValidationError as e:
        print("Expected ValidationError raised for summary > 2 sentences:\n", e, "\n")

if __name__ == "__main__":
    test_models()
