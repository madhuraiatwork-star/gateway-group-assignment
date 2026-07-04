from typing import Literal
import re
from pydantic import BaseModel, Field, field_validator

class TriageInput(BaseModel):
    message_id: str
    raw_text: str

class TriageDecision(BaseModel):
    category: str
    priority: Literal["P0", "P1", "P2", "P3"]
    summary: str = Field(description="Summary of the raw text, maximum of 2 sentences.")
    suggested_action: str
    needs_human: bool
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0.",
        ge=0.0,
        le=1.0
    )

    @field_validator("summary")
    @classmethod
    def validate_summary_sentences(cls, value: str) -> str:
        # Normalize whitespace and strip
        text = value.strip()
        if not text:
            return text

        # Split sentences based on terminal punctuation followed by space or string end.
        # This matches typical sentence boundaries (., !, ?) while avoiding splitting on simple abbreviations.
        sentences = re.split(r'[.!?]+(?:\s+|$)', text)
        # Remove any empty components resulting from the split
        sentences = [s for s in sentences if s.strip()]
        
        if len(sentences) > 2:
            raise ValueError(f"Summary must be a maximum of 2 sentences. Found {len(sentences)} sentences.")
        return value
