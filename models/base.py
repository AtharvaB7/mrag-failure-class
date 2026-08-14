"""Common interface all VLM wrappers implement, plus the model-agnostic
prompt-building and answer-letter-extraction logic. These two pieces are
fully testable without any real model (they're pure string processing), so
they're separated from the actual weight-loading/generation code in
qwen2vl.py / llava_next.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from data.schema import CHOICE_KEYS, MRAGBenchInstance


class VLM(Protocol):
    def answer(self, instance: MRAGBenchInstance, retrieved_images: list) -> str:
        """Return a single choice letter (A/B/C/D)."""
        ...


def build_prompt(instance: MRAGBenchInstance, num_retrieved_images: int) -> str:
    """Build the text prompt for a VLM call.

    Layout: the query image is ALWAYS shown first (it's the actual subject
    of the question, per the schema notes -- must be present in every
    setting including no-retrieval), followed by any retrieved evidence
    images, followed by the question and lettered choices.

    num_retrieved_images == 0 covers the no-retrieval setting.
    """
    lines = []
    lines.append("Image 1 (query image): the image the question is about.")
    for i in range(num_retrieved_images):
        lines.append(f"Image {i + 2} (retrieved evidence): additional reference image.")
    lines.append("")
    lines.append(f"Question: {instance.question}")
    for key in CHOICE_KEYS:
        lines.append(f"{key}. {instance.choices[key]}")
    lines.append("")
    lines.append(
        "Answer with a single letter (A, B, C, or D) corresponding to the "
        "correct choice. Do not include any other text."
    )
    return "\n".join(lines)


_LETTER_RE = re.compile(r"\b([ABCD])\b")


def extract_choice_letter(model_output: str) -> str | None:
    """Extract a single A/B/C/D choice letter from raw model text output.

    Handles common formats: "A", "A.", "The answer is A", "**A**", etc.
    Returns None if no clear single letter can be extracted (caller should
    treat this as an incorrect/undetermined prediction, never silently
    default to a specific letter).
    """
    if not model_output:
        return None
    stripped = model_output.strip()

    # Fast path: output is exactly one of the four letters (most common case
    # given the prompt explicitly asks for just the letter).
    if stripped.upper() in CHOICE_KEYS:
        return stripped.upper()

    matches = _LETTER_RE.findall(stripped.upper())
    if not matches:
        return None
    # If multiple distinct letters appear (e.g. model rambled and mentioned
    # more than one choice), we can't confidently pick one -- return None
    # rather than guessing, so this gets correctly counted as a failure
    # rather than silently attributed to a possibly-wrong letter.
    unique = set(matches)
    if len(unique) > 1:
        return None
    return matches[0]


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    dtype: str = "float16"
