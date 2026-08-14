import pytest

from data.schema import MRAGBenchInstance
from models.base import build_prompt, extract_choice_letter


def _instance():
    return MRAGBenchInstance(
        id="q1",
        aspect="Scope",
        scenario="Scope",
        image=object(),
        gt_images=[object(), object()],
        question="Can you identify this animal?",
        choices={"A": "silky_terrier", "B": "Yorkshire_terrier", "C": "Australian_terrier", "D": "Cairn_terrier"},
        answer_choice="A",
    )


def test_build_prompt_no_retrieval_only_mentions_query_image():
    prompt = build_prompt(_instance(), num_retrieved_images=0)
    assert "Image 1 (query image)" in prompt
    assert "Image 2" not in prompt
    assert "Can you identify this animal?" in prompt
    assert "A. silky_terrier" in prompt
    assert "D. Cairn_terrier" in prompt


def test_build_prompt_with_retrieved_images_lists_each_one():
    prompt = build_prompt(_instance(), num_retrieved_images=3)
    assert "Image 1 (query image)" in prompt
    assert "Image 2 (retrieved evidence)" in prompt
    assert "Image 3 (retrieved evidence)" in prompt
    assert "Image 4 (retrieved evidence)" in prompt
    assert "Image 5" not in prompt


def test_build_prompt_asks_for_single_letter():
    prompt = build_prompt(_instance(), num_retrieved_images=0)
    assert "single letter" in prompt.lower()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("A", "A"),
        ("a", "A"),
        (" B \n", "B"),
        ("The answer is C.", "C"),
        ("**D**", "D"),
        ("Answer: A", "A"),
        ("I believe the correct choice is B because...", "B"),
    ],
)
def test_extract_choice_letter_common_formats(raw, expected):
    assert extract_choice_letter(raw) == expected


def test_extract_choice_letter_returns_none_for_empty():
    assert extract_choice_letter("") is None
    assert extract_choice_letter(None) is None


def test_extract_choice_letter_returns_none_when_ambiguous():
    # model mentions two different letters -- must not guess
    assert extract_choice_letter("It could be A or maybe B") is None


def test_extract_choice_letter_returns_none_for_no_letter():
    assert extract_choice_letter("I'm not sure, this is unclear.") is None


def test_extract_choice_letter_repeated_same_letter_is_fine():
    # same letter mentioned twice is unambiguous, not a conflict
    assert extract_choice_letter("A. Definitely A, I'm confident.") == "A"
