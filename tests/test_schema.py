import pytest
from PIL import Image

from data.schema import (
    CHOICE_KEYS,
    MRAGBenchInstance,
    build_deduped_corpus,
    hash_image_bytes,
)


def _make_image(color, size=(16, 16)):
    return Image.new("RGB", size, color=color)


def _raw_row(**overrides):
    row = {
        "id": "q1",
        "aspect": "angle",
        "scenario": "Animal",
        "image": _make_image((255, 0, 0)),
        "gt_images": [_make_image((0, 255, 0))],
        "question": "Can you identify this animal?",
        "A": "Cat",
        "B": "Dog",
        "C": "Bird",
        "D": "Fish",
        "answer_choice": "B",
        "answer": "Dog",
        "image_type": "cropped",
        "source": "some_dataset",
        "retrieved_images": [],
    }
    row.update(overrides)
    return row


def test_from_hf_row_parses_confirmed_columns():
    inst = MRAGBenchInstance.from_hf_row(_raw_row())
    assert inst.id == "q1"
    assert inst.answer_choice == "B"
    assert inst.choices == {"A": "Cat", "B": "Dog", "C": "Bird", "D": "Fish"}
    assert set(CHOICE_KEYS) == {"A", "B", "C", "D"}


def test_from_hf_row_ignores_answer_and_retrieved_images_fields():
    # answer (free text) and retrieved_images (MRAG-Bench's own retrieval)
    # must NOT end up anywhere on the instance -- this project builds its own.
    inst = MRAGBenchInstance.from_hf_row(_raw_row())
    assert not hasattr(inst, "answer")
    assert not hasattr(inst, "retrieved_images")


def test_from_hf_row_missing_column_raises():
    row = _raw_row()
    del row["answer_choice"]
    with pytest.raises(KeyError):
        MRAGBenchInstance.from_hf_row(row)


def test_from_hf_row_missing_choice_column_raises():
    row = _raw_row()
    del row["C"]
    with pytest.raises(KeyError):
        MRAGBenchInstance.from_hf_row(row)


def test_caption_text_for_bm25_is_question():
    inst = MRAGBenchInstance.from_hf_row(_raw_row(question="What breed is shown?"))
    assert inst.caption_text_for_bm25() == "What breed is shown?"


def test_hash_image_bytes_is_stable_for_identical_content():
    img1 = _make_image((10, 20, 30))
    img2 = _make_image((10, 20, 30))
    assert hash_image_bytes(img1) == hash_image_bytes(img2)


def test_hash_image_bytes_differs_for_different_content():
    img1 = _make_image((10, 20, 30))
    img2 = _make_image((30, 20, 10))
    assert hash_image_bytes(img1) != hash_image_bytes(img2)


def test_build_deduped_corpus_merges_shared_gt_images():
    shared_evidence = _make_image((5, 5, 5))
    unique_evidence_1 = _make_image((1, 1, 1))
    unique_evidence_2 = _make_image((2, 2, 2))

    inst1 = MRAGBenchInstance.from_hf_row(
        _raw_row(id="q1", gt_images=[shared_evidence, unique_evidence_1])
    )
    inst2 = MRAGBenchInstance.from_hf_row(
        _raw_row(id="q2", gt_images=[shared_evidence, unique_evidence_2])
    )

    result = build_deduped_corpus([inst1, inst2])
    corpus = result["corpus"]
    instance_gt_hashes = result["instance_gt_hashes"]

    # 3 distinct images total (shared + 2 unique), not 4
    assert len(corpus) == 3

    # both instances' gt hash lists should include the shared image's hash
    shared_hash = hash_image_bytes(shared_evidence)
    assert shared_hash in instance_gt_hashes["q1"]
    assert shared_hash in instance_gt_hashes["q2"]
    assert len(instance_gt_hashes["q1"]) == 2
    assert len(instance_gt_hashes["q2"]) == 2


def test_build_deduped_corpus_handles_instance_with_no_gt_images():
    inst = MRAGBenchInstance.from_hf_row(_raw_row(id="q1", gt_images=[]))
    result = build_deduped_corpus([inst])
    assert result["corpus"] == {}
    assert result["instance_gt_hashes"]["q1"] == []
