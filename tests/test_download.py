import json
from pathlib import Path

import pytest
from PIL import Image

from data.download import _write_instances_and_corpus
from data.schema import CHOICE_KEYS


class FakeHFDataset(list):
    """Minimal stand-in for a HF `datasets.Dataset`: iterable of row dicts,
    with a `column_names` attribute. Enough to test the file-writing logic
    without any network access.
    """

    def __init__(self, rows):
        super().__init__(rows)
        self.column_names = list(rows[0].keys()) if rows else []


def _row(id_, gt_images, question="Q?"):
    return {
        "id": id_,
        "aspect": "angle",
        "scenario": "Animal",
        "image": Image.new("RGB", (8, 8), color=(id_.__hash__() % 255, 0, 0)),
        "gt_images": gt_images,
        "question": question,
        "A": "a", "B": "b", "C": "c", "D": "d",
        "answer_choice": "A",
        "answer": "a",
        "image_type": "cropped",
        "source": "src",
        "retrieved_images": [],
    }


def test_write_instances_and_corpus_dedupes_shared_gt_images(tmp_path):
    shared = Image.new("RGB", (8, 8), color=(9, 9, 9))
    unique1 = Image.new("RGB", (8, 8), color=(1, 1, 1))
    unique2 = Image.new("RGB", (8, 8), color=(2, 2, 2))

    ds = FakeHFDataset([
        _row("q1", [shared, unique1]),
        _row("q2", [shared, unique2]),
    ])

    stats = _write_instances_and_corpus(ds, tmp_path)

    assert stats["n_instances"] == 2
    assert stats["n_gt_images_total"] == 4
    assert stats["n_gt_images_deduped"] == 3  # shared counted once

    # default format is JPEG (faster than PNG for bulk re-encoding)
    corpus_files = list((tmp_path / "corpus").glob("*.jpg"))
    assert len(corpus_files) == 3

    query_files = list((tmp_path / "query_images").glob("*.jpg"))
    assert {f.stem for f in query_files} == {"q1", "q2"}

    instances_path = tmp_path / "instances.jsonl"
    lines = instances_path.read_text().strip().split("\n")
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert rec1["id"] == "q1"
    assert rec1["query_image_path"] == "query_images/q1.jpg"
    assert len(rec1["gt_image_hashes"]) == 2
    assert set(CHOICE_KEYS) == {"A", "B", "C", "D"}  # sanity: taxonomy constant unchanged
    assert rec1["choices"] == {"A": "a", "B": "b", "C": "c", "D": "d"}


def test_write_instances_and_corpus_png_format_still_supported(tmp_path):
    ds = FakeHFDataset([_row("q1", [Image.new("RGB", (8, 8), (5, 5, 5))])])
    _write_instances_and_corpus(ds, tmp_path, image_format="PNG")
    assert list((tmp_path / "corpus").glob("*.png"))
    assert list((tmp_path / "query_images").glob("*.png"))
    assert not list((tmp_path / "corpus").glob("*.jpg"))


def test_write_instances_and_corpus_empty_gt_images(tmp_path):
    ds = FakeHFDataset([_row("q1", [])])
    stats = _write_instances_and_corpus(ds, tmp_path)
    assert stats["n_gt_images_total"] == 0
    assert stats["n_gt_images_deduped"] == 0
    corpus_files = list((tmp_path / "corpus").glob("*.png"))
    assert corpus_files == []


def test_write_instances_and_corpus_row_missing_column_raises(tmp_path):
    ds = FakeHFDataset([_row("q1", [])])
    del ds[0]["answer_choice"]
    with pytest.raises(KeyError):
        _write_instances_and_corpus(ds, tmp_path)
