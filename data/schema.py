"""MRAG-Bench schema, confirmed by an actual `load_dataset(...).column_names`
call (uclanlp/MRAG-Bench, split="test"). Do not modify these field names
without re-confirming against a real load_dataset() call.

Columns: ['id', 'aspect', 'scenario', 'image', 'gt_images', 'question',
          'A', 'B', 'C', 'D', 'answer_choice', 'answer', 'image_type',
          'source', 'retrieved_images']

Key facts baked into this module:
- `image`: the query image (PIL) — the actual subject of the question. Must
  be shown to the VLM in EVERY setting, including no-retrieval.
- `gt_images`: ground-truth evidence images, no stable IDs — corpus is built
  here via dedup by hashing encoded image bytes (same evidence image can
  support multiple questions).
- `answer_choice`: the letter (A/B/C/D) — this is the grading ground truth,
  NOT the free-text `answer` column.
- `retrieved_images`: MRAG-Bench's own precomputed retrieval. NOT used by
  this project — we build and compare our own retrieval settings.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any


CHOICE_KEYS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class MRAGBenchInstance:
    id: str
    aspect: str
    scenario: str
    image: Any  # PIL.Image, the query image
    gt_images: list[Any]  # list of PIL.Image, evidence images
    question: str
    choices: dict[str, str]  # {"A": ..., "B": ..., "C": ..., "D": ...}
    answer_choice: str  # ground truth letter, e.g. "B"

    @classmethod
    def from_hf_row(cls, row: dict[str, Any]) -> "MRAGBenchInstance":
        """Build from a raw HF datasets row dict. Deliberately ignores
        `answer` (free text) and `retrieved_images` (not used by this
        project) per the confirmed-schema notes above.
        """
        missing = [
            k
            for k in ("id", "aspect", "scenario", "image", "gt_images", "question", "answer_choice")
            for k in [k]
            if k not in row
        ]
        missing += [c for c in CHOICE_KEYS if c not in row]
        if missing:
            raise KeyError(f"Row missing expected MRAG-Bench columns: {sorted(set(missing))}")
        return cls(
            id=str(row["id"]),
            aspect=row["aspect"],
            scenario=row["scenario"],
            image=row["image"],
            gt_images=list(row["gt_images"]),
            question=row["question"],
            choices={k: row[k] for k in CHOICE_KEYS},
            answer_choice=row["answer_choice"],
        )

    def caption_text_for_bm25(self) -> str:
        """Text surrogate used for sparse retrieval indexing of THIS instance's
        own query, i.e. what we query BM25 with — just the question text.
        (Distinct from corpus doc captions, which come from wherever the
        evidence-image caption/metadata pipeline sources them.)
        """
        return self.question


def hash_image_bytes(image: Any) -> str:
    """Stable content hash for a PIL Image, used to dedup gt_images across
    instances into a single corpus (gt_images has no stable IDs, and the
    same evidence image can back multiple questions).

    Uses the raw encoded PNG bytes so that pixel-identical images always
    hash identically regardless of any transient object identity.
    """
    buf = io.BytesIO()
    # Convert to a consistent mode before saving so that e.g. an RGB and an
    # RGBA copy of the "same" image (as far as dedup is concerned) don't
    # silently diverge -- we normalize to RGB.
    image.convert("RGB").save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def build_deduped_corpus(instances: list[MRAGBenchInstance]) -> dict[str, Any]:
    """Build a deduplicated corpus of {content_hash: PIL.Image} from every
    gt_images entry across all instances, plus a mapping of
    {instance_id: [content_hash, ...]} recording which corpus images are
    ground-truth evidence for which question (used later for retrieval
    recall scoring).
    """
    corpus: dict[str, Any] = {}
    instance_gt_hashes: dict[str, list[str]] = {}
    for inst in instances:
        hashes = []
        for img in inst.gt_images:
            h = hash_image_bytes(img)
            corpus.setdefault(h, img)
            hashes.append(h)
        instance_gt_hashes[inst.id] = hashes
    return {"corpus": corpus, "instance_gt_hashes": instance_gt_hashes}
