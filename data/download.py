"""Download MRAG-Bench and build the deduplicated evidence-image corpus.

Writes atomically: all work happens in a temp directory, then a single
os.replace() swaps it into place, so a mid-download/mid-build crash never
leaves a corrupted/partial output directory behind.

Usage (Colab or local):
    python -m data.download --out_dir data/cache

Output layout under --out_dir:
    instances.jsonl          one JSON line per question (id, scenario, question,
                              choices, answer_choice, query_image_path, gt_image_hashes)
    corpus/<hash>.jpg         deduplicated evidence images, filename = content hash
                              (JPEG by default for encode speed; pass
                              image_format="PNG" to _write_instances_and_corpus
                              for exact fidelity if ever needed)
    query_images/<id>.jpg     query images, filename = instance id
    manifest.json             counts + column_names actually seen, for sanity checking
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from datasets import load_dataset

from data.schema import CHOICE_KEYS, MRAGBenchInstance, hash_image_bytes


def _write_instances_and_corpus(ds, tmp_dir: Path, image_format: str = "JPEG") -> dict:
    """
    image_format: "JPEG" (default) is used instead of PNG for speed -- PNG
    encoding thousands of images serially is noticeably slower and gives no
    visible progress, which can look "stuck" even when it's working. Source
    images from MRAG-Bench are already JPEGs, so re-encoding to JPEG (quality
    92) is lossless-enough for VLM input and much faster than PNG. Pass
    image_format="PNG" if exact pixel fidelity is required for some reason.
    """
    corpus_dir = tmp_dir / "corpus"
    query_dir = tmp_dir / "query_images"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    ext = "jpg" if image_format == "JPEG" else "png"
    save_kwargs = {"format": image_format}
    if image_format == "JPEG":
        save_kwargs["quality"] = 92

    seen_hashes: set[str] = set()
    n_gt_images_total = 0
    n_gt_images_deduped = 0
    n_instances = len(ds)

    instances_path = tmp_dir / "instances.jsonl"
    with open(instances_path, "w") as f:
        for idx, row in enumerate(ds):
            inst = MRAGBenchInstance.from_hf_row(row)

            # save query image
            query_path = query_dir / f"{inst.id}.{ext}"
            inst.image.convert("RGB").save(query_path, **save_kwargs)

            # save gt evidence images, deduped by content hash
            gt_hashes = []
            for img in inst.gt_images:
                h = hash_image_bytes(img)
                gt_hashes.append(h)
                n_gt_images_total += 1
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    n_gt_images_deduped += 1
                    img.convert("RGB").save(corpus_dir / f"{h}.{ext}", **save_kwargs)

            record = {
                "id": inst.id,
                "aspect": inst.aspect,
                "scenario": inst.scenario,
                "question": inst.question,
                "choices": inst.choices,
                "answer_choice": inst.answer_choice,
                "query_image_path": f"query_images/{inst.id}.{ext}",
                "gt_image_hashes": gt_hashes,
            }
            f.write(json.dumps(record) + "\n")

            if (idx + 1) % 50 == 0 or (idx + 1) == n_instances:
                print(
                    f"  [{idx + 1}/{n_instances}] instances written, "
                    f"{n_gt_images_deduped} unique corpus images so far",
                    flush=True,
                )

    return {
        "n_instances": n_instances,
        "n_gt_images_total": n_gt_images_total,
        "n_gt_images_deduped": n_gt_images_deduped,
    }


def download_mrag_bench(out_dir: str, split: str = "test", hf_dataset_id: str = "uclanlp/MRAG-Bench") -> None:
    out_path = Path(out_dir)

    print(f"Loading {hf_dataset_id} split={split} ...")
    ds = load_dataset(hf_dataset_id, split=split)
    print("column_names:", ds.column_names)

    expected_columns = {
        "id", "aspect", "scenario", "image", "gt_images", "question",
        *CHOICE_KEYS, "answer_choice", "answer", "image_type", "source",
        "retrieved_images",
    }
    actual_columns = set(ds.column_names)
    if not expected_columns.issubset(actual_columns):
        missing = expected_columns - actual_columns
        raise ValueError(
            f"Dataset schema drift detected. Missing expected columns: {sorted(missing)}. "
            f"Actual columns: {sorted(actual_columns)}. "
            "Do not proceed -- re-check data/schema.py against this output before continuing."
        )

    # do all work in a temp dir, then atomically swap into place
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        print("Writing instances + building deduped corpus ...")
        stats = _write_instances_and_corpus(ds, tmp_dir)

        manifest = {
            "hf_dataset_id": hf_dataset_id,
            "split": split,
            "column_names_seen": sorted(actual_columns),
            **stats,
        }
        with open(tmp_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        if out_path.exists():
            backup = Path(str(out_path) + ".bak")
            if backup.exists():
                shutil.rmtree(backup)
            out_path.rename(backup)
            os.replace(tmp_dir, out_path)
            shutil.rmtree(backup)
        else:
            os.replace(tmp_dir, out_path)

    print("Done. Manifest:")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Download MRAG-Bench and build deduped corpus")
    parser.add_argument("--out_dir", type=str, default="data/cache")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--hf_dataset_id", type=str, default="uclanlp/MRAG-Bench")
    args = parser.parse_args()
    download_mrag_bench(out_dir=args.out_dir, split=args.split, hf_dataset_id=args.hf_dataset_id)


if __name__ == "__main__":
    main()
