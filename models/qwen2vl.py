"""Qwen2-VL-7B-Instruct wrapper. Native transformers architecture (no
trust_remote_code needed for Qwen2-VL itself).

NOT testable in this sandbox (no GPU, no network to huggingface.co). The
prompt-building and answer-extraction logic this depends on (models/base.py)
IS fully tested. Run tests/test_qwen2vl_smoke.py on Colab to confirm the
actual generation call before trusting this for a real eval run.
"""
from __future__ import annotations

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from data.schema import MRAGBenchInstance
from models.base import build_prompt, extract_choice_letter


class Qwen2VLModel:
    def __init__(
        self,
        hf_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 64,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        # NOTE: dtype=, not the deprecated torch_dtype=
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_id, dtype=self.dtype
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)

    @torch.no_grad()
    def answer(self, instance: MRAGBenchInstance, retrieved_images: list) -> str | None:
        images = [instance.image, *retrieved_images]
        prompt_text = build_prompt(instance, num_retrieved_images=len(retrieved_images))

        # Qwen2-VL chat template expects a content list interleaving image
        # placeholders and text -- one {"type": "image"} entry per image, in
        # the same order as `images`, matching build_prompt's numbering.
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt_text}]
        messages = [{"role": "user", "content": content}]

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[chat_text], images=images, return_tensors="pt").to(
            self.device
        )
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        # strip the prompt tokens off the front of the generated output
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return extract_choice_letter(output_text)
