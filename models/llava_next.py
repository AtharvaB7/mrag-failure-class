"""LLaVA-NeXT (llava-v1.6-mistral-7b-hf) wrapper. Fully native transformers
architecture (no trust_remote_code) -- used as the secondary VLM in place of
InternVL2.5, which has a known incompatibility with current
transformers/accelerate (AttributeError on 'all_tied_weights_keys' inside
transformers' own weight-loading code, unrelated to device_map).

NOT testable in this sandbox (no GPU, no network to huggingface.co). Run
tests/test_llava_next_smoke.py on Colab before trusting this for a real
eval run.
"""
from __future__ import annotations

import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

from data.schema import MRAGBenchInstance
from models.base import build_prompt, extract_choice_letter


class LlavaNextModel:
    def __init__(
        self,
        hf_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 64,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            hf_id, dtype=self.dtype
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)

    @torch.no_grad()
    def answer(self, instance: MRAGBenchInstance, retrieved_images: list) -> str | None:
        images = [instance.image, *retrieved_images]
        prompt_text = build_prompt(instance, num_retrieved_images=len(retrieved_images))

        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt_text}]
        messages = [{"role": "user", "content": content}]

        chat_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(text=chat_text, images=images, return_tensors="pt").to(
            self.device
        )
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return extract_choice_letter(output_text)
