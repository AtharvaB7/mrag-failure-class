"""Real SigLIP image encoder implementing the `ImageEncoder` Protocol from
retrieval/dense.py. Loaded in fp16, batched, handles the transformers
version drift where `get_image_features()` returns either a raw tensor or a
`BaseModelOutputWithPooling` (needs `.pooler_output`).

NOT testable in this sandbox (no GPU, no network to huggingface.co). The
interface contract (encode_images(images, batch_size) -> np.ndarray of shape
(N, D)) is exactly what's already tested against a fake encoder in
tests/test_dense.py -- this file just needs to satisfy that contract for
real. Run tests/test_siglip_encoder_smoke.py on your Colab GPU to confirm
before using this inside DenseRetriever for real.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


class SiglipImageEncoder:
    def __init__(
        self,
        hf_id: str = "google/siglip-base-patch16-224",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        # NOTE: use dtype=, not the deprecated torch_dtype=
        self.model = AutoModel.from_pretrained(hf_id, dtype=self.dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)

    @torch.no_grad()
    def encode_images(self, images, batch_size: int = 32) -> np.ndarray:
        all_embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            inputs = {k: v.to(self.dtype) if v.dtype.is_floating_point else v for k, v in inputs.items()}
            output = self.model.get_image_features(**inputs)
            # transformers API drift: some versions return a raw tensor,
            # others return BaseModelOutputWithPooling (needs .pooler_output)
            if hasattr(output, "pooler_output"):
                emb = output.pooler_output
            else:
                emb = output
            all_embs.append(emb.float().cpu().numpy())
        return np.concatenate(all_embs, axis=0)
