"""Smoke test for the real SiglipImageEncoder. Auto-skips unless a GPU AND
network access to huggingface.co are both available -- i.e. it will skip in
this sandbox and should be run for real on your Colab A100.
"""
import socket

import pytest
import torch


def _hf_reachable() -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=2)
        return True
    except OSError:
        return False


requires_gpu_and_network = pytest.mark.skipif(
    not (torch.cuda.is_available() and _hf_reachable()),
    reason="requires a CUDA GPU and network access to huggingface.co (run on Colab)",
)


@requires_gpu_and_network
def test_siglip_encoder_returns_correct_shape():
    from PIL import Image

    from retrieval.encoders.siglip import SiglipImageEncoder

    encoder = SiglipImageEncoder()
    images = [Image.new("RGB", (224, 224), color=(i, i, i)) for i in range(5)]
    embs = encoder.encode_images(images, batch_size=2)
    assert embs.shape[0] == 5
    assert embs.ndim == 2


@requires_gpu_and_network
def test_siglip_encoder_deterministic_for_identical_images():
    from PIL import Image

    from retrieval.encoders.siglip import SiglipImageEncoder
    import numpy as np

    encoder = SiglipImageEncoder()
    img = Image.new("RGB", (224, 224), color=(50, 100, 150))
    embs = encoder.encode_images([img, img], batch_size=2)
    np.testing.assert_allclose(embs[0], embs[1], rtol=1e-4, atol=1e-4)
