"""Optional real Transformers API check using tiny random local weights, on CPU."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mrrate_vlm.config import Config
from mrrate_vlm.embed import MedSiglipEncoder, feature_tensor


def test_actual_siglip_feature_api_without_downloads(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    import huggingface_hub

    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "synthetic": 3, "brain": 4}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="$A [EOS]", special_tokens=[("[EOS]", 2)]
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    image_processor = transformers.SiglipImageProcessor(size={"height": 16, "width": 16})

    class Processor:
        def __call__(self, **kwargs):
            return image_processor(**kwargs)

    processor = Processor()
    processor.tokenizer = tokenizer
    model = transformers.SiglipModel(
        transformers.SiglipConfig(
            text_config={
                "vocab_size": 8,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "max_position_embeddings": 64,
                "projection_size": 32,
            },
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "image_size": 16,
                "patch_size": 8,
            },
        )
    )
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(
            repo_info=lambda **kw: SimpleNamespace(sha="local-random-api-test")
        ),
    )
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda **kw: processor)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda **kw: model)
    encoder = MedSiglipEncoder(Config(data_dir=str(tmp_path), device="cpu", batch_size=2))
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    images = [Image.new("RGB", (24, 26), (30 + i, 90, 110)) for i in range(3)]
    visual = encoder.encode_images(images)
    text, audit = encoder.encode_texts(["synthetic brain", "synthetic " * 80])
    assert visual.shape == (3, 32)
    assert text.shape == (2, 32)
    np.testing.assert_allclose(np.linalg.norm(visual, axis=1), 1, rtol=1e-6)
    assert audit[1]["truncated"]
    assert audit[1]["used_tokens"] == 64
    with torch.inference_mode():
        output = model.get_image_features(pixel_values=torch.ones(1, 3, 16, 16))
    assert feature_tensor(output).shape == (1, 32)
