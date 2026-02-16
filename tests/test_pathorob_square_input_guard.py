from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from ijepath.eval.plugins.pathorob import PathoROBPlugin


class _DummyTeacher:
    def eval(self):
        return self

    def __call__(self, image, masks=None):
        bsz = int(image.shape[0])
        return torch.zeros((bsz, 4, 8), dtype=image.dtype, device=image.device)


def test_pathorob_square_input_guard_raises_on_non_square_transformed_batch(tmp_path: Path):
    image_path = tmp_path / "rect.jpg"
    Image.new("RGB", (320, 256), color=(127, 127, 127)).save(image_path)

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "s0",
                "slide_id": "slide0",
                "label": "tumor",
                "medical_center": "RUMC",
                "image_path": str(image_path),
            }
        ]
    )

    plugin = PathoROBPlugin(
        cfg={
            "type": "pathorob",
            "enable": True,
            "batch_size_per_gpu": 1,
            "feature_num_workers": 0,
            "transforms": {"resize": 256, "crop_size": None, "normalize": "imagenet"},
            "enforce_square_inputs": True,
            "datasets": {"camelyon": {"enable": False}},
        },
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="square"):
        plugin._extract_features(_DummyTeacher(), manifest)
