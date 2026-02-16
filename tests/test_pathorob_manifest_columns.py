from pathlib import Path

import pandas as pd
import pytest

from ijepath.eval.pathorob.datasets import load_manifest


def test_load_manifest_requires_sample_id(tmp_path: Path):
    manifest = pd.DataFrame(
        [
            {
                "image_path": "/tmp/a.png",
                "label": "normal",
                "medical_center": "RUMC",
                "slide_id": "s1",
            }
        ]
    )
    csv_path = tmp_path / "manifest.csv"
    manifest.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="sample_id"):
        load_manifest(str(csv_path), dataset_name="camelyon")
