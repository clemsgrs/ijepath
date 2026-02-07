import math
from typing import List

import torch


class ContextTargetFootprintMaskCollator:
    """Mask collator using target footprints as predictor masks and their complement as encoder masks."""

    def __init__(
        self,
        input_size: int | tuple[int, int],
        patch_size: int,
        nenc: int = 1,
        min_keep: int = 4,
    ) -> None:
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_h = int(input_size[0])
        self.input_w = int(input_size[1])
        self.patch_size = int(patch_size)
        self.nenc = int(nenc)
        self.min_keep = int(min_keep)

        self.grid_h = self.input_h // self.patch_size
        self.grid_w = self.input_w // self.patch_size
        self.num_tokens = self.grid_h * self.grid_w

    def _box_to_patch_indices(self, box_xyxy: torch.Tensor) -> torch.Tensor:
        x0, y0, x1, y1 = [float(v) for v in box_xyxy.tolist()]
        px0 = max(0, min(self.grid_w, int(math.floor(x0 / self.patch_size))))
        py0 = max(0, min(self.grid_h, int(math.floor(y0 / self.patch_size))))
        px1 = max(0, min(self.grid_w, int(math.ceil(x1 / self.patch_size))))
        py1 = max(0, min(self.grid_h, int(math.ceil(y1 / self.patch_size))))

        if px1 <= px0:
            px1 = min(self.grid_w, px0 + 1)
        if py1 <= py0:
            py1 = min(self.grid_h, py0 + 1)

        indices = []
        for py in range(py0, py1):
            base = py * self.grid_w
            for px in range(px0, px1):
                indices.append(base + px)

        if not indices:
            indices = [0]
        return torch.tensor(indices, dtype=torch.long)

    def __call__(self, batch):
        batch_size = len(batch)
        context_images = torch.stack([sample["context_image"] for sample in batch], dim=0)
        target_images = torch.stack([sample["target_images"] for sample in batch], dim=0)
        target_boxes = torch.stack([sample["target_boxes_in_context_pixels"] for sample in batch], dim=0)
        sample_metadata = [sample["sample_metadata"] for sample in batch]

        num_targets = target_boxes.shape[1]

        per_sample_pred_masks: List[List[torch.Tensor]] = []
        min_keep_pred = self.num_tokens
        for sample_idx in range(batch_size):
            sample_masks = []
            for target_idx in range(num_targets):
                indices = self._box_to_patch_indices(target_boxes[sample_idx, target_idx])
                sample_masks.append(indices)
                min_keep_pred = min(min_keep_pred, int(indices.numel()))
            per_sample_pred_masks.append(sample_masks)

        min_keep_pred = max(1, min_keep_pred)
        collated_masks_pred: List[torch.Tensor] = []
        for target_idx in range(num_targets):
            target_masks = []
            for sample_idx in range(batch_size):
                target_masks.append(per_sample_pred_masks[sample_idx][target_idx][:min_keep_pred])
            collated_masks_pred.append(torch.stack(target_masks, dim=0))

        all_indices = torch.arange(self.num_tokens, dtype=torch.long)
        per_sample_enc_masks = []
        min_keep_enc = self.num_tokens
        for sample_idx in range(batch_size):
            pred_union = torch.unique(torch.cat(per_sample_pred_masks[sample_idx], dim=0))
            keep_mask = torch.ones(self.num_tokens, dtype=torch.bool)
            keep_mask[pred_union] = False
            enc_indices = all_indices[keep_mask]
            if enc_indices.numel() < self.min_keep:
                enc_indices = all_indices[: max(self.min_keep, 1)]
            min_keep_enc = min(min_keep_enc, int(enc_indices.numel()))
            per_sample_enc_masks.append(enc_indices)

        min_keep_enc = max(self.min_keep, min_keep_enc)
        collated_masks_enc = []
        for _ in range(self.nenc):
            enc_stack = torch.stack([m[:min_keep_enc] for m in per_sample_enc_masks], dim=0)
            collated_masks_enc.append(enc_stack)

        collated_batch = {
            "context_images": context_images,
            "target_images": target_images,
            "target_boxes_in_context_pixels": target_boxes,
            "sample_metadata": sample_metadata,
        }
        return collated_batch, collated_masks_enc, collated_masks_pred
