"""
Credit: laksjdjf
https://github.com/laksjdjf/cgem156-ComfyUI/blob/main/scripts/attention_couple/node.py
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn.functional import interpolate

from .debug import ForgeCoupleDebug


def repeat_div(value: int, iterations: int) -> int:
    for _ in range(iterations):
        value = math.ceil(value / 2)

    return value


def _attention_size(num_tokens: int, shape: tuple[int, ...]) -> tuple[int, int]:
    width, height = shape[3], shape[2]
    scale = math.ceil(math.log2(math.sqrt(height * width / num_tokens)))
    return repeat_div(height, scale), repeat_div(width, scale)


def get_mask(mask: torch.Tensor, batch_size: int, num_tokens: int, shape: tuple[int, ...]):
    """
    Credit: hako-mikan
    https://github.com/hako-mikan/sd-webui-regional-prompter/blob/main/scripts/attention.py

    Issue Found/Fixed by. arcusmaximus & www
    https://github.com/arcusmaximus/sd-forge-couple/tree/draggable-box-ui
    """

    size = _attention_size(num_tokens, shape)

    num_conds = mask.shape[0]
    mask_downsample = interpolate(mask, size=size, mode="nearest")
    mask_downsample = mask_downsample.view(num_conds, num_tokens, 1)

    return mask_downsample.repeat_interleave(batch_size, dim=0)


def _bbox_coords(bbox: Any) -> tuple[int, int, int, int]:
    try:
        return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    except Exception:
        pass
    if all(hasattr(bbox, key) for key in ("x", "y", "w", "h")):
        x1, y1 = int(bbox.x), int(bbox.y)
        return x1, y1, x1 + int(bbox.w), y1 + int(bbox.h)
    if all(hasattr(bbox, key) for key in ("x1", "y1", "x2", "y2")):
        return int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    raise TypeError(f"Unsupported MultiDiffusion bbox object: {bbox!r}")


def _empty_crop(mask: torch.Tensor) -> torch.Tensor:
    return torch.zeros((1, 1, 1, 1), device=mask.device, dtype=mask.dtype)


def _downsample_tile_mask(mask_crop: torch.Tensor, num_tokens: int, shape: tuple[int, ...]) -> torch.Tensor:
    size = _attention_size(num_tokens, shape)
    mask_downsample = interpolate(mask_crop, size=size, mode="nearest")
    return mask_downsample.view(1, num_tokens, 1)


def get_multidiffusion_mask(
    mask: torch.Tensor,
    batch_size: int,
    num_tokens: int,
    shape: tuple[int, ...],
    ctx: Any,
) -> torch.Tensor:
    """Return attention masks cropped to active MultiDiffusion tile bboxes.

    Output order intentionally matches get_mask(): condition-major, with each
    condition repeated across the effective batch. Under MultiDiffusion the
    effective batch is ordered by tile, so each condition emits masks in
    bbox order and repeats each bbox crop by the per-tile batch count.
    """

    tile_count = len(getattr(ctx, "bboxes", []) or [])
    if tile_count <= 0:
        ForgeCoupleDebug.incr("fallbacks_hit")
        ForgeCoupleDebug.log("mask", "falling back to legacy mask", reason="no_bboxes")
        return get_mask(mask, batch_size, num_tokens, shape)

    if batch_size % tile_count != 0:
        ForgeCoupleDebug.incr("fallbacks_hit")
        ForgeCoupleDebug.log(
            "mask",
            "falling back to legacy mask",
            reason="batch_not_divisible_by_tile_count",
            batch_size=batch_size,
            tile_count=tile_count,
        )
        return get_mask(mask, batch_size, num_tokens, shape)

    if mask.ndim != 4:
        raise ValueError(f"Expected mask shape [num_conds, 1, H, W], got {tuple(mask.shape)}")

    num_conds, _, mask_h, mask_w = mask.shape
    full_latent_h = int(ctx.full_latent_h)
    full_latent_w = int(ctx.full_latent_w)
    if full_latent_h <= 0 or full_latent_w <= 0:
        ForgeCoupleDebug.incr("fallbacks_hit")
        ForgeCoupleDebug.log(
            "mask",
            "falling back to legacy mask",
            reason="invalid_full_latent_shape",
            full_latent_h=full_latent_h,
            full_latent_w=full_latent_w,
        )
        return get_mask(mask, batch_size, num_tokens, shape)

    per_tile_batch = batch_size // tile_count
    sx = mask_w / full_latent_w
    sy = mask_h / full_latent_h
    target_size = _attention_size(num_tokens, shape)

    rows: list[torch.Tensor] = []
    for cond_i in range(num_conds):
        cond_mask = mask[cond_i : cond_i + 1]
        for bbox_i, bbox in enumerate(ctx.bboxes):
            x1, y1, x2, y2 = _bbox_coords(bbox)
            px1 = max(0, min(mask_w, round(x1 * sx)))
            px2 = max(0, min(mask_w, round(x2 * sx)))
            py1 = max(0, min(mask_h, round(y1 * sy)))
            py2 = max(0, min(mask_h, round(y2 * sy)))

            if px2 <= px1 or py2 <= py1:
                ForgeCoupleDebug.incr("fallbacks_hit")
                ForgeCoupleDebug.log(
                    "mask",
                    "degenerate multidiffusion mask crop",
                    cond=cond_i,
                    bbox=bbox_i,
                    bbox_coords=(x1, y1, x2, y2),
                    pixel_coords=(px1, py1, px2, py2),
                )
                crop = _empty_crop(mask)
            else:
                crop = cond_mask[:, :, py1:py2, px1:px2]

            ForgeCoupleDebug.incr("mask_crops_applied")
            ForgeCoupleDebug.log(
                "mask",
                "cropped multidiffusion mask",
                cond=cond_i,
                bbox=bbox_i,
                bbox_coords=(x1, y1, x2, y2),
                pixel_coords=(px1, py1, px2, py2),
                crop_shape=tuple(crop.shape),
                downsample_target=target_size,
                repeat=per_tile_batch,
            )
            ForgeCoupleDebug.dump_mask(f"cond{cond_i}_bbox{bbox_i}_crop", crop)
            rows.append(_downsample_tile_mask(crop, num_tokens, shape).repeat_interleave(per_tile_batch, dim=0))

    return torch.cat(rows, dim=0)


def get_dit_mask(mask: torch.Tensor, seq_len: int, w: int, h: int, patch_size: int = 2):
    """Dynamically resizes and flattens the 2D mask to match the DiT's sequence length"""

    num_conds = mask.shape[0]

    h_p = h // (8 * patch_size)
    w_p = w // (8 * patch_size)
    t_p = max(seq_len // (h_p * w_p), 1)

    mask_for_interp = mask.view(num_conds, 1, mask.shape[-2], mask.shape[-1])

    mask_resized = interpolate(
        mask_for_interp, size=(h_p, w_p), mode="bilinear", align_corners=False
    )

    mask_flattened = mask_resized.view(num_conds, 1, h_p * w_p)
    mask_flattened = mask_flattened.repeat(1, t_p, 1)

    return mask_flattened.view(num_conds, 1, seq_len, 1)


def lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def lcm_for_list(numbers: list[int]) -> int:
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = lcm(current_lcm, number)
    return current_lcm
