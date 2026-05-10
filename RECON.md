# Recon: MultiDiffusion-aware Forge Couple masks

## Workspace / fork status

`gh` is not available in the execution container (`bash: gh: command not found`). The authenticated GitHub connector is available as user `Neb963`, and the fork `Neb963/sd-forge-couple` already exists. I created branch `feat/multidiffusion-tile-aware-masks` from `main` at commit `1eb253b346b71ec983e9cee6b093db73a9f65607`.

Local `git clone` could not be used because the execution container cannot resolve `github.com`. Repository writes are therefore performed through the GitHub connector.

## `lib_couple/attention_couple.py`

Relevant source: `lib_couple/attention_couple.py`.

- Lines 13-14 import the legacy mask path: `from .attention_masks import get_mask, lcm_for_list`.
- Lines 21-27 build `num_conds` from kwargs, then stack `base_mask` and all `mask_i` tensors into a full-canvas tensor: `mask = torch.stack(mask, dim=0).to(device=device, dtype=dtype)`.
- Lines 29-33 validate full coverage and normalize all region masks by their per-pixel sum. This should stay global.
- Lines 35-39 build regional prompt embeddings from `cond_i`. These embeddings are not spatial and should stay global.
- Lines 42-77 define `attn2_patch`. `cond_or_uncond = extra_options["cond_or_uncond"]`; `cls.batch_size = q.shape[0] // num_chunks`. For unconditional chunks (`cond_or_uncond == 1`), the original q/k are kept. For positive chunks, q is repeated `num_conds` times and regional conditioning is appended in condition-major order.
- Lines 80-99 define `attn2_output_patch`. The legacy code calls `get_mask(mask, cls.batch_size, out.shape[1], extra_options["original_shape"])` without any tile metadata. This is the intervention point. `cond_or_uncond` handling must remain unchanged: unconditional chunks pass through; positive chunks multiply `out[pos : pos + num_conds * cls.batch_size]` by `mask_downsample`, reshape to `(num_conds, cls.batch_size, tokens, channels)`, then sum over conditions.

## `lib_couple/attention_masks.py`

Relevant source: `lib_couple/attention_masks.py`.

- Lines 10-14 define `repeat_div(value, iterations)` as repeated ceil division by 2.
- Lines 17-35 define `get_mask(mask, batch_size, num_tokens, shape)`. It reads `width, height = shape[3], shape[2]`, derives a downsample scale from `height * width / num_tokens`, interpolates the full mask to the attention spatial size using nearest-neighbor, flattens to `[num_conds, num_tokens, 1]`, then repeats each condition by `batch_size` along dim 0.
- Existing output order is condition-major: cond0 repeated across effective batch, cond1 repeated across effective batch, etc. The MultiDiffusion-aware path must preserve that order.
- Lines 38-58 define `get_dit_mask`, unrelated to SD1/SDXL cross-attention mask cropping.
- Lines 61-68 define LCM helpers used by `attn2_patch`.

## `lib_couple/mapping.py`

Relevant source: `lib_couple/mapping.py`.

- Lines 15-24 define `text2cond`, which builds prompt embeddings from full prompt texts. This remains global.
- Lines 27-75 define `basic_mapping(sd_model, couples, width, height, ...)`. Prompt embeddings are produced with `SdConditioning([couples[tile]], False, width, height, None)`, and masks are created in full image coordinates `(height, width)`.
- Lines 78-109 define `advanced_mapping`. Coordinates are normalized full-canvas values converted by `x_from = int(width * x1)`, `y_from = int(height * y1)`, etc.
- Lines 126-169 define `mask_mapping`, which resizes manually drawn masks to `(width, height)`.
- Conclusion: mappings and embeddings are full-canvas by design. Only the final spatial mask application should become tile-aware.

## `lib_couple/tile_funcs.py`

Relevant source: `lib_couple/tile_funcs.py`.

- Lines 9-10 define `SIZE = 1024` for overlap calculation, independent of actual latent tiling.
- Lines 13-36 calculate which region masks overlap an external tile.
- Lines 38-69 prepare 1024-square binary masks for Basic / Advanced / Mask modes.
- Lines 86-145 build `self.tiles`, a list of per-external-tile prompt strings.
- This is not MultiDiffusion Integrated. It is prompt-per-tile logic intended for external tilers such as SD Upscale. This patch does not extend or modify it.

## Forge Neo `tiled_diffusion.py`

Relevant source inspected: `Haoming02/sd-webui-forge-classic`, branch `neo`, file `extensions-builtin/sd_forge_multidiffusion/lib_multidiffusion/tiled_diffusion.py`.

Important source facts:

- `BBox.__init__` stores latent-space `x`, `y`, `w`, `h`, `box = [x, y, x + w, y + h]`, and `slicer = slice(None), slice(None), slice(y, y + h), slice(x, x + w)`.
- `BBox.__getitem__` returns entries from `box`, so `bbox[0:4]` semantics are `x1, y1, x2, y2`.
- In `MultiDiffusion.__call__`, after optional 5D squeeze, `N, C, H, W = x_in.shape`; within `for batch_id, bboxes in enumerate(self.batched_bboxes):`, `x_tile`, `ts_tile`, and `c_tile` are built. The exact call site is `x_tile_out = model_function(x_tile, ts_tile, **c_tile)`. At this point `bboxes`, `H`, and `W` are in scope.
- In `MixtureOfDiffusers.__call__`, after optional 5D squeeze, `N, C, H, W = x_in.shape`; within `for batch_id, bboxes in enumerate(self.batched_bboxes):`, `x_tile`, `t_tile`, and `c_tile` are built. The exact call site is `x_tile_out = model_function(x_tile, t_tile, **c_tile)`. At this point `bboxes`, `H`, and `W` are in scope.
- ControlNet hints are explicitly cropped per bbox in `process_controlnet`, using `bbox[1] * opt_f : bbox[3] * opt_f` and `bbox[0] * opt_f : bbox[2] * opt_f`. This confirms the intended tile-aware spatial behavior.

## Implementation decision

The Forge Couple repository can implement the context holder and tile-aware mask cropper, but the active MultiDiffusion tile bbox must be set by Forge Neo's MultiDiffusion wrapper. Because `tiled_diffusion.py` lives in a different repository, this branch includes an installable patch file for that external source rather than pretending the fork can directly modify upstream Forge Neo.
