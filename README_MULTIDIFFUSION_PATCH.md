# MultiDiffusion compatibility patch

## Problem statement

Vanilla Forge Couple builds full-image regional attention masks, then resizes those full masks to the current attention map. Under MultiDiffusion Integrated, the UNet is invoked on latent tiles. Without the tile bbox, Forge Couple cannot know which global image area the current attention map represents, so regional masks can be applied as if each tile were the whole canvas.

## Architecture

```text
Forge Neo MultiDiffusion / MixtureOfDiffusers
  for each latent tile batch:
    bboxes + full latent H/W
      -> contextvars bridge
        -> Forge Couple attn2_output_patch
          -> crop full-canvas mask by current bbox
          -> downsample cropped mask to attention-token grid
          -> apply existing attention-couple blending
```

Prompt embeddings remain global. Only spatial masks become tile-aware.

## Files changed

- `lib_couple/attention_masks.py` — adds `get_multidiffusion_mask(...)`, bbox parsing, tile crop/downsample logic, and fallback handling.
- `lib_couple/attention_couple.py` — detects active MultiDiffusion context and routes mask generation through the tile-aware path.
- `lib_couple/settings.py` — adds debug logging and mask dump options.
- `scripts/forge_couple.py` — wires debug settings into sampling and removes raw debug `print()` usage.
- `README.md` — points to this document.

## New files

- `lib_couple/debug.py` — structured debug logger, mask dumper, and counters.
- `lib_couple/multidiffusion_context.py` — `contextvars` bridge used by the external MultiDiffusion wrapper.
- `patches/forge_neo_tiled_diffusion_context.patch` — patch for Forge Neo's built-in MultiDiffusion file.
- `tests/` — pytest tests for context isolation, mask crop math, fallback behavior, no-MD passthrough, and edge tiles.
- `RECON.md` — source reconnaissance with actual file paths and relevant variables.

## Public API additions

- `MultiDiffusionTileContext(bboxes, full_latent_h, full_latent_w, tile_batch_count)`
- `set_md_tile_context(ctx)`
- `reset_md_tile_context(token)`
- `get_md_tile_context()`
- `get_multidiffusion_mask(mask, batch_size, num_tokens, shape, ctx)`

## Debug usage

Debug can be enabled with either:

```bash
FORGE_COUPLE_DEBUG=1 ./webui.sh
```

or the Forge Couple settings:

- Settings -> Forge Couple -> Debug logging
- Settings -> Forge Couple -> Dump masks to disk

Logs are written to:

```text
logs/forge_couple_debug.log
```

Mask dumps are written to:

```text
logs/forge_couple_masks/
```

## External Forge Neo patch

Apply the patch from the Forge Neo root:

```bash
git apply /path/to/sd-forge-couple/patches/forge_neo_tiled_diffusion_context.patch
```

The import is soft. If Forge Couple is absent, MultiDiffusion continues to run without the context bridge.

## Validation procedure

Use a deterministic test before production work:

```text
1024x1536
Mixture of Diffusers ON
Tile: 512x768
Overlap: 64
Tile batch: 1
No ControlNet
Forge Couple Advanced mode
Left half: red clothing, red fabric
Right half: blue clothing, blue fabric
Fixed seed
```

| Case | Expected result |
|---|---|
| Forge Couple only | left/right split works |
| MultiDiffusion only | no regional split |
| MultiDiffusion + old Forge Couple | possible leakage or wrong split under tiled attention |
| MultiDiffusion + patched Forge Couple | split follows global left/right regions per tile |

Then add Depth, OpenPose, and masked Reference one at a time.

## Known limitations

- The Forge Neo MultiDiffusion source lives outside this repository; this branch ships a patch file rather than directly modifying Forge Neo.
- Interaction with masked Reference, OpenPose, and Depth ControlNet stacked together still needs real image-generation validation.
- Anima / DiT mask code is not made MultiDiffusion-aware here; this patch targets the SD1/SDXL attention-couple path.
- If Forge Neo later changes `tiled_diffusion.py`, the patch may need rebasing.

## Rollback

In this repository:

```bash
git checkout main
git branch -D feat/multidiffusion-tile-aware-masks
```

For Forge Neo after applying the external patch:

```bash
git checkout -- extensions-builtin/sd_forge_multidiffusion/lib_multidiffusion/tiled_diffusion.py
```

or revert the commit that applied the patch.

## Upstream-clean version note

The cleaner upstream design is for Forge's model patcher to pass tile metadata through `extra_options`, for example `extra_options["tile_bbox"]` and `extra_options["full_latent_shape"]`. That would avoid cross-extension context state. This branch uses `contextvars` because it is smaller, soft-importable, and does not require changing Forge's model-patcher API.
