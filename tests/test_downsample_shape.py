import torch

from lib_couple.attention_masks import get_mask, get_multidiffusion_mask
from lib_couple.multidiffusion_context import MultiDiffusionTileContext


def test_multidiffusion_mask_output_shape_matches_legacy_contract():
    mask = torch.rand((3, 1, 96, 160))
    ctx = MultiDiffusionTileContext(
        bboxes=[(0, 0, 10, 12), (10, 0, 20, 12)],
        full_latent_h=12,
        full_latent_w=20,
        tile_batch_count=2,
    )
    out = get_multidiffusion_mask(mask, batch_size=4, num_tokens=120, shape=(4, 4, 12, 10), ctx=ctx)
    legacy = get_mask(mask, batch_size=4, num_tokens=120, shape=(4, 4, 12, 10))
    assert out.shape == legacy.shape == (12, 120, 1)
