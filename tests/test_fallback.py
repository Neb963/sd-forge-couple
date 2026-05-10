import torch

from lib_couple.attention_masks import get_mask, get_multidiffusion_mask
from lib_couple.debug import ForgeCoupleDebug
from lib_couple.multidiffusion_context import MultiDiffusionTileContext


def test_batch_not_divisible_falls_back_without_raising():
    ForgeCoupleDebug.counters["fallbacks_hit"] = 0
    mask = torch.rand((2, 1, 32, 32))
    ctx = MultiDiffusionTileContext(
        bboxes=[(0, 0, 2, 2), (2, 0, 4, 2)],
        full_latent_h=4,
        full_latent_w=4,
        tile_batch_count=2,
    )
    out = get_multidiffusion_mask(mask, batch_size=3, num_tokens=16, shape=(3, 4, 4, 4), ctx=ctx)
    assert torch.equal(out, get_mask(mask, 3, 16, (3, 4, 4, 4)))
    assert ForgeCoupleDebug.counters["fallbacks_hit"] >= 1
