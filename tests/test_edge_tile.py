import torch

from lib_couple.attention_masks import get_multidiffusion_mask
from lib_couple.multidiffusion_context import MultiDiffusionTileContext


def test_edge_tile_smaller_than_nominal_keeps_shape():
    mask = torch.ones((2, 1, 96, 160))
    ctx = MultiDiffusionTileContext(
        bboxes=[(12, 8, 20, 12)],
        full_latent_h=12,
        full_latent_w=20,
        tile_batch_count=1,
    )
    out = get_multidiffusion_mask(mask, batch_size=1, num_tokens=32, shape=(1, 4, 4, 8), ctx=ctx)
    assert out.shape == (2, 32, 1)
    assert torch.all(out == 1.0)
