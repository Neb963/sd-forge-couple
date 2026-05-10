import torch

from lib_couple.attention_masks import get_multidiffusion_mask
from lib_couple.multidiffusion_context import MultiDiffusionTileContext


def test_mask_crop_math_with_non_square_ratio():
    mask = torch.zeros((1, 1, 60, 100))
    mask[:, :, 20:40, 30:70] = 1.0
    ctx = MultiDiffusionTileContext(
        bboxes=[(3, 2, 7, 4), (0, 0, 2, 2)],
        full_latent_h=6,
        full_latent_w=10,
        tile_batch_count=2,
    )
    out = get_multidiffusion_mask(mask, batch_size=2, num_tokens=16, shape=(2, 4, 4, 4), ctx=ctx)
    assert out.shape == (2, 16, 1)
    assert torch.all(out[0] == 1.0)
    assert torch.all(out[1] == 0.0)


def test_mask_crop_math_with_different_x_y_scale():
    mask = torch.zeros((1, 1, 20, 80))
    mask[:, :, 5:15, 20:60] = 1.0
    ctx = MultiDiffusionTileContext(
        bboxes=[(2, 1, 6, 3)],
        full_latent_h=4,
        full_latent_w=8,
        tile_batch_count=1,
    )
    out = get_multidiffusion_mask(mask, batch_size=1, num_tokens=8, shape=(1, 4, 2, 4), ctx=ctx)
    assert out.shape == (1, 8, 1)
    assert torch.all(out == 1.0)
