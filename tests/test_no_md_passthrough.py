import torch

from lib_couple.attention_masks import get_mask
from lib_couple.multidiffusion_context import get_md_tile_context


def _mask_dispatch(mask, batch_size, num_tokens, shape):
    ctx = get_md_tile_context()
    if ctx is None:
        return get_mask(mask, batch_size, num_tokens, shape)
    raise AssertionError("unexpected active context")


def test_no_md_passthrough_is_bit_identical():
    mask = torch.rand((2, 1, 64, 64))
    expected = get_mask(mask, 2, 64, (2, 4, 8, 8))
    actual = _mask_dispatch(mask, 2, 64, (2, 4, 8, 8))
    assert torch.equal(actual, expected)
