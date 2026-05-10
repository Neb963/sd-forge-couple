from concurrent.futures import ThreadPoolExecutor

from lib_couple.multidiffusion_context import (
    MultiDiffusionTileContext,
    get_md_tile_context,
    reset_md_tile_context,
    set_md_tile_context,
)


def _worker(value):
    ctx = MultiDiffusionTileContext(
        bboxes=[(value, 0, value + 1, 1)],
        full_latent_h=1,
        full_latent_w=10,
        tile_batch_count=1,
    )
    token = set_md_tile_context(ctx)
    try:
        return get_md_tile_context().bboxes[0][0]
    finally:
        reset_md_tile_context(token)


def test_contextvars_do_not_leak_between_threads():
    with ThreadPoolExecutor(max_workers=4) as ex:
        values = list(ex.map(_worker, range(8)))
    assert values == list(range(8))
    assert get_md_tile_context() is None
