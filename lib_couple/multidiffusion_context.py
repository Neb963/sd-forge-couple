from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from .debug import ForgeCoupleDebug


@dataclass(frozen=True)
class MultiDiffusionTileContext:
    bboxes: list[Any]
    full_latent_h: int
    full_latent_w: int
    tile_batch_count: int


_current_md_tile_context: ContextVar[MultiDiffusionTileContext | None] = ContextVar(
    "forge_couple_current_md_tile_context",
    default=None,
)


def set_md_tile_context(ctx: MultiDiffusionTileContext) -> Token:
    ForgeCoupleDebug.incr("tiles_seen", len(ctx.bboxes))
    ForgeCoupleDebug.log(
        "context",
        "set multidiffusion tile context",
        full_latent_h=ctx.full_latent_h,
        full_latent_w=ctx.full_latent_w,
        tile_batch_count=ctx.tile_batch_count,
        bbox_count=len(ctx.bboxes),
    )
    ForgeCoupleDebug.dump_context(ctx)
    return _current_md_tile_context.set(ctx)


def reset_md_tile_context(token: Token) -> None:
    ForgeCoupleDebug.log("context", "reset multidiffusion tile context")
    _current_md_tile_context.reset(token)


def get_md_tile_context() -> MultiDiffusionTileContext | None:
    return _current_md_tile_context.get()
