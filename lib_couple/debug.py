from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


class ForgeCoupleDebug:
    """Structured debug logging and optional mask dumping for Forge Couple."""

    enabled: bool = os.environ.get("FORGE_COUPLE_DEBUG", "").strip() in {"1", "true", "True", "yes", "on"}
    dump_masks_enabled: bool = os.environ.get("FORGE_COUPLE_DUMP_MASKS", "").strip() in {"1", "true", "True", "yes", "on"}
    counters: dict[str, int] = {
        "tiles_seen": 0,
        "mask_crops_applied": 0,
        "fallbacks_hit": 0,
    }
    _lock = Lock()

    @classmethod
    def set_enabled(cls, enabled: bool) -> None:
        cls.enabled = bool(enabled)
        cls.log("debug", "debug logging toggled", enabled=cls.enabled)

    @classmethod
    def set_dump_masks_enabled(cls, enabled: bool) -> None:
        cls.dump_masks_enabled = bool(enabled)
        cls.log("debug", "mask dumping toggled", enabled=cls.dump_masks_enabled)

    @classmethod
    def _logs_root(cls) -> Path:
        return Path.cwd() / "logs"

    @classmethod
    def _log_file(cls) -> Path:
        root = cls._logs_root()
        root.mkdir(parents=True, exist_ok=True)
        return root / "forge_couple_debug.log"

    @classmethod
    def log(cls, category: str, msg: str, **kv: Any) -> None:
        if not cls.enabled:
            return
        now = datetime.now().isoformat(timespec="milliseconds")
        fields = " ".join(f"{k}={v!r}" for k, v in sorted(kv.items()))
        line = f"{now} [forge-couple:{category}] {msg}"
        if fields:
            line = f"{line} {fields}"
        with cls._lock:
            print(line, file=sys.stdout)
            try:
                with cls._log_file().open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                # Logging must never break sampling.
                pass

    @classmethod
    def incr(cls, key: str, amount: int = 1) -> None:
        cls.counters[key] = cls.counters.get(key, 0) + amount

    @classmethod
    def dump_mask(cls, name: str, tensor: Any) -> None:
        if not (cls.enabled and cls.dump_masks_enabled):
            return
        if torch is None or Image is None:
            cls.log("mask", "mask dump unavailable", reason="torch_or_pil_missing")
            return
        try:
            t = tensor.detach().float().cpu()
            while t.ndim > 2:
                t = t[0]
            t_min = t.min().item() if t.numel() else 0.0
            t_max = t.max().item() if t.numel() else 1.0
            if t_max > t_min:
                t = (t - t_min) / (t_max - t_min)
            arr = (t.clamp(0, 1).numpy() * 255).astype("uint8")
            out_dir = cls._logs_root() / "forge_couple_masks"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            Image.fromarray(arr).save(out_dir / f"{ts}_{safe_name}.png")
        except Exception as exc:
            cls.log("mask", "failed to dump mask", name=name, error=repr(exc))

    @classmethod
    def dump_context(cls, ctx: Any) -> None:
        if ctx is None:
            cls.log("context", "no active multidiffusion context")
            return
        boxes = []
        for bbox in getattr(ctx, "bboxes", []):
            try:
                boxes.append(tuple(int(bbox[i]) for i in range(4)))
            except Exception:
                boxes.append(repr(bbox))
        cls.log(
            "context",
            "active multidiffusion context",
            full_latent_h=getattr(ctx, "full_latent_h", None),
            full_latent_w=getattr(ctx, "full_latent_w", None),
            tile_count=len(getattr(ctx, "bboxes", [])),
            bboxes=boxes,
            counters=dict(cls.counters),
        )
