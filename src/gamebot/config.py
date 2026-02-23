from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ActionConfig:
    type: str


@dataclass
class TemplateConfig:
    name: str
    path: str
    threshold: float
    action: ActionConfig


@dataclass
class SwipeConfig:
    x1: int
    y1: int
    x2: int
    y2: int
    duration_ms: int


@dataclass
class SwipePctConfig:
    x1: float
    y1: float
    x2: float
    y2: float
    duration_ms: int


@dataclass
class RectPctConfig:
    x: float
    y: float
    w: float
    h: float


@dataclass
class FlowStepConfig:
    type: str
    template: str = ""
    target_text: str = ""
    min_ocr_confidence: int = 55
    max_scrolls: int = 0
    swipe: SwipeConfig | None = None
    swipe_pct: SwipePctConfig | None = None
    ocr_region_pct: RectPctConfig | None = None


@dataclass
class BotConfig:
    adb_path: str
    serial: str
    loop_interval_sec: float
    jitter_sec: float
    screenshot_dir: str
    save_debug_screenshots: bool
    templates: list[TemplateConfig]
    startup_flow: list[FlowStepConfig]


def _validate(cfg: dict[str, Any]) -> dict[str, Any]:
    required = [
        "adb_path",
        "serial",
        "loop_interval_sec",
        "jitter_sec",
        "screenshot_dir",
        "save_debug_screenshots",
        "templates",
        "startup_flow",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    return cfg


def _parse_swipe(item: dict[str, Any]) -> SwipeConfig:
    return SwipeConfig(
        x1=int(item["x1"]),
        y1=int(item["y1"]),
        x2=int(item["x2"]),
        y2=int(item["y2"]),
        duration_ms=int(item.get("duration_ms", 200)),
    )


def _parse_swipe_pct(item: dict[str, Any]) -> SwipePctConfig:
    return SwipePctConfig(
        x1=float(item["x1"]),
        y1=float(item["y1"]),
        x2=float(item["x2"]),
        y2=float(item["y2"]),
        duration_ms=int(item.get("duration_ms", 200)),
    )


def _parse_rect_pct(item: dict[str, Any]) -> RectPctConfig:
    return RectPctConfig(
        x=float(item["x"]),
        y=float(item["y"]),
        w=float(item["w"]),
        h=float(item["h"]),
    )


def _validate_pct_range(label: str, value: float) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"{label} must be between 0 and 1, got {value}")


def _validate_rect_pct(rect: RectPctConfig) -> None:
    _validate_pct_range("ocr_region_pct.x", rect.x)
    _validate_pct_range("ocr_region_pct.y", rect.y)
    if rect.w <= 0 or rect.h <= 0:
        raise ValueError("ocr_region_pct.w and ocr_region_pct.h must be > 0")
    if rect.x + rect.w > 1 or rect.y + rect.h > 1:
        raise ValueError("ocr_region_pct rectangle must stay within screen bounds")


def _validate_swipe_pct(swipe: SwipePctConfig) -> None:
    _validate_pct_range("swipe_pct.x1", swipe.x1)
    _validate_pct_range("swipe_pct.y1", swipe.y1)
    _validate_pct_range("swipe_pct.x2", swipe.x2)
    _validate_pct_range("swipe_pct.y2", swipe.y2)


def load_config(path: str) -> BotConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    cfg = _validate(raw)

    templates: list[TemplateConfig] = []
    for item in cfg["templates"]:
        action = ActionConfig(type=item["action"]["type"])
        templates.append(
            TemplateConfig(
                name=item["name"],
                path=item["path"],
                threshold=float(item["threshold"]),
                action=action,
            )
        )

    startup_flow: list[FlowStepConfig] = []
    for step in cfg["startup_flow"]:
        step_type = step["type"]
        flow_step = FlowStepConfig(
            type=step_type,
            template=step.get("template", ""),
            target_text=step.get("target_text", ""),
            min_ocr_confidence=int(step.get("min_ocr_confidence", 55)),
            max_scrolls=int(step.get("max_scrolls", 0)),
            swipe=_parse_swipe(step["swipe"]) if "swipe" in step else None,
            swipe_pct=_parse_swipe_pct(step["swipe_pct"]) if "swipe_pct" in step else None,
            ocr_region_pct=(
                _parse_rect_pct(step["ocr_region_pct"])
                if "ocr_region_pct" in step
                else None
            ),
        )

        if flow_step.swipe_pct is not None:
            _validate_swipe_pct(flow_step.swipe_pct)
        if flow_step.ocr_region_pct is not None:
            _validate_rect_pct(flow_step.ocr_region_pct)

        has_swipe = flow_step.swipe is not None or flow_step.swipe_pct is not None

        if step_type == "click_template" and not flow_step.template:
            raise ValueError("click_template step requires `template`")
        if step_type == "pick_airport" and (
            not flow_step.template or not has_swipe or flow_step.max_scrolls < 1
        ):
            raise ValueError(
                "pick_airport step requires `template`, (`swipe` or `swipe_pct`), and `max_scrolls >= 1`"
            )
        if step_type == "pick_airport_text" and (
            not flow_step.target_text or not has_swipe or flow_step.max_scrolls < 1
        ):
            raise ValueError(
                "pick_airport_text step requires `target_text`, (`swipe` or `swipe_pct`), and `max_scrolls >= 1`"
            )

        startup_flow.append(flow_step)

    return BotConfig(
        adb_path=cfg["adb_path"],
        serial=cfg["serial"],
        loop_interval_sec=float(cfg["loop_interval_sec"]),
        jitter_sec=float(cfg["jitter_sec"]),
        screenshot_dir=cfg["screenshot_dir"],
        save_debug_screenshots=bool(cfg["save_debug_screenshots"]),
        templates=templates,
        startup_flow=startup_flow,
    )
