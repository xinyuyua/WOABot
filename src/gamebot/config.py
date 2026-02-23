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
class FlowStepConfig:
    type: str
    template: str
    max_scrolls: int = 0
    swipe: SwipeConfig | None = None


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
            template=step["template"],
            max_scrolls=int(step.get("max_scrolls", 0)),
            swipe=_parse_swipe(step["swipe"]) if "swipe" in step else None,
        )
        if step_type == "pick_airport" and (flow_step.swipe is None or flow_step.max_scrolls < 1):
            raise ValueError(
                "pick_airport step requires `swipe` and `max_scrolls >= 1`"
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
