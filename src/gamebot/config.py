from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ActionConfig:
    type: str
    tap_offset_x: int = 0
    tap_offset_y: int = 0


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
class OffsetRectPctConfig:
    x: float
    y: float
    w: float
    h: float


@dataclass
class FlowStepConfig:
    type: str
    template: str = ""
    image: str = ""
    image_threshold: float = 0.90
    target_text: str = ""
    min_ocr_confidence: int = 55
    max_scrolls: int = 0
    swipe_scale: float = 1.0
    swipe: SwipeConfig | None = None
    swipe_pct: SwipePctConfig | None = None
    ocr_region_pct: RectPctConfig | None = None


@dataclass
class Phase2CategoryConfig:
    tab_template: str


@dataclass
class Phase2Config:
    enabled: bool
    parse_plane_info: bool
    post_start_delay_sec: float
    inter_click_delay_sec: float
    action_cycle_delay_sec: float
    idle_cycle_delay_sec: float
    card_list_region_pct: RectPctConfig | None
    card_key_icon_template: str
    card_anchor_x_pct: float
    card_start_y_pct: float
    card_step_y_pct: float
    grey_mode_template: str
    filter_button_template: str
    actionable_filter_template: str
    processing: Phase2CategoryConfig
    landing: Phase2CategoryConfig
    depart: Phase2CategoryConfig
    processing_claim_rewards_template: str
    processing_extend_contract_template: str
    processing_maintenance_template: str
    processing_claim_rewards_and_upgrade_popup_template: str
    processing_claim_reward_popup_template: str
    processing_finish_handling_template: str
    processing_not_enough_message_template: str
    processing_assign_crew_disabled_template: str
    processing_add_enabled_template: str
    processing_add_disabled_template: str
    processing_toggle_button_template: str
    processing_start_handling_template: str
    processing_max_add_clicks: int
    incorrect_enabled_templates: list[str]
    incorrect_enabled_max_passes: int
    landing_clear_to_land_template: str
    landing_select_stand_disabled_template: str
    landing_empty_stand_card_template: str
    landing_confirm_button_template: str
    depart_execute_button_template: str
    depart_yellow_button_region_pct: RectPctConfig | None
    plane_header_anchor_template: str
    plane_name_from_anchor_pct: OffsetRectPctConfig | None
    plane_model_from_anchor_pct: OffsetRectPctConfig | None
    plane_name_region_pct: RectPctConfig | None
    plane_model_region_pct: RectPctConfig | None
    crew_available_region_pct: RectPctConfig | None
    crew_required_region_pct: RectPctConfig | None


@dataclass
class BotConfig:
    adb_path: str
    serial: str
    loop_interval_sec: float
    jitter_sec: float
    screenshot_dir: str
    debug_logging: bool
    save_debug_screenshots: bool
    test_mode: bool
    templates: list[TemplateConfig]
    startup_flow: list[FlowStepConfig]
    phase2: Phase2Config


def _validate(cfg: dict[str, Any]) -> dict[str, Any]:
    required = [
        "adb_path",
        "serial",
        "loop_interval_sec",
        "jitter_sec",
        "screenshot_dir",
        "save_debug_screenshots",
        "test_mode",
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


def _parse_offset_rect_pct(item: dict[str, Any]) -> OffsetRectPctConfig:
    return OffsetRectPctConfig(
        x=float(item["x"]),
        y=float(item["y"]),
        w=float(item["w"]),
        h=float(item["h"]),
    )


def _validate_pct_range(label: str, value: float) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"{label} must be between 0 and 1, got {value}")


def _validate_rect_pct(rect: RectPctConfig, label_prefix: str) -> None:
    _validate_pct_range(f"{label_prefix}.x", rect.x)
    _validate_pct_range(f"{label_prefix}.y", rect.y)
    if rect.w <= 0 or rect.h <= 0:
        raise ValueError(f"{label_prefix}.w and {label_prefix}.h must be > 0")
    if rect.x + rect.w > 1 or rect.y + rect.h > 1:
        raise ValueError(f"{label_prefix} rectangle must stay within screen bounds")


def _validate_swipe_pct(swipe: SwipePctConfig) -> None:
    _validate_pct_range("swipe_pct.x1", swipe.x1)
    _validate_pct_range("swipe_pct.y1", swipe.y1)
    _validate_pct_range("swipe_pct.x2", swipe.x2)
    _validate_pct_range("swipe_pct.y2", swipe.y2)


def _validate_offset_rect_pct(rect: OffsetRectPctConfig, label_prefix: str) -> None:
    if rect.w <= 0 or rect.h <= 0:
        raise ValueError(f"{label_prefix}.w and {label_prefix}.h must be > 0")
    if rect.w > 1 or rect.h > 1:
        raise ValueError(f"{label_prefix}.w and {label_prefix}.h must be <= 1")
    if rect.x < -1 or rect.x > 1 or rect.y < -1 or rect.y > 1:
        raise ValueError(f"{label_prefix}.x and {label_prefix}.y must be between -1 and 1")


def _build_default_phase2() -> dict[str, Any]:
    return {
        "enabled": True,
        "parse_plane_info": True,
        "post_start_delay_sec": 2.0,
        "inter_click_delay_sec": 0.3,
        "action_cycle_delay_sec": 0.5,
        "idle_cycle_delay_sec": 2.0,
        "card_list_region_pct": {"x": 0.74, "y": 0.14, "w": 0.24, "h": 0.68},
        "card_key_icon_template": "",
        "card_anchor_x_pct": 0.20,
        "card_start_y_pct": 0.10,
        "card_step_y_pct": 0.17,
        "grey_mode_template": "grey_mode_button",
        "filter_button_template": "filter_button",
        "actionable_filter_template": "actionable_filter_button",
        "processing": {
            "tab_template": "processing_tab_button",
        },
        "landing": {
            "tab_template": "landing_tab_button",
        },
        "depart": {
            "tab_template": "depart_tab_button",
        },
        "processing_claim_rewards_template": "processing_claim_rewards_button",
        "processing_extend_contract_template": "",
        "processing_maintenance_template": "",
        "processing_claim_rewards_and_upgrade_popup_template": "processing_claim_rewards_and_upgrade_popup_button",
        "processing_claim_reward_popup_template": "",
        "processing_finish_handling_template": "processing_finish_handling_button",
        "processing_not_enough_message_template": "processing_not_enough_crew_message",
        "processing_assign_crew_disabled_template": "processing_assign_crew_disabled_button",
        "processing_add_enabled_template": "processing_add_enabled_button",
        "processing_add_disabled_template": "processing_add_disabled_button",
        "processing_toggle_button_template": "processing_ramp_agent_toggle_button",
        "processing_start_handling_template": "processing_start_handling_button",
        "processing_max_add_clicks": 12,
        "incorrect_enabled_templates": [],
        "incorrect_enabled_max_passes": 2,
        "landing_clear_to_land_template": "landing_clear_to_land_button",
        "landing_select_stand_disabled_template": "landing_select_stand_disabled_button",
        "landing_empty_stand_card_template": "landing_empty_stand_option",
        "landing_confirm_button_template": "landing_confirm_button",
        "depart_execute_button_template": "depart_execute_button",
        "depart_yellow_button_region_pct": {"x": 0.02, "y": 0.72, "w": 0.24, "h": 0.24},
        "plane_header_anchor_template": "",
        "plane_name_from_anchor_pct": {"x": -0.205, "y": -0.090, "w": 0.075, "h": 0.058},
        "plane_model_from_anchor_pct": {"x": -0.135, "y": -0.090, "w": 0.055, "h": 0.058},
    }


def _load_phase2(raw: dict[str, Any]) -> Phase2Config:
    merged = _build_default_phase2()
    merged.update(raw)

    processing_raw = dict(merged["processing"])
    landing_raw = dict(merged["landing"])
    depart_raw = dict(merged["depart"])

    phase2 = Phase2Config(
        enabled=bool(merged.get("enabled", True)),
        parse_plane_info=bool(merged.get("parse_plane_info", True)),
        post_start_delay_sec=float(merged.get("post_start_delay_sec", 2.0)),
        inter_click_delay_sec=float(merged.get("inter_click_delay_sec", 0.3)),
        action_cycle_delay_sec=float(merged.get("action_cycle_delay_sec", 0.5)),
        idle_cycle_delay_sec=float(merged.get("idle_cycle_delay_sec", 2.0)),
        card_list_region_pct=(
            _parse_rect_pct(merged["card_list_region_pct"])
            if "card_list_region_pct" in merged
            else None
        ),
        card_key_icon_template=str(merged.get("card_key_icon_template", "")),
        card_anchor_x_pct=float(merged.get("card_anchor_x_pct", 0.45)),
        card_start_y_pct=float(merged.get("card_start_y_pct", 0.10)),
        card_step_y_pct=float(merged.get("card_step_y_pct", 0.17)),
        grey_mode_template=str(merged["grey_mode_template"]),
        filter_button_template=str(merged["filter_button_template"]),
        actionable_filter_template=str(merged["actionable_filter_template"]),
        processing=Phase2CategoryConfig(
            tab_template=str(processing_raw["tab_template"]),
        ),
        landing=Phase2CategoryConfig(
            tab_template=str(landing_raw["tab_template"]),
        ),
        depart=Phase2CategoryConfig(
            tab_template=str(depart_raw["tab_template"]),
        ),
        processing_claim_rewards_template=str(
            merged.get("processing_claim_rewards_template", "processing_claim_rewards_button")
        ),
        processing_extend_contract_template=str(
            merged.get("processing_extend_contract_template", "")
        ),
        processing_maintenance_template=str(
            merged.get("processing_maintenance_template", "")
        ),
        processing_claim_rewards_and_upgrade_popup_template=str(
            merged.get(
                "processing_claim_rewards_and_upgrade_popup_template",
                "processing_claim_rewards_and_upgrade_popup_button",
            )
        ),
        processing_claim_reward_popup_template=str(
            merged.get("processing_claim_reward_popup_template", "")
        ),
        processing_finish_handling_template=str(
            merged.get("processing_finish_handling_template", "processing_finish_handling_button")
        ),
        processing_not_enough_message_template=str(
            merged.get(
                "processing_not_enough_message_template",
                merged.get("processing_not_enough_message", "processing_not_enough_crew_message"),
            )
        ),
        processing_assign_crew_disabled_template=str(
            merged.get(
                "processing_assign_crew_disabled_template",
                "processing_assign_crew_disabled_button",
            )
        ),
        processing_add_enabled_template=str(
            merged.get(
                "processing_add_enabled_template",
                merged.get("processing_add_button_template", "processing_add_enabled_button"),
            )
        ),
        processing_add_disabled_template=str(
            merged.get("processing_add_disabled_template", "processing_add_disabled_button")
        ),
        processing_toggle_button_template=str(
            merged.get(
                "processing_toggle_button_template",
                merged.get(
                    "processing_ramp_agent_template",
                    merged.get("processing_toggle_button", "processing_ramp_agent_toggle_button"),
                ),
            )
        ),
        processing_start_handling_template=str(
            merged.get("processing_start_handling_template", "processing_start_handling_button")
        ),
        processing_max_add_clicks=int(merged.get("processing_max_add_clicks", 12)),
        incorrect_enabled_templates=[str(x) for x in merged.get("incorrect_enabled_templates", [])],
        incorrect_enabled_max_passes=int(merged.get("incorrect_enabled_max_passes", 2)),
        landing_clear_to_land_template=str(
            merged.get("landing_clear_to_land_template", merged.get("landing_direct_button_template", ""))
        ),
        landing_select_stand_disabled_template=str(
            merged.get("landing_select_stand_disabled_template", "landing_select_stand_disabled_button")
        ),
        landing_empty_stand_card_template=str(
            merged.get(
                "landing_empty_stand_card_template",
                merged.get("landing_empty_stand_card", "landing_empty_stand_option"),
            )
        ),
        landing_confirm_button_template=str(
            merged.get("landing_confirm_button_template", "landing_confirm_button")
        ),
        depart_execute_button_template=str(merged["depart_execute_button_template"]),
        depart_yellow_button_region_pct=(
            _parse_rect_pct(merged["depart_yellow_button_region_pct"])
            if "depart_yellow_button_region_pct" in merged
            else None
        ),
        plane_header_anchor_template=str(merged.get("plane_header_anchor_template", "")),
        plane_name_from_anchor_pct=(
            _parse_offset_rect_pct(merged["plane_name_from_anchor_pct"])
            if "plane_name_from_anchor_pct" in merged
            else None
        ),
        plane_model_from_anchor_pct=(
            _parse_offset_rect_pct(merged["plane_model_from_anchor_pct"])
            if "plane_model_from_anchor_pct" in merged
            else None
        ),
        plane_name_region_pct=(
            _parse_rect_pct(merged["plane_name_region_pct"])
            if "plane_name_region_pct" in merged
            else None
        ),
        plane_model_region_pct=(
            _parse_rect_pct(merged["plane_model_region_pct"])
            if "plane_model_region_pct" in merged
            else None
        ),
        crew_available_region_pct=(
            _parse_rect_pct(merged["crew_available_region_pct"])
            if "crew_available_region_pct" in merged
            else None
        ),
        crew_required_region_pct=(
            _parse_rect_pct(merged["crew_required_region_pct"])
            if "crew_required_region_pct" in merged
            else None
        ),
    )

    if phase2.inter_click_delay_sec < 0:
        raise ValueError("phase2.inter_click_delay_sec must be >= 0")
    if phase2.post_start_delay_sec < 0:
        raise ValueError("phase2.post_start_delay_sec must be >= 0")
    if phase2.action_cycle_delay_sec <= 0 or phase2.idle_cycle_delay_sec <= 0:
        raise ValueError("phase2.action_cycle_delay_sec and phase2.idle_cycle_delay_sec must be > 0")
    if phase2.processing_max_add_clicks < 1:
        raise ValueError("phase2.processing_max_add_clicks must be >= 1")
    if phase2.incorrect_enabled_max_passes < 1:
        raise ValueError("phase2.incorrect_enabled_max_passes must be >= 1")
    _validate_pct_range("phase2.card_anchor_x_pct", phase2.card_anchor_x_pct)
    _validate_pct_range("phase2.card_start_y_pct", phase2.card_start_y_pct)
    if phase2.card_step_y_pct <= 0 or phase2.card_step_y_pct > 1:
        raise ValueError("phase2.card_step_y_pct must be > 0 and <= 1")

    if phase2.plane_name_region_pct is not None:
        _validate_rect_pct(phase2.plane_name_region_pct, "phase2.plane_name_region_pct")
    if phase2.plane_model_region_pct is not None:
        _validate_rect_pct(phase2.plane_model_region_pct, "phase2.plane_model_region_pct")
    if phase2.plane_name_from_anchor_pct is not None:
        _validate_offset_rect_pct(
            phase2.plane_name_from_anchor_pct,
            "phase2.plane_name_from_anchor_pct",
        )
    if phase2.plane_model_from_anchor_pct is not None:
        _validate_offset_rect_pct(
            phase2.plane_model_from_anchor_pct,
            "phase2.plane_model_from_anchor_pct",
        )
    if phase2.crew_available_region_pct is not None:
        _validate_rect_pct(
            phase2.crew_available_region_pct,
            "phase2.crew_available_region_pct",
        )
    if phase2.crew_required_region_pct is not None:
        _validate_rect_pct(
            phase2.crew_required_region_pct,
            "phase2.crew_required_region_pct",
        )
    if phase2.depart_yellow_button_region_pct is not None:
        _validate_rect_pct(
            phase2.depart_yellow_button_region_pct,
            "phase2.depart_yellow_button_region_pct",
        )
    if phase2.card_list_region_pct is not None:
        _validate_rect_pct(
            phase2.card_list_region_pct,
            "phase2.card_list_region_pct",
        )

    return phase2


def load_config(path: str) -> BotConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    cfg = _validate(raw)

    templates: list[TemplateConfig] = []
    for item in cfg["templates"]:
        action = ActionConfig(
            type=item["action"]["type"],
            tap_offset_x=int(item["action"].get("tap_offset_x", 0)),
            tap_offset_y=int(item["action"].get("tap_offset_y", 0)),
        )
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
            image=step.get("image", ""),
            image_threshold=float(step.get("image_threshold", 0.90)),
            target_text=step.get("target_text", ""),
            min_ocr_confidence=int(step.get("min_ocr_confidence", 55)),
            max_scrolls=int(step.get("max_scrolls", 0)),
            swipe_scale=float(step.get("swipe_scale", 1.0)),
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
        if flow_step.swipe_scale <= 0 or flow_step.swipe_scale > 1:
            raise ValueError("swipe_scale must be > 0 and <= 1")
        if flow_step.ocr_region_pct is not None:
            _validate_rect_pct(flow_step.ocr_region_pct, "ocr_region_pct")

        has_swipe = flow_step.swipe is not None or flow_step.swipe_pct is not None

        if step_type == "click_template" and not flow_step.template:
            raise ValueError("click_template step requires `template`")
        if step_type == "pick_airport" and (
            not flow_step.template or not has_swipe or flow_step.max_scrolls < 1
        ):
            raise ValueError(
                "pick_airport step requires `template`, (`swipe` or `swipe_pct`), and `max_scrolls >= 1`"
            )
        if step_type == "pick_airport_image" and (
            not flow_step.image or not has_swipe or flow_step.max_scrolls < 1
        ):
            raise ValueError(
                "pick_airport_image step requires `image`, (`swipe` or `swipe_pct`), and `max_scrolls >= 1`"
            )
        if step_type == "pick_airport_text" and (
            not flow_step.target_text or not has_swipe or flow_step.max_scrolls < 1
        ):
            raise ValueError(
                "pick_airport_text step requires `target_text`, (`swipe` or `swipe_pct`), and `max_scrolls >= 1`"
            )

        startup_flow.append(flow_step)

    phase2 = _load_phase2(dict(cfg.get("phase2", {})))

    return BotConfig(
        adb_path=cfg["adb_path"],
        serial=cfg["serial"],
        loop_interval_sec=float(cfg["loop_interval_sec"]),
        jitter_sec=float(cfg["jitter_sec"]),
        screenshot_dir=cfg["screenshot_dir"],
        debug_logging=bool(cfg.get("debug_logging", False)),
        save_debug_screenshots=bool(cfg["save_debug_screenshots"]),
        test_mode=bool(cfg["test_mode"]),
        templates=templates,
        startup_flow=startup_flow,
        phase2=phase2,
    )
