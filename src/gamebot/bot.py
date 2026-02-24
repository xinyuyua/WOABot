from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from .adb_client import AdbClient
from .config import (
    ActionConfig,
    BotConfig,
    FlowStepConfig,
    OffsetRectPctConfig,
    Phase2CategoryConfig,
    RectPctConfig,
)
from .detector import MatchResult, Template, load_template, match_template

ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"


@dataclass
class PlaneRecord:
    name: str
    category: str
    last_action: str
    last_seen_epoch_ms: int


@dataclass
class GameBot:
    config: BotConfig

    def __post_init__(self) -> None:
        self.adb = AdbClient(self.config.adb_path, self.config.serial)
        self.templates = {
            t.name: load_template(path=t.path, name=t.name, threshold=t.threshold)
            for t in self.config.templates
        }
        self.template_actions = {t.name: t.action for t in self.config.templates}
        self.flow_image_templates: dict[str, Template] = {}

        self.startup_index = 0
        self.airport_scroll_count = 0

        self.phase2_grey_enabled = False
        self.phase2_filter_enabled = False
        self.phase2_started = False
        self.phase2_missing_template_warned: set[str] = set()
        self.phase2_plane_memory: dict[str, PlaneRecord] = {}
        self.phase2_test_mode_category_index = 0
        self.phase2_test_mode_card_index: dict[str, int] = {
            "processing": 0,
            "landing": 0,
            "depart": 0,
        }
        self.shared_action_button_region_px: tuple[int, int, int, int] | None = None
        self.processing_add_button_anchor_px: tuple[int, int] | None = None
        self.button_tap_cache_px: dict[str, tuple[int, int]] = {}
        self.current_plane_name = "unknown"
        self.current_plane_model = "unknown"

        self._next_sleep_override_sec: float | None = None

        Path(self.config.screenshot_dir).mkdir(parents=True, exist_ok=True)

    def _capture_frame(self) -> np.ndarray:
        return self._decode_frame(self.adb.screenshot_png_bytes())

    def _log_warn(self, msg: str, frame: np.ndarray | None = None) -> None:
        print(f"{ANSI_RED}[WARN]{ANSI_RESET} !! {msg}")
        self._capture_failure_snapshot("warn", frame=frame)

    def _log_error(self, msg: str, frame: np.ndarray | None = None) -> None:
        print(f"{ANSI_RED}[ERROR]{ANSI_RESET} !! {msg}")
        self._capture_failure_snapshot("error", frame=frame)

    def _log_fail(self, msg: str, frame: np.ndarray | None = None) -> None:
        print(f"{ANSI_RED}[FAIL]{ANSI_RESET} !! {msg}")
        self._capture_failure_snapshot("fail", frame=frame)

    def _log_debug(self, msg: str) -> None:
        if self.config.debug_logging:
            print(f"[DEBUG] {msg}")

    def _sleep(self, seconds: float) -> None:
        delay = max(0.01, seconds + random.uniform(-0.1, 0.1))
        time.sleep(delay)

    def _decode_frame(self, png_bytes: bytes) -> np.ndarray:
        if not png_bytes:
            raise ValueError(
                "Screenshot bytes are empty or not a valid PNG stream. "
                "Check `adb devices` and run: adb exec-out screencap -p > /tmp/screen.png"
            )
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(
                f"Failed to decode screenshot bytes (len={len(png_bytes)}). "
                "Try: adb exec-out screencap -p > /tmp/screen.png and inspect the file."
            )
        return frame

    def _tap_match_center(
        self,
        frame: np.ndarray,
        match: MatchResult,
        offset_x: int = 0,
        offset_y: int = 0,
        debug_label: str | None = None,
    ) -> tuple[int, int]:
        x, y = self._resolve_match_center(frame, match, offset_x, offset_y)
        self.adb.tap(x, y)
        self._save_action_debug(
            frame,
            debug_label or f"tap_match_{match.name}",
            tap_xy=(x, y),
            match=match,
        )
        return x, y

    def _resolve_match_center(
        self,
        frame: np.ndarray,
        match: MatchResult,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> tuple[int, int]:
        x = match.x + (match.w // 2) + offset_x
        y = match.y + (match.h // 2) + offset_y
        height, width = frame.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        return x, y

    def _find_match(self, frame: np.ndarray, template_name: str) -> MatchResult | None:
        tmpl = self.templates.get(template_name)
        if tmpl is None:
            raise ValueError(f"Template not defined in config.templates: {template_name}")
        return match_template(frame, tmpl)

    def _get_flow_image_template(self, image_name: str, threshold: float) -> Template:
        key = f"{image_name}:{threshold:.4f}"
        tmpl = self.flow_image_templates.get(key)
        if tmpl is None:
            image_path = Path("templates") / image_name
            tmpl = load_template(path=str(image_path), name=image_name, threshold=threshold)
            self.flow_image_templates[key] = tmpl
        return tmpl

    def _best_template_score(self, frame: np.ndarray, tmpl: Template) -> tuple[float, int, int]:
        result = cv2.matchTemplate(frame, tmpl.image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return float(max_val), int(max_loc[0]), int(max_loc[1])

    def _save_debug(self, frame: np.ndarray, label: str, force: bool = False) -> str | None:
        if not force and not self.config.save_debug_screenshots:
            return None
        ts = int(time.time() * 1000)
        safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label).strip("_") or "debug"
        out = Path(self.config.screenshot_dir) / f"{safe_label}_{ts}.png"
        cv2.imwrite(str(out), frame)
        return str(out)

    def _save_action_debug(
        self,
        frame: np.ndarray,
        label: str,
        tap_xy: tuple[int, int] | None = None,
        match: MatchResult | None = None,
    ) -> str | None:
        if not self.config.save_debug_screenshots:
            return None
        debug = frame.copy()
        if match is not None:
            cv2.rectangle(
                debug,
                (match.x, match.y),
                (match.x + match.w, match.y + match.h),
                (0, 255, 255),
                2,
            )
        if tap_xy is not None:
            cv2.circle(debug, tap_xy, 8, (0, 0, 255), -1)
        path = self._save_debug(debug, f"action_{label}", force=False)
        if path:
            print(f"[DEBUG] action_screenshot={path}")
        return path

    def _capture_failure_snapshot(self, label: str, frame: np.ndarray | None = None) -> None:
        snapshot = frame
        if snapshot is None:
            try:
                snapshot = self._capture_frame()
            except Exception:
                snapshot = None
        if snapshot is None:
            return
        path = self._save_debug(snapshot, f"{label}_snapshot", force=True)
        if path:
            print(f"[DEBUG] failure_screenshot={path}")

    def _to_abs_xy(self, frame: np.ndarray, x_pct: float, y_pct: float) -> tuple[int, int]:
        height, width = frame.shape[:2]
        x = int(round(x_pct * width))
        y = int(round(y_pct * height))
        return x, y

    def _tap_abs(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        do_tap: bool = True,
        debug_label: str = "tap_abs",
    ) -> tuple[int, int]:
        height, width = frame.shape[:2]
        cx = max(0, min(x, width - 1))
        cy = max(0, min(y, height - 1))
        if do_tap:
            self.adb.tap(cx, cy)
            self._save_action_debug(frame, debug_label, tap_xy=(cx, cy))
        return cx, cy

    def _resolve_swipe(self, frame: np.ndarray, step: FlowStepConfig) -> tuple[int, int, int, int, int]:
        if step.swipe_pct is not None:
            x1, y1 = self._to_abs_xy(frame, step.swipe_pct.x1, step.swipe_pct.y1)
            x2, y2 = self._to_abs_xy(frame, step.swipe_pct.x2, step.swipe_pct.y2)
            duration_ms = step.swipe_pct.duration_ms
        elif step.swipe is not None:
            x1, y1, x2, y2 = step.swipe.x1, step.swipe.y1, step.swipe.x2, step.swipe.y2
            duration_ms = step.swipe.duration_ms
        else:
            raise ValueError("Step requires swipe config (`swipe` or `swipe_pct`)")

        if step.swipe_scale < 1.0:
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            dx = (x2 - x1) * step.swipe_scale / 2.0
            dy = (y2 - y1) * step.swipe_scale / 2.0
            x1 = int(round(mx - dx))
            y1 = int(round(my - dy))
            x2 = int(round(mx + dx))
            y2 = int(round(my + dy))

        return x1, y1, x2, y2, duration_ms

    def _run_search_swipe(self, frame: np.ndarray, step: FlowStepConfig) -> bool:
        if self.airport_scroll_count >= step.max_scrolls:
            return False

        x1, y1, x2, y2, duration_ms = self._resolve_swipe(frame, step)
        self.adb.swipe(x1, y1, x2, y2, duration_ms)
        if self.config.save_debug_screenshots:
            debug = frame.copy()
            cv2.arrowedLine(debug, (x1, y1), (x2, y2), (0, 255, 255), 3, tipLength=0.2)
            path = self._save_debug(debug, "action_swipe_search")
            if path:
                print(f"[DEBUG] action_screenshot={path}")
        self.airport_scroll_count += 1
        return True

    def _crop_by_rect_pct(
        self, frame: np.ndarray, rect: RectPctConfig
    ) -> tuple[np.ndarray, int, int]:
        height, width = frame.shape[:2]
        x = int(round(rect.x * width))
        y = int(round(rect.y * height))
        w = int(round(rect.w * width))
        h = int(round(rect.h * height))

        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        return frame[y : y + h, x : x + w], x, y

    def _box_from_rect_pct(self, frame: np.ndarray, rect: RectPctConfig) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        x1 = int(round(rect.x * width))
        y1 = int(round(rect.y * height))
        x2 = int(round((rect.x + rect.w) * width))
        y2 = int(round((rect.y + rect.h) * height))
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return x1, y1, x2, y2

    def _crop_by_box(
        self, frame: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, int, int]:
        x1, y1, x2, y2 = box
        return frame[y1:y2, x1:x2], x1, y1

    def _box_from_anchor_offset(
        self,
        frame: np.ndarray,
        anchor_x: int,
        anchor_y: int,
        offset: OffsetRectPctConfig,
    ) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        x1 = anchor_x + int(round(offset.x * w))
        y1 = anchor_y + int(round(offset.y * h))
        bw = max(1, int(round(offset.w * w)))
        bh = max(1, int(round(offset.h * h)))
        x2 = x1 + bw
        y2 = y1 + bh
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return x1, y1, x2, y2

    def _run_click_template_step(self, frame: np.ndarray, step: FlowStepConfig) -> bool:
        tmpl = self.templates.get(step.template)
        if tmpl is None:
            raise ValueError(f"Template not defined in config.templates: {step.template}")
        match = match_template(frame, tmpl)
        if not match:
            if step.template == "play_button":
                self._try_reclick_airport_for_play_recovery(frame)
            return False

        action_cfg = self.template_actions.get(step.template, ActionConfig(type="tap_center"))
        if action_cfg.type != "tap_center":
            raise ValueError(
                f"Unsupported action type `{action_cfg.type}` for template `{step.template}`"
            )

        x, y = self._tap_match_center(
            frame,
            match,
            action_cfg.tap_offset_x,
            action_cfg.tap_offset_y,
        )
        debug = frame.copy()
        cv2.rectangle(debug, (match.x, match.y), (match.x + match.w, match.y + match.h), (0, 255, 255), 2)
        cv2.circle(debug, (x, y), 8, (0, 0, 255), -1)
        self._save_debug(debug, f"flow_hit_{match.name}")
        self.startup_index += 1
        return True

    def _try_reclick_airport_for_play_recovery(self, frame: np.ndarray) -> None:
        airport_step = None
        for i in range(self.startup_index - 1, -1, -1):
            candidate = self.config.startup_flow[i]
            if candidate.type == "pick_airport_image" and candidate.image:
                airport_step = candidate
                break

        if airport_step is None:
            return

        tmpl = self._get_flow_image_template(airport_step.image, airport_step.image_threshold)
        match = match_template(frame, tmpl)
        if match:
            self._tap_match_center(frame, match)
            return

    def _run_pick_airport_step(self, frame: np.ndarray, step: FlowStepConfig) -> bool:
        match = self._find_match(frame, step.template)
        if match:
            self._tap_match_center(frame, match)
            self._save_debug(frame, f"airport_hit_{match.name}")
            self.airport_scroll_count = 0
            self.startup_index += 1
            return True

        if self.airport_scroll_count >= step.max_scrolls:
            raise RuntimeError(
                "Airport template "
                f"`{step.template}` not found after {step.max_scrolls} attempts"
            )

        self._run_search_swipe(frame, step)
        return False

    def _run_pick_airport_image_step(self, frame: np.ndarray, step: FlowStepConfig) -> bool:
        tmpl = self._get_flow_image_template(step.image, step.image_threshold)
        match = match_template(frame, tmpl)
        if match:
            self._tap_match_center(frame, match)
            self._save_debug(frame, f"airport_image_hit_{match.name}")
            self.airport_scroll_count = 0
            self.startup_index += 1
            return True

        if self.airport_scroll_count >= step.max_scrolls:
            raise RuntimeError(
                "Airport image "
                f"`{step.image}` not found after {step.max_scrolls} attempts"
            )

        if self.config.save_debug_screenshots:
            best_conf, best_x, best_y = self._best_template_score(frame, tmpl)
            debug = frame.copy()
            h, w = tmpl.image.shape[:2]
            cv2.rectangle(debug, (best_x, best_y), (best_x + w, best_y + h), (0, 255, 255), 2)
            self._save_debug(debug, f"airport_image_miss_{Path(step.image).stem}")

        self._run_search_swipe(frame, step)
        return False

    def _find_text_center(
        self,
        frame: np.ndarray,
        target_text: str,
        min_confidence: int,
        ocr_region_pct: RectPctConfig | None = None,
    ) -> tuple[int, int, float, str] | None:
        def _is_fuzzy_match(target_norm: str, text_norm: str) -> bool:
            if not target_norm or not text_norm:
                return False
            if target_norm in text_norm:
                return True
            if len(target_norm) != len(text_norm):
                return False
            mismatches = sum(1 for a, b in zip(target_norm, text_norm) if a != b)
            return mismatches <= 1

        offset_x, offset_y = 0, 0
        search_frame = frame
        if ocr_region_pct is not None:
            search_frame, offset_x, offset_y = self._crop_by_rect_pct(frame, ocr_region_pct)

        gray = cv2.cvtColor(search_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        enlarged = cv2.resize(gray, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.adaptiveThreshold(
            enlarged,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        inverted = cv2.bitwise_not(thresh)

        variants = [("gray", enlarged), ("thresh", thresh), ("inverted", inverted)]

        try:
            ocr_config = "--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            datasets = [
                (
                    name,
                    pytesseract.image_to_data(
                        img, output_type=pytesseract.Output.DICT, config=ocr_config
                    ),
                )
                for name, img in variants
            ]
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract OCR is not installed. Run `brew install tesseract`."
            ) from exc

        scale = 1.7
        target = re.sub(r"[^a-z0-9]", "", target_text.lower())
        best: tuple[int, int, float, str] | None = None
        debug_seen: list[str] = []

        for variant_name, data in datasets:
            for i, raw_text in enumerate(data.get("text", [])):
                text = raw_text.strip()
                if not text:
                    continue

                conf_str = data["conf"][i]
                try:
                    conf = float(conf_str)
                except ValueError:
                    continue

                if conf < min_confidence:
                    continue

                normalized = re.sub(r"[^a-z0-9]", "", text.lower())
                if normalized:
                    debug_seen.append(f"{variant_name}:{text}:{conf:.0f}")
                if not _is_fuzzy_match(target, normalized):
                    continue

                x = offset_x + int((int(data["left"][i]) + int(data["width"][i]) // 2) / scale)
                y = offset_y + int((int(data["top"][i]) + int(data["height"][i]) // 2) / scale)
                candidate = (x, y, conf, text)
                if best is None or conf > best[2]:
                    best = candidate

        if best is None and debug_seen:
            print("[OCR] candidates:", ", ".join(debug_seen[:8]))
        return best

    def _run_pick_airport_text_step(self, frame: np.ndarray, step: FlowStepConfig) -> bool:
        found = self._find_text_center(
            frame,
            step.target_text,
            step.min_ocr_confidence,
            step.ocr_region_pct,
        )
        if found:
            x, y, conf, text = found
            self._tap_abs(frame, x, y, do_tap=True, debug_label="airport_text_hit_ocr")
            self._save_debug(frame, "airport_text_hit")
            self.airport_scroll_count = 0
            self.startup_index += 1
            return True

        if self.airport_scroll_count >= step.max_scrolls:
            raise RuntimeError(
                "Airport text "
                f"`{step.target_text}` not found after {step.max_scrolls} attempts"
            )

        self._run_search_swipe(frame, step)
        return False

    def _run_startup_flow_step(self, frame: np.ndarray) -> bool:
        if self.startup_index >= len(self.config.startup_flow):
            return False

        step = self.config.startup_flow[self.startup_index]
        if step.type == "click_template":
            return self._run_click_template_step(frame, step)
        if step.type == "pick_airport":
            return self._run_pick_airport_step(frame, step)
        if step.type == "pick_airport_image":
            return self._run_pick_airport_image_step(frame, step)
        if step.type == "pick_airport_text":
            return self._run_pick_airport_text_step(frame, step)

        raise ValueError(f"Unsupported startup step type: {step.type}")

    def _warn_missing_template_once(self, template_name: str) -> None:
        if template_name in self.phase2_missing_template_warned:
            return
        self.phase2_missing_template_warned.add(template_name)
        self._log_warn(f"phase2 template not configured or missing from templates: {template_name}")

    def _is_static_cached_template(self, template_name: str) -> bool:
        static_non_button_templates = {
            "processing_not_enough_crew_message",
            "processing_not_enough_message",
        }
        lower = template_name.lower()
        return "button" in lower or template_name in static_non_button_templates

    def _click_template_named(
        self,
        frame: np.ndarray,
        template_name: str,
        tag: str,
        dry_run: bool = False,
        allow_cache: bool = True,
    ) -> bool:
        tmpl = self.templates.get(template_name)
        if tmpl is None:
            self._warn_missing_template_once(template_name)
            return False

        if allow_cache and not dry_run and self._is_static_cached_template(template_name):
            cached = self.button_tap_cache_px.get(template_name)
            if cached is not None:
                cx, cy = cached
                if self._is_tab_template_name(template_name) and not self._is_point_in_card_list_region(
                    frame, cx, cy
                ):
                    self._log_debug(
                        f"[CACHE] button_reject template={template_name} tap=({cx},{cy}) reason=outside_card_list_region"
                    )
                    cached = None
                if cached is None:
                    pass
                else:
                    # Validate cache against a small local ROI to avoid stale false clicks.
                    h, w = frame.shape[:2]
                    roi_half_w = max(30, tmpl.image.shape[1] * 2)
                    roi_half_h = max(20, tmpl.image.shape[0] * 2)
                    x1 = max(0, cx - roi_half_w)
                    y1 = max(0, cy - roi_half_h)
                    x2 = min(w - 1, cx + roi_half_w)
                    y2 = min(h - 1, cy + roi_half_h)
                    roi = frame[y1 : y2 + 1, x1 : x2 + 1]
                    if roi.shape[0] >= tmpl.image.shape[0] and roi.shape[1] >= tmpl.image.shape[1]:
                        local_match = match_template(roi, tmpl)
                        if local_match is not None:
                            tx, ty = self._tap_abs(
                                frame,
                                cx,
                                cy,
                                do_tap=True,
                                debug_label=f"cached_button_{tag}",
                            )
                            self._log_debug(
                                f"[CACHE] button_hit template={template_name} tap=({tx},{ty}) "
                                f"local_conf={local_match.confidence:.3f}"
                            )
                            return True
                    # Cache stale for current frame; force full-screen rematch and refresh.
                    self._log_debug(f"[CACHE] button_stale template={template_name} cached=({cx},{cy})")

        match = match_template(frame, tmpl)
        if not match:
            return False
        if self._is_tab_template_name(template_name):
            cx, cy = self._resolve_match_center(frame, match)
            if not self._is_point_in_card_list_region(frame, cx, cy):
                self._log_debug(
                    f"[PHASE2] reject_tab_match template={template_name} tap=({cx},{cy}) reason=outside_card_list_region"
                )
                return False

        action_cfg = self.template_actions.get(template_name, ActionConfig(type="tap_center"))
        if action_cfg.type != "tap_center":
            self._log_warn(f"Unsupported action type for {template_name}: {action_cfg.type}")
            return False

        if dry_run:
            x, y = self._resolve_match_center(
                frame,
                match,
                action_cfg.tap_offset_x,
                action_cfg.tap_offset_y,
            )
            print(
                f"[PHASE2-TEST] would_click tag={tag} template={template_name} "
                f"tap=({x},{y}) conf={match.confidence:.3f}"
            )
        else:
            tx, ty = self._tap_match_center(
                frame,
                match,
                action_cfg.tap_offset_x,
                action_cfg.tap_offset_y,
                debug_label=f"template_{tag}",
            )
            if self._is_static_cached_template(template_name):
                self.button_tap_cache_px[template_name] = (tx, ty)
                self._log_debug(
                    f"[CACHE] button_set template={template_name} tap=({tx},{ty}) "
                    f"conf={match.confidence:.3f}"
                )
            self._update_shared_action_button_region(frame, template_name, match)
        return True

    def _is_tab_template_name(self, template_name: str) -> bool:
        return template_name in {
            self.config.phase2.processing.tab_template,
            self.config.phase2.landing.tab_template,
            self.config.phase2.depart.tab_template,
        }

    def _is_point_in_card_list_region(self, frame: np.ndarray, x: int, y: int) -> bool:
        rect = self.config.phase2.card_list_region_pct
        if rect is None:
            return True
        h, w = frame.shape[:2]
        x1 = int(round(rect.x * w))
        y1 = int(round(rect.y * h))
        x2 = int(round((rect.x + rect.w) * w))
        y2 = int(round((rect.y + rect.h) * h))
        return x1 <= x <= x2 and y1 <= y <= y2

    def _actionable_filter_guard_y(self, frame: np.ndarray) -> int | None:
        actionable_tmpl = self.config.phase2.actionable_filter_template
        cached = self.button_tap_cache_px.get(actionable_tmpl)
        if cached is not None:
            return cached[1]
        match = self._match_template_named(frame, actionable_tmpl)
        if match is None:
            return None
        _, cy = self._resolve_match_center(frame, match)
        return cy

    def _click_leftmost_template_named(
        self,
        frame: np.ndarray,
        template_name: str,
        tag: str,
    ) -> bool:
        tmpl = self.templates.get(template_name)
        if tmpl is None:
            self._warn_missing_template_once(template_name)
            return False
        result = cv2.matchTemplate(frame, tmpl.image, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= tmpl.threshold)
        if len(xs) == 0:
            return False
        idx = min(range(len(xs)), key=lambda i: (int(xs[i]), int(ys[i])))
        x = int(xs[idx])
        y = int(ys[idx])
        match = MatchResult(
            name=tmpl.name,
            confidence=float(result[y, x]),
            x=x,
            y=y,
            w=tmpl.image.shape[1],
            h=tmpl.image.shape[0],
        )
        self._tap_match_center(
            frame,
            match,
            debug_label=f"leftmost_{tag}",
        )
        return True

    def _has_template_named(self, frame: np.ndarray, template_name: str) -> bool:
        tmpl = self.templates.get(template_name)
        if tmpl is None:
            self._warn_missing_template_once(template_name)
            return False
        return match_template(frame, tmpl) is not None

    def _match_template_named(self, frame: np.ndarray, template_name: str) -> MatchResult | None:
        tmpl = self.templates.get(template_name)
        if tmpl is None:
            self._warn_missing_template_once(template_name)
            return None
        return match_template(frame, tmpl)

    def _find_preferred_lower_left_match_named(
        self, frame: np.ndarray, template_name: str
    ) -> MatchResult | None:
        tmpl = self.templates.get(template_name)
        if tmpl is None:
            self._warn_missing_template_once(template_name)
            return None
        result = cv2.matchTemplate(frame, tmpl.image, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= tmpl.threshold)
        if len(xs) == 0:
            return None

        # Deduplicate dense neighboring hits from the same visual object.
        candidates: list[tuple[int, int, float]] = []
        min_dx = max(4, tmpl.image.shape[1] // 3)
        min_dy = max(4, tmpl.image.shape[0] // 3)
        ordered = sorted(
            [(int(x), int(y), float(result[y, x])) for x, y in zip(xs.tolist(), ys.tolist())],
            key=lambda t: t[2],
            reverse=True,
        )
        for x, y, conf in ordered:
            if any(abs(x - ox) <= min_dx and abs(y - oy) <= min_dy for ox, oy, _ in candidates):
                continue
            candidates.append((x, y, conf))

        # Constrain to likely detail-card area (left side), to avoid right-side list icons.
        h, w = frame.shape[:2]
        left_panel = [
            c
            for c in candidates
            if c[0] <= int(w * 0.45) and c[1] >= int(h * 0.18)
        ]
        pool = left_panel if left_panel else candidates

        # Lower-left preference: largest y, then smallest x, then highest confidence.
        x, y, conf = sorted(pool, key=lambda t: (-t[1], t[0], -t[2]))[0]
        return MatchResult(
            name=tmpl.name,
            confidence=conf,
            x=x,
            y=y,
            w=tmpl.image.shape[1],
            h=tmpl.image.shape[0],
        )

    def _best_conf_for_template(self, frame: np.ndarray, template_name: str) -> float | None:
        tmpl = self.templates.get(template_name)
        if tmpl is None:
            return None
        conf, _, _ = self._best_template_score(frame, tmpl)
        return conf

    def _find_rightmost_match_named(self, frame: np.ndarray, template_name: str) -> MatchResult | None:
        tmpl = self.templates.get(template_name)
        if tmpl is None:
            self._warn_missing_template_once(template_name)
            return None
        result = cv2.matchTemplate(frame, tmpl.image, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= tmpl.threshold)
        if len(xs) == 0:
            return None
        idx = max(range(len(xs)), key=lambda i: (int(xs[i]), float(result[ys[i], xs[i]])))
        x = int(xs[idx])
        y = int(ys[idx])
        return MatchResult(
            name=tmpl.name,
            confidence=float(result[y, x]),
            x=x,
            y=y,
            w=tmpl.image.shape[1],
            h=tmpl.image.shape[0],
        )

    def _set_processing_add_anchor_from_match(self, frame: np.ndarray, match: MatchResult) -> None:
        x, y = self._resolve_match_center(frame, match)
        self.processing_add_button_anchor_px = (x, y)
        print(f"[PHASE2] processing add_anchor set=({x},{y}) conf={match.confidence:.3f}")

    def _click_processing_add_anchor(self, frame: np.ndarray) -> bool:
        if self.processing_add_button_anchor_px is None:
            return False
        x, y = self.processing_add_button_anchor_px
        tx, ty = self._tap_abs(
            frame,
            x,
            y,
            do_tap=not self.config.test_mode,
            debug_label="processing_add_anchor",
        )
        self._log_debug(f"[PHASE2] processing add_anchor tap=({tx},{ty})")
        return True

    def _is_shared_action_anchor_template(self, template_name: str) -> bool:
        anchors = {
            self.config.phase2.landing_confirm_button_template,
            self.config.phase2.landing_clear_to_land_template,
            self.config.phase2.depart_execute_button_template,
        }
        return template_name in anchors

    def _update_shared_action_button_region(
        self,
        frame: np.ndarray,
        template_name: str,
        match: MatchResult,
    ) -> None:
        if not self._is_shared_action_anchor_template(template_name):
            return

        h, w = frame.shape[:2]
        # Expand around known action button to cover nearby text/shape variance.
        pad_x = int(match.w * 0.55)
        pad_y = int(match.h * 0.45)
        x1 = max(0, match.x - pad_x)
        y1 = max(0, match.y - pad_y)
        x2 = min(w - 1, match.x + match.w + pad_x)
        y2 = min(h - 1, match.y + match.h + pad_y)
        self.shared_action_button_region_px = (x1, y1, x2, y2)

    def _click_yellow_button_in_region(self, frame: np.ndarray, rect: RectPctConfig) -> bool:
        if self.shared_action_button_region_px is not None:
            x1, y1, x2, y2 = self.shared_action_button_region_px
            crop = frame[y1 : y2 + 1, x1 : x2 + 1]
            ox, oy = x1, y1
        else:
            crop, ox, oy = self._crop_by_rect_pct(frame, rect)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Yellow range tuned for UI buttons.
        lower = np.array([18, 70, 120], dtype=np.uint8)
        upper = np.array([45, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 1500:
            return False

        x, y, w, h = cv2.boundingRect(largest)
        cx = ox + x + w // 2
        cy = oy + y + h // 2
        self._tap_abs(
            frame,
            cx,
            cy,
            do_tap=not self.config.test_mode,
            debug_label="yellow_button_action",
        )
        print(f"[PHASE2] depart yellow_button tap=({cx},{cy}) area={int(area)}")
        return True

    def _extract_text_from_region(self, frame: np.ndarray, rect: RectPctConfig) -> str:
        crop, _, _ = self._crop_by_rect_pct(frame, rect)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25,
            5,
        )
        try:
            return pytesseract.image_to_string(thresh, config="--oem 3 --psm 7").strip()
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError("Tesseract OCR is not installed. Run `brew install tesseract`.") from exc

    def _extract_int_from_text(self, text: str) -> int | None:
        m = re.search(r"\d+", text)
        if not m:
            return None
        return int(m.group(0))

    def _extract_plane_name(self, frame: np.ndarray) -> str:
        rect = self.config.phase2.plane_name_region_pct
        if rect is None:
            return "unknown"
        crop, _, _ = self._crop_by_rect_pct(frame, rect)
        ch, cw = crop.shape[:2]
        if ch < 3 or cw < 3:
            return "unknown"
        # Use only the upper part where the ID text is rendered (avoid country/flag row noise).
        crop = crop[: max(1, int(ch * 0.62)), :]

        candidates = self._ocr_candidates(crop, allow_hyphen=True)
        if not candidates:
            return "unknown"

        if self.config.debug_logging:
            preview = [f"{t}:{c:.0f}" for t, c in candidates[:8]]
            print(f"[OCR] plane_name_candidates={preview}")

        # Merge common split pattern: "OS" + "1038" => "OS1038" and "OS-1038".
        alpha_tokens = [t for t, _ in candidates if re.fullmatch(r"[A-Z]{1,5}", t)]
        digit_tokens = [t for t, _ in candidates if re.fullmatch(r"\d{2,6}", t)]
        merged: list[tuple[str, float]] = []
        for a in alpha_tokens[:6]:
            for d in digit_tokens[:6]:
                merged.append((f"{a}{d}", 65.0))
                merged.append((f"{a}-{d}", 64.0))
        candidates.extend(merged)

        best_token = "unknown"
        best_score = float("-inf")
        seen: set[str] = set()
        for token, conf in candidates:
            if token in seen:
                continue
            seen.add(token)
            has_alpha = bool(re.search(r"[A-Z]", token))
            has_digit = bool(re.search(r"\d", token))
            score = conf
            score += min(len(token), 8) * 2.0
            if has_alpha and has_digit:
                score += 20.0
            if "-" in token:
                score += 2.0
            if len(token) <= 1:
                score -= 40.0
            if score > best_score:
                best_score = score
                best_token = token
        return best_token

    def _extract_plane_model(self, frame: np.ndarray) -> str:
        rect = self.config.phase2.plane_model_region_pct
        if rect is None:
            return "unknown"
        crop, _, _ = self._crop_by_rect_pct(frame, rect)
        ch, cw = crop.shape[:2]
        if ch < 3 or cw < 3:
            return "unknown"
        # Use upper band to focus on model text and avoid lower icons.
        crop = crop[: max(1, int(ch * 0.62)), :]

        candidates = self._ocr_candidates(crop, allow_hyphen=False)
        if not candidates:
            return "unknown"

        if self.config.debug_logging:
            preview = [f"{t}:{c:.0f}" for t, c in candidates[:8]]
            print(f"[OCR] plane_model_candidates={preview}")

        best_token = "unknown"
        best_score = float("-inf")
        seen: set[str] = set()
        for token, conf in candidates:
            if token in seen:
                continue
            seen.add(token)
            has_alpha = bool(re.search(r"[A-Z]", token))
            has_digit = bool(re.search(r"\d", token))
            score = conf
            score += min(len(token), 8) * 2.0
            if has_alpha and has_digit:
                score += 18.0
            if len(token) < 2:
                score -= 30.0
            if score > best_score:
                best_score = score
                best_token = token
        return best_token

    def _save_ocr_crop_debug(self, crop: np.ndarray, label: str, parsed: str) -> None:
        if not self.config.debug_logging:
            return
        path = self._save_debug(crop, f"ocr_{label}", force=True)
        if path:
            self._log_debug(f"[OCR] region={label} parsed='{parsed}' screenshot={path}")

    def _save_ocr_regions_overlay_debug(
        self,
        frame: np.ndarray,
        name_box: tuple[int, int, int, int],
        model_box: tuple[int, int, int, int],
        union_box: tuple[int, int, int, int],
        parsed_name: str,
        parsed_model: str,
        force: bool = False,
    ) -> None:
        if not force and not self.config.debug_logging:
            return
        debug = frame.copy()
        nx1, ny1, nx2, ny2 = name_box
        mx1, my1, mx2, my2 = model_box
        ux1, uy1, ux2, uy2 = union_box
        # Red = name/model configured regions, Cyan = union OCR region.
        cv2.rectangle(debug, (nx1, ny1), (nx2, ny2), (0, 0, 255), 2)
        cv2.rectangle(debug, (mx1, my1), (mx2, my2), (0, 0, 255), 2)
        cv2.rectangle(debug, (ux1, uy1), (ux2, uy2), (255, 255, 0), 2)
        cv2.putText(
            debug,
            f"name={parsed_name}",
            (max(0, nx1), max(15, ny1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            f"model={parsed_model}",
            (max(0, mx1), max(15, my1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        path = self._save_debug(debug, "ocr_plane_regions_overlay", force=True)
        if path:
            if force:
                self._log_warn(
                    f"ocr_identity_parse_failed name='{parsed_name}' model='{parsed_model}' overlay={path}",
                    frame=frame,
                )
            else:
                self._log_debug(f"[OCR] region_overlay screenshot={path}")

    def _extract_text_from_box(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        field: str,
        allow_hyphen: bool = True,
        prefer_mixed: bool = True,
    ) -> str:
        crop, _, _ = self._crop_by_box(frame, box)
        ch, cw = crop.shape[:2]
        if ch < 3 or cw < 3:
            return "unknown"

        # Fast path: minimal passes for low latency.
        candidates: list[tuple[str, float]] = []
        for ratio, bonus in ((0.64, 8.0), (1.0, 0.0)):
            hh = max(1, int(ch * ratio))
            sub = crop[:hh, :]
            for token, conf in self._ocr_candidates(sub, allow_hyphen=allow_hyphen, fast=True):
                candidates.append((token, conf + bonus))

        if self.config.debug_logging and candidates:
            preview = [f"{t}:{c:.0f}" for t, c in candidates[:10]]
            self._log_debug(f"[OCR] {field}_box_candidates={preview}")

        def _pick_best_token(pool: list[tuple[str, float]]) -> tuple[str, float]:
            best_token = "unknown"
            best_score = float("-inf")
            seen: set[str] = set()
            for token, conf in pool:
                if token in seen:
                    continue
                seen.add(token)
                has_alpha = bool(re.search(r"[A-Z]", token))
                has_digit = bool(re.search(r"\d", token))
                score = conf + min(len(token), 8) * 2.0
                if has_digit:
                    score += 8.0
                if prefer_mixed and has_alpha and has_digit:
                    score += 18.0
                if prefer_mixed and not has_digit:
                    score -= 12.0
                if len(token) <= 1:
                    score -= 40.0
                if score > best_score:
                    best_score = score
                    best_token = token
            return best_token, best_score

        best_token, best_score = _pick_best_token(candidates)

        # Adaptive retry: if fast pass is weak, rerun heavier OCR on the same anchored box.
        if best_token == "unknown" or (best_score < 38 and not re.search(r"\d", best_token)):
            slow_candidates: list[tuple[str, float]] = []
            for ratio, bonus in ((0.64, 8.0), (1.0, 0.0)):
                hh = max(1, int(ch * ratio))
                sub = crop[:hh, :]
                for token, conf in self._ocr_candidates(sub, allow_hyphen=allow_hyphen, fast=False):
                    slow_candidates.append((token, conf + bonus))
            if self.config.debug_logging and slow_candidates:
                preview = [f"{t}:{c:.0f}" for t, c in slow_candidates[:10]]
                self._log_debug(f"[OCR] {field}_slow_candidates={preview}")
            slow_best_token, slow_best_score = _pick_best_token(slow_candidates)
            if slow_best_score > best_score:
                best_token = slow_best_token

        return best_token

    def _extract_plane_identity(self, frame: np.ndarray) -> tuple[str, str]:
        name_box: tuple[int, int, int, int] | None = None
        model_box: tuple[int, int, int, int] | None = None

        anchor_template = self.config.phase2.plane_header_anchor_template
        if (
            anchor_template
            and self.config.phase2.plane_name_from_anchor_pct is not None
            and self.config.phase2.plane_model_from_anchor_pct is not None
            and anchor_template in self.templates
        ):
            anchor_match = self._find_preferred_lower_left_match_named(frame, anchor_template)
            if anchor_match is not None:
                ax, ay = self._resolve_match_center(frame, anchor_match)
                name_box = self._box_from_anchor_offset(
                    frame,
                    ax,
                    ay,
                    self.config.phase2.plane_name_from_anchor_pct,
                )
                model_box = self._box_from_anchor_offset(
                    frame,
                    ax,
                    ay,
                    self.config.phase2.plane_model_from_anchor_pct,
                )
                self._log_debug(
                    f"[OCR] anchor={anchor_template} center=({ax},{ay}) "
                    f"name_box={name_box} model_box={model_box}"
                )

        if name_box is None or model_box is None:
            name_rect = self.config.phase2.plane_name_region_pct
            model_rect = self.config.phase2.plane_model_region_pct
            if name_rect is None or model_rect is None:
                return self._extract_plane_name(frame), self._extract_plane_model(frame)
            name_box = self._box_from_rect_pct(frame, name_rect)
            model_box = self._box_from_rect_pct(frame, model_rect)

        nx1, ny1, nx2, ny2 = name_box
        mx1, my1, mx2, my2 = model_box

        ux1 = min(nx1, mx1)
        uy1 = min(ny1, my1)
        ux2 = max(nx2, mx2)
        uy2 = max(ny2, my2)
        if ux2 <= ux1 or uy2 <= uy1:
            return self._extract_plane_name(frame), self._extract_plane_model(frame)

        name = self._extract_text_from_box(
            frame,
            (nx1, ny1, nx2, ny2),
            field="plane_name",
            allow_hyphen=True,
            prefer_mixed=True,
        )
        model = self._extract_text_from_box(
            frame,
            (mx1, my1, mx2, my2),
            field="plane_model",
            allow_hyphen=True,
            prefer_mixed=True,
        )

        need_alt_name = name == "unknown" or not re.search(r"\d", name)
        need_alt_model = model == "unknown" or not re.search(r"\d", model)
        if need_alt_name:
            alt_name = self._extract_plane_name(frame)
            if name == "unknown" or (not re.search(r"\d", name) and re.search(r"\d", alt_name)):
                name = alt_name
        if need_alt_model:
            alt_model = self._extract_plane_model(frame)
            if model == "unknown" or (not re.search(r"\d", model) and re.search(r"\d", alt_model)):
                model = alt_model

        ocr_failed = name == "unknown" or model == "unknown"
        self._save_ocr_regions_overlay_debug(
            frame,
            (nx1, ny1, nx2, ny2),
            (mx1, my1, mx2, my2),
            (ux1, uy1, ux2, uy2),
            name,
            model,
            force=ocr_failed,
        )
        union_crop, _, _ = self._crop_by_box(frame, (ux1, uy1, ux2, uy2))
        self._save_ocr_crop_debug(union_crop, "plane_identity_union", parsed=f"name={name} model={model}")

        # Also export exact configured per-field regions for side-by-side tuning.
        name_crop, _, _ = self._crop_by_box(frame, (nx1, ny1, nx2, ny2))
        model_crop, _, _ = self._crop_by_box(frame, (mx1, my1, mx2, my2))
        self._save_ocr_crop_debug(name_crop, "plane_name_region", parsed=name)
        self._save_ocr_crop_debug(model_crop, "plane_model_region", parsed=model)
        return name, model

    def _ocr_candidates(
        self, crop: np.ndarray, allow_hyphen: bool, fast: bool = False
    ) -> list[tuple[str, float]]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        enlarged = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.adaptiveThreshold(
            enlarged,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        inverted = cv2.bitwise_not(thresh)
        otsu = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants = [thresh, enlarged] if fast else [enlarged, thresh, inverted, otsu]
        psms = (7, 8) if fast else (6, 7, 8, 11)
        whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" if allow_hyphen else "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        norm_pattern = r"[^A-Z0-9-]" if allow_hyphen else r"[^A-Z0-9]"

        raw_candidates: list[tuple[str, float]] = []
        try:
            for img in variants:
                for psm in psms:
                    cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"
                    data = pytesseract.image_to_data(
                        img,
                        output_type=pytesseract.Output.DICT,
                        config=cfg,
                    )
                    for i, raw in enumerate(data.get("text", [])):
                        txt = raw.strip().upper()
                        if not txt:
                            continue
                        txt = re.sub(norm_pattern, "", txt)
                        if not txt:
                            continue
                        try:
                            conf = float(data["conf"][i])
                        except ValueError:
                            conf = 0.0
                        if conf < 20:
                            continue
                        raw_candidates.append((txt, conf))
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError("Tesseract OCR is not installed. Run `brew install tesseract`.") from exc

        by_token: dict[str, float] = {}
        for token, conf in raw_candidates:
            prev = by_token.get(token)
            if prev is None or conf > prev:
                by_token[token] = conf
        ranked = sorted(by_token.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def _extract_crew_counts(self, frame: np.ndarray) -> tuple[int | None, int | None]:
        available_rect = self.config.phase2.crew_available_region_pct
        required_rect = self.config.phase2.crew_required_region_pct
        if available_rect is None or required_rect is None:
            return None, None

        available_text = self._extract_text_from_region(frame, available_rect)
        required_text = self._extract_text_from_region(frame, required_rect)
        return self._extract_int_from_text(available_text), self._extract_int_from_text(required_text)

    def _record_plane_action(
        self,
        plane_name: str,
        category: str,
        action: str,
        plane_model: str | None = None,
    ) -> None:
        now_ms = int(time.time() * 1000)
        name = plane_name if plane_name else "unknown"
        model = plane_model if plane_model else self.current_plane_model
        key = f"{name}|{model}"
        self.phase2_plane_memory[key] = PlaneRecord(
            name=name,
            category=category,
            last_action=action,
            last_seen_epoch_ms=now_ms,
        )
        print(
            f"{ANSI_GREEN}[PLANE]{ANSI_RESET} "
            f"name='{name}' model='{model}' category={category} action={action}"
        )
        normalized = action.lower()
        if normalized.startswith("skip_"):
            return
        is_failure = (
            "not_started" in normalized
            or "not_found" in normalized
            or "no_action" in normalized
            or "not_enough" in normalized
        )
        if is_failure:
            self._log_fail(
                f"plane_action_failure name='{name}' model='{model}' category={category} action={action}"
            )

    def _handle_processing(self, frame: np.ndarray, plane_name: str, plane_model: str) -> bool:
        # Edge case: finish handling is available.
        if self._click_template_named(
            frame,
            self.config.phase2.processing_finish_handling_template,
            "processing_finish_handling",
        ):
            self._record_plane_action(plane_name, "processing", "finish_handling", plane_model)
            return True

        # Case 1: claim rewards is available.
        if self._click_template_named(
            frame,
            self.config.phase2.processing_claim_rewards_template,
            "processing_claim_rewards",
        ):
            popup_templates = [
                self.config.phase2.processing_claim_rewards_and_upgrade_popup_template,
                self.config.phase2.processing_claim_reward_popup_template,
            ]
            popup_clicked = False
            for attempt in range(2):
                if popup_clicked:
                    break
                self._sleep(0.12)
                popup_frame = self._capture_frame()
                for popup_tmpl in popup_templates:
                    if not popup_tmpl:
                        continue
                    if self._click_template_named(
                        popup_frame,
                        popup_tmpl,
                        f"processing_claim_popup_{attempt}",
                    ):
                        popup_clicked = True
                        break
            self._record_plane_action(
                plane_name,
                "processing",
                f"claim_rewards popup_confirm={popup_clicked}",
                plane_model,
            )
            return True

        frame = self._capture_frame()
        assign_crew_disabled = self._has_template_named(
            frame,
            self.config.phase2.processing_assign_crew_disabled_template,
        )
        if not assign_crew_disabled:
            self._record_plane_action(plane_name, "processing", "skip_no_processing_action", plane_model)
            return False

        # Case 2: assign crew disabled -> click add first -> if not enough before toggle, skip -> then toggle -> start handling.
        toggled = False

        add_clicks = 0
        add_loop_reason = "unknown"
        for _ in range(self.config.phase2.processing_max_add_clicks):
            frame = self._capture_frame()
            if self._has_template_named(
                frame,
                self.config.phase2.processing_not_enough_message_template,
            ):
                add_loop_reason = "not_enough_before_toggle"
                break
            add_match = self._find_rightmost_match_named(
                frame, self.config.phase2.processing_add_enabled_template
            )
            has_enabled = add_match is not None
            has_disabled = self._has_template_named(
                frame,
                self.config.phase2.processing_add_disabled_template,
            )

            if has_enabled:
                if self.processing_add_button_anchor_px is None and add_match is not None:
                    self._set_processing_add_anchor_from_match(frame, add_match)
                if not self._click_processing_add_anchor(frame):
                    add_loop_reason = "add_anchor_not_set"
                    break
                add_clicks += 1
                add_loop_reason = "clicked_until_limit_or_disabled"
                self._sleep(0.08)
                check_frame = self._capture_frame()
                if (
                    self._has_template_named(
                        check_frame,
                        self.config.phase2.processing_not_enough_message_template,
                    )
                    and self._has_template_named(
                        check_frame,
                        self.config.phase2.processing_add_enabled_template,
                    )
                ):
                    add_loop_reason = "not_enough_after_add"
                    break
                continue

            if has_disabled:
                add_loop_reason = "already_disabled"
                break

            if self.processing_add_button_anchor_px is not None:
                if not self._click_processing_add_anchor(frame):
                    add_loop_reason = "add_anchor_not_set"
                    break
                add_clicks += 1
                add_loop_reason = "clicked_by_anchor_without_template"
                self._sleep(0.08)
                continue

            add_loop_reason = "add_button_not_found_and_no_anchor"
            break

        if add_loop_reason == "not_enough_after_add":
            self._record_plane_action(
                plane_name,
                "processing",
                f"skip_not_enough_after_add add_clicks={add_clicks} toggled={toggled}",
                plane_model,
            )
            return False

        frame = self._capture_frame()
        if self._has_template_named(
            frame,
            self.config.phase2.processing_not_enough_message_template,
        ):
            self._record_plane_action(
                plane_name,
                "processing",
                f"skip_not_enough_before_toggle add_clicks={add_clicks} toggled={toggled}",
                plane_model,
            )
            return False

        frame = self._capture_frame()
        toggled = self._click_template_named(
            frame,
            self.config.phase2.processing_toggle_button_template,
            "processing_toggle",
        )
        if toggled:
            self._sleep(0.08)

        frame = self._capture_frame()
        started = self._click_template_named(
            frame,
            self.config.phase2.processing_start_handling_template,
            "processing_start_handling",
        )

        if started:
            self._record_plane_action(
                plane_name,
                "processing",
                f"assign_crew_started add_clicks={add_clicks} toggled={toggled} reason={add_loop_reason}",
                plane_model,
            )
            return True

        self._record_plane_action(
            plane_name,
            "processing",
            f"assign_crew_not_started add_clicks={add_clicks} toggled={toggled} reason={add_loop_reason}",
            plane_model,
        )
        if add_clicks == 0:
            add_enabled_conf = self._best_conf_for_template(
                self._capture_frame(),
                self.config.phase2.processing_add_enabled_template,
            )
            add_disabled_conf = self._best_conf_for_template(
                self._capture_frame(),
                self.config.phase2.processing_add_disabled_template,
            )
            print(
                "[PHASE2] processing add_debug "
                f"enabled_conf={add_enabled_conf} disabled_conf={add_disabled_conf}"
            )
        return add_clicks > 0 or toggled

    def _handle_landing(self, frame: np.ndarray, plane_name: str, plane_model: str) -> bool:
        # Case 1: stand selection flow (priority to avoid false clear-to-land hits).
        frame = self._capture_frame()
        stand_selection_needed = (
            self._has_template_named(frame, self.config.phase2.landing_select_stand_disabled_template)
            or self._has_template_named(frame, self.config.phase2.landing_empty_stand_card_template)
        )

        if stand_selection_needed:
            empty_clicked = self._click_leftmost_template_named(
                frame,
                self.config.phase2.landing_empty_stand_card_template,
                "landing_empty_stand",
            )
            if not empty_clicked:
                frame = self._capture_frame()
                empty_clicked = self._click_leftmost_template_named(
                    frame,
                    self.config.phase2.landing_empty_stand_card_template,
                    "landing_empty_stand_retry",
                )

            if not empty_clicked:
                self._record_plane_action(plane_name, "landing", "stand_selection_no_empty_card", plane_model)
                return False

            self._sleep(0.15)
            frame = self._capture_frame()
            if self._click_template_named(
                frame,
                self.config.phase2.landing_confirm_button_template,
                "landing_confirm",
            ):
                self._record_plane_action(plane_name, "landing", "select_stand_confirm", plane_model)
                return True

            self._record_plane_action(plane_name, "landing", "stand_selected_confirm_not_found", plane_model)
            return False

        # Case 2: direct clearance is available.
        if self._click_template_named(
            frame,
            self.config.phase2.landing_clear_to_land_template,
            "landing_clear_to_land",
        ):
            self._record_plane_action(plane_name, "landing", "clear_to_land", plane_model)
            return True

        # No landing action state recognized.
        clear_conf = self._best_conf_for_template(
            frame,
            self.config.phase2.landing_clear_to_land_template,
        )
        select_disabled_conf = self._best_conf_for_template(
            frame,
            self.config.phase2.landing_select_stand_disabled_template,
        )
        empty_conf = self._best_conf_for_template(
            frame,
            self.config.phase2.landing_empty_stand_card_template,
        )
        # Fallback: lower-left yellow action button, same position class as depart.
        if (
            self.config.phase2.depart_yellow_button_region_pct is not None
            and self._click_yellow_button_in_region(frame, self.config.phase2.depart_yellow_button_region_pct)
        ):
            self._record_plane_action(plane_name, "landing", "clear_to_land_yellow_fallback", plane_model)
            return True

        self._record_plane_action(
            plane_name,
            "landing",
            (
                "skip_no_landing_action "
                f"clear_conf={clear_conf} "
                f"select_disabled_conf={select_disabled_conf} "
                f"empty_conf={empty_conf}"
            ),
            plane_model,
        )
        return False

    def _handle_depart(self, frame: np.ndarray, plane_name: str, plane_model: str) -> bool:
        depart_tmpl = self.config.phase2.depart_execute_button_template
        if depart_tmpl and depart_tmpl in self.templates:
            if self._click_template_named(frame, depart_tmpl, "depart_execute"):
                self._record_plane_action(plane_name, "depart", "execute_depart", plane_model)
                return True

        if self.config.phase2.depart_yellow_button_region_pct is not None:
            if self._click_yellow_button_in_region(
                frame,
                self.config.phase2.depart_yellow_button_region_pct,
            ):
                self._record_plane_action(plane_name, "depart", "execute_depart_yellow_button", plane_model)
                return True

        self._record_plane_action(plane_name, "depart", "skip_no_depart_action", plane_model)
        return False

    def _handle_category(
        self,
        frame: np.ndarray,
        category_name: str,
        category_cfg: Phase2CategoryConfig,
    ) -> bool:
        if not self._click_template_named(frame, category_cfg.tab_template, f"{category_name}_tab"):
            return False

        attempts = self._estimate_cards_per_category(self._capture_frame(), category_name)
        attempts = max(1, attempts)
        for _ in range(attempts):
            frame = self._capture_frame()
            if not self._select_next_card(frame, category_name, dry_run=False, log_prefix="[PHASE2]"):
                break
            self._sleep(0.1)
            frame = self._capture_frame()
            if self.config.phase2.parse_plane_info:
                plane_name, plane_model = self._extract_plane_identity(frame)
            else:
                plane_name, plane_model = "unknown", "unknown"
            self.current_plane_name = plane_name
            self.current_plane_model = plane_model

            if category_name == "processing" and self._handle_processing(frame, plane_name, plane_model):
                return True
            if category_name == "landing" and self._handle_landing(frame, plane_name, plane_model):
                return True
            if category_name == "depart" and self._handle_depart(frame, plane_name, plane_model):
                return True

        self._log_debug(f"[PHASE2] no actionable {category_name} card in current sweep")
        return False

    def _incorrect_enabled_template_names(self) -> list[str]:
        names: list[str] = []
        for name in self.config.phase2.incorrect_enabled_templates:
            if name and name not in names:
                names.append(name)
        # Auto-pick templates named like button_enabled_incorrect_1, _2, ...
        for name in self.templates.keys():
            if re.fullmatch(r"button_enabled_incorrect_\d+", name):
                if name not in names:
                    names.append(name)
        return names

    def _clear_incorrect_enabled_buttons(self) -> bool:
        names = self._incorrect_enabled_template_names()
        if not names:
            return False

        clicked_any = False
        for _ in range(self.config.phase2.incorrect_enabled_max_passes):
            clicked_in_pass = False
            frame = self._capture_frame()
            guard_y = self._actionable_filter_guard_y(frame)
            for name in names:
                if name not in self.templates:
                    continue
                candidate = self._match_template_named(frame, name)
                if candidate is None:
                    continue
                _, cy = self._resolve_match_center(frame, candidate)
                if guard_y is not None and cy > guard_y:
                    self._log_debug(
                        f"[PHASE2] skip incorrect_enabled below guard template={name} tap_y={cy} guard_y={guard_y}"
                    )
                    continue
                if self._click_template_named(
                    frame,
                    name,
                    f"incorrect_enabled_{name}",
                    allow_cache=False,
                ):
                    clicked_in_pass = True
                    clicked_any = True
                    self._sleep(self.config.phase2.inter_click_delay_sec)
                    frame = self._capture_frame()
            if not clicked_in_pass:
                break
        if clicked_any:
            self._log_debug("[PHASE2] corrected one or more incorrect-enabled buttons")
        return clicked_any

    def _phase2_setup(self, frame: np.ndarray) -> bool:
        action_taken = False

        if not self.phase2_grey_enabled:
            if self.config.phase2.grey_mode_template in self.templates:
                if self._click_template_named(frame, self.config.phase2.grey_mode_template, "grey_mode"):
                    self.phase2_grey_enabled = True
                    action_taken = True
                    self._sleep(self.config.phase2.inter_click_delay_sec)
                    frame = self._capture_frame()
            else:
                self._warn_missing_template_once(self.config.phase2.grey_mode_template)
                self.phase2_grey_enabled = True

        if not self.phase2_filter_enabled:
            if (
                self.config.phase2.filter_button_template in self.templates
                and self._click_template_named(
                    frame,
                    self.config.phase2.filter_button_template,
                    "filter_button",
                )
            ):
                action_taken = True
                self._sleep(self.config.phase2.inter_click_delay_sec)
                frame = self._capture_frame()
            elif self.config.phase2.filter_button_template not in self.templates:
                self._warn_missing_template_once(self.config.phase2.filter_button_template)

            if self.config.phase2.actionable_filter_template in self.templates:
                if self._click_template_named(frame, self.config.phase2.actionable_filter_template, "actionable_filter"):
                    self.phase2_filter_enabled = True
                    action_taken = True
                    self._sleep(self.config.phase2.inter_click_delay_sec)
            else:
                self._warn_missing_template_once(self.config.phase2.actionable_filter_template)
                self.phase2_filter_enabled = True

        return action_taken

    def _run_phase2_cycle(self, frame: np.ndarray) -> bool:
        if not self.config.phase2.enabled:
            return False

        if self._phase2_setup(frame):
            return True

        if self.config.test_mode:
            any_action = False
            categories: list[tuple[str, Phase2CategoryConfig]] = [
                ("processing", self.config.phase2.processing),
                ("landing", self.config.phase2.landing),
                ("depart", self.config.phase2.depart),
            ]

            for name, cfg in categories:
                any_action = self._clear_incorrect_enabled_buttons() or any_action
                frame = self._capture_frame()
                tab_clicked = self._click_template_named(
                    frame,
                    cfg.tab_template,
                    f"test_mode_{name}_tab",
                    dry_run=False,
                )
                card_clicked = False
                if tab_clicked:
                    self._sleep(self.config.phase2.inter_click_delay_sec)
                    card_clicked = self._test_mode_select_next_card(
                        self._capture_frame(),
                        name,
                        dry_run=False,
                    )
                    if card_clicked and self.config.phase2.parse_plane_info:
                        detail_frame = self._capture_frame()
                        test_plane_name, test_plane_model = self._extract_plane_identity(detail_frame)
                        print(
                            f"{ANSI_GREEN}[PLANE-TEST]{ANSI_RESET} "
                            f"name='{test_plane_name}' model='{test_plane_model}' category={name}"
                        )
                print(
                    f"[PHASE2-TEST] category={name} tab_clicked={tab_clicked} card_clicked={card_clicked}"
                )
                any_action = any_action or tab_clicked or card_clicked
                self._sleep(self.config.phase2.inter_click_delay_sec)

            return any_action

        any_action = False
        any_action = self._clear_incorrect_enabled_buttons() or any_action
        frame = self._capture_frame()
        if self._handle_category(frame, "processing", self.config.phase2.processing):
            return True
        self._sleep(self.config.phase2.inter_click_delay_sec)

        any_action = self._clear_incorrect_enabled_buttons() or any_action
        frame = self._capture_frame()
        if self._handle_category(frame, "landing", self.config.phase2.landing):
            return True
        self._sleep(self.config.phase2.inter_click_delay_sec)

        any_action = self._clear_incorrect_enabled_buttons() or any_action
        frame = self._capture_frame()
        if self._handle_category(frame, "depart", self.config.phase2.depart):
            return True

        if not any_action:
            print("[PHASE2] no action performed in this cycle")
        return any_action

    def _estimate_cards_per_category(self, frame: np.ndarray, category: str) -> int:
        region = self.config.phase2.card_list_region_pct
        if region is None:
            return 1
        if self.config.phase2.card_key_icon_template:
            icon_matches = self._find_template_matches_in_region(
                frame,
                self.config.phase2.card_key_icon_template,
                region,
            )
            if icon_matches:
                return len(icon_matches)
        cards_per = int((1.0 - self.config.phase2.card_start_y_pct) / self.config.phase2.card_step_y_pct) + 1
        return max(1, cards_per)

    def _select_next_card(
        self,
        frame: np.ndarray,
        category: str,
        dry_run: bool = False,
        log_prefix: str = "[PHASE2-TEST]",
    ) -> bool:
        region = self.config.phase2.card_list_region_pct
        if region is None:
            self._warn_missing_template_once("phase2.card_list_region_pct")
            return False

        if self.config.phase2.card_key_icon_template:
            icon_matches = self._find_template_matches_in_region(
                frame,
                self.config.phase2.card_key_icon_template,
                region,
            )
            if icon_matches:
                cursor = self.phase2_test_mode_card_index.get(category, 0)
                slot = cursor % len(icon_matches)
                picked = icon_matches[slot]
                if dry_run:
                    tx, ty = self._resolve_match_center(frame, picked)
                else:
                    tx, ty = self._tap_match_center(
                        frame,
                        picked,
                        debug_label=f"card_select_icon_{category}",
                    )
                self.phase2_test_mode_card_index[category] = cursor + 1
                self._sleep(0.1)
                print(
                    f"{log_prefix} select_card category={category} mode=icon "
                    f"slot={slot}/{len(icon_matches)} tap=({tx},{ty}) "
                    f"icon_conf={picked.confidence:.3f}"
                )
                return True
            # Icon mode is configured; no icons means no visible actionable cards.
            return False

        # Auto-compute visible slots from start+step geometry.
        cards_per = int((1.0 - self.config.phase2.card_start_y_pct) / self.config.phase2.card_step_y_pct) + 1
        cards_per = max(1, cards_per)
        cursor = self.phase2_test_mode_card_index.get(category, 0)
        slot = cursor % cards_per
        local_y = (
            self.config.phase2.card_start_y_pct
            + slot * self.config.phase2.card_step_y_pct
        )
        if local_y > 0.98:
            local_y = 0.98

        x_pct = region.x + region.w * self.config.phase2.card_anchor_x_pct
        y_pct = region.y + region.h * local_y
        x, y = self._to_abs_xy(frame, x_pct, y_pct)
        tx, ty = self._tap_abs(
            frame,
            x,
            y,
            do_tap=not dry_run,
            debug_label=f"card_select_grid_{category}",
        )
        self.phase2_test_mode_card_index[category] = cursor + 1
        self._sleep(0.1)
        print(
            f"{log_prefix} select_card category={category} mode=grid slot={slot} "
            f"tap=({tx},{ty}) x_pct={x_pct:.3f} y_pct={y_pct:.3f}"
        )
        return True

    def _test_mode_select_next_card(self, frame: np.ndarray, category: str, dry_run: bool = False) -> bool:
        return self._select_next_card(
            frame,
            category,
            dry_run=dry_run,
            log_prefix="[PHASE2-TEST]",
        )

    def _find_template_matches_in_region(
        self,
        frame: np.ndarray,
        template_name: str,
        region: RectPctConfig,
    ) -> list[MatchResult]:
        tmpl = self.templates.get(template_name)
        if tmpl is None:
            self._warn_missing_template_once(template_name)
            return []

        crop, ox, oy = self._crop_by_rect_pct(frame, region)
        ch, cw = crop.shape[:2]
        th, tw = tmpl.image.shape[:2]
        if ch < th or cw < tw:
            return []

        result = cv2.matchTemplate(crop, tmpl.image, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= tmpl.threshold)

        candidates: list[MatchResult] = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            conf = float(result[y, x])
            candidates.append(
                MatchResult(
                    name=template_name,
                    confidence=conf,
                    x=ox + int(x),
                    y=oy + int(y),
                    w=int(tw),
                    h=int(th),
                )
            )

        candidates.sort(key=lambda m: m.confidence, reverse=True)

        selected: list[MatchResult] = []
        min_dist = max(tw, th) * 0.7
        min_dist_sq = min_dist * min_dist
        for m in candidates:
            cx = m.x + m.w / 2.0
            cy = m.y + m.h / 2.0
            keep = True
            for s in selected:
                sx = s.x + s.w / 2.0
                sy = s.y + s.h / 2.0
                dx = cx - sx
                dy = cy - sy
                if (dx * dx + dy * dy) < min_dist_sq:
                    keep = False
                    break
            if keep:
                selected.append(m)

        selected.sort(key=lambda m: (m.y, m.x))
        return selected

    def step(self) -> bool:
        frame = self._capture_frame()

        if self.startup_index < len(self.config.startup_flow):
            self._next_sleep_override_sec = None
            return self._run_startup_flow_step(frame)

        if not self.phase2_started:
            self.phase2_started = True
            if self.config.phase2.post_start_delay_sec > 0:
                self._sleep(self.config.phase2.post_start_delay_sec)

        action_performed = self._run_phase2_cycle(frame)
        self._next_sleep_override_sec = (
            self.config.phase2.action_cycle_delay_sec
            if action_performed
            else self.config.phase2.idle_cycle_delay_sec
        )
        return action_performed

    def run(self) -> None:
        print("[INFO] Starting bot loop. Press Ctrl+C to stop.")
        while True:
            try:
                self.step()
            except Exception as exc:
                self._log_error(str(exc))

            if self._next_sleep_override_sec is not None:
                sleep_time = self._next_sleep_override_sec
            else:
                sleep_time = self.config.loop_interval_sec + random.uniform(0, self.config.jitter_sec)
            self._sleep(max(0.05, sleep_time))
