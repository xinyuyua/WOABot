from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from .adb_client import AdbClient
from .config import BotConfig, FlowStepConfig, RectPctConfig
from .detector import MatchResult, load_template, match_template


@dataclass
class GameBot:
    config: BotConfig

    def __post_init__(self) -> None:
        self.adb = AdbClient(self.config.adb_path, self.config.serial)
        self.templates = {
            t.name: load_template(path=t.path, name=t.name, threshold=t.threshold)
            for t in self.config.templates
        }
        self.template_actions = {t.name: t.action.type for t in self.config.templates}
        self.startup_index = 0
        self.airport_scroll_count = 0
        Path(self.config.screenshot_dir).mkdir(parents=True, exist_ok=True)

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

    def _tap_match_center(self, match: MatchResult) -> tuple[int, int]:
        x = match.x + (match.w // 2)
        y = match.y + (match.h // 2)
        self.adb.tap(x, y)
        return x, y

    def _find_match(self, frame: np.ndarray, template_name: str) -> MatchResult | None:
        tmpl = self.templates.get(template_name)
        if tmpl is None:
            raise ValueError(f"Template not defined in config.templates: {template_name}")
        return match_template(frame, tmpl)

    def _save_debug(self, frame: np.ndarray, label: str) -> None:
        if not self.config.save_debug_screenshots:
            return
        ts = int(time.time() * 1000)
        out = Path(self.config.screenshot_dir) / f"{label}_{ts}.png"
        cv2.imwrite(str(out), frame)

    def _to_abs_xy(self, frame: np.ndarray, x_pct: float, y_pct: float) -> tuple[int, int]:
        height, width = frame.shape[:2]
        x = int(round(x_pct * width))
        y = int(round(y_pct * height))
        return x, y

    def _resolve_swipe(self, frame: np.ndarray, step: FlowStepConfig) -> tuple[int, int, int, int, int]:
        if step.swipe_pct is not None:
            x1, y1 = self._to_abs_xy(frame, step.swipe_pct.x1, step.swipe_pct.y1)
            x2, y2 = self._to_abs_xy(frame, step.swipe_pct.x2, step.swipe_pct.y2)
            return x1, y1, x2, y2, step.swipe_pct.duration_ms

        if step.swipe is not None:
            return (
                step.swipe.x1,
                step.swipe.y1,
                step.swipe.x2,
                step.swipe.y2,
                step.swipe.duration_ms,
            )

        raise ValueError("Step requires swipe config (`swipe` or `swipe_pct`)")

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

    def _run_click_template_step(self, frame: np.ndarray, step: FlowStepConfig) -> bool:
        match = self._find_match(frame, step.template)
        if not match:
            print(f"[FLOW] Waiting for template: {step.template}")
            return False

        action_type = self.template_actions.get(step.template, "tap_center")
        if action_type != "tap_center":
            raise ValueError(
                f"Unsupported action type `{action_type}` for template `{step.template}`"
            )

        x, y = self._tap_match_center(match)
        print(
            f"[FLOW] click_template hit={match.name} conf={match.confidence:.3f} tap=({x},{y})"
        )
        self._save_debug(frame, f"flow_hit_{match.name}")
        self.startup_index += 1
        return True

    def _run_pick_airport_step(self, frame: np.ndarray, step: FlowStepConfig) -> bool:
        match = self._find_match(frame, step.template)
        if match:
            x, y = self._tap_match_center(match)
            print(
                f"[FLOW] pick_airport found={match.name} conf={match.confidence:.3f} tap=({x},{y})"
            )
            self._save_debug(frame, f"airport_hit_{match.name}")
            self.airport_scroll_count = 0
            self.startup_index += 1
            return True

        if self.airport_scroll_count >= step.max_scrolls:
            raise RuntimeError(
                f"Airport template `{step.template}` not found after {step.max_scrolls} scrolls"
            )

        x1, y1, x2, y2, duration_ms = self._resolve_swipe(frame, step)
        self.adb.swipe(x1, y1, x2, y2, duration_ms)
        self.airport_scroll_count += 1
        print(
            f"[FLOW] pick_airport not found, scroll {self.airport_scroll_count}/{step.max_scrolls}"
        )
        return False

    def _find_text_center(
        self,
        frame: np.ndarray,
        target_text: str,
        min_confidence: int,
        ocr_region_pct: RectPctConfig | None = None,
    ) -> tuple[int, int, float, str] | None:
        offset_x, offset_y = 0, 0
        search_frame = frame
        if ocr_region_pct is not None:
            search_frame, offset_x, offset_y = self._crop_by_rect_pct(frame, ocr_region_pct)

        gray = cv2.cvtColor(search_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        try:
            data = pytesseract.image_to_data(
                gray, output_type=pytesseract.Output.DICT, config="--psm 6"
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract OCR is not installed. Run `brew install tesseract`."
            ) from exc

        target = target_text.lower()
        best: tuple[int, int, float, str] | None = None

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
            if target not in text.lower():
                continue

            x = offset_x + int(data["left"][i]) + int(data["width"][i]) // 2
            y = offset_y + int(data["top"][i]) + int(data["height"][i]) // 2
            candidate = (x, y, conf, text)
            if best is None or conf > best[2]:
                best = candidate

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
            self.adb.tap(x, y)
            print(
                f"[FLOW] pick_airport_text found='{text}' conf={conf:.1f} tap=({x},{y})"
            )
            self._save_debug(frame, "airport_text_hit")
            self.airport_scroll_count = 0
            self.startup_index += 1
            return True

        if self.airport_scroll_count >= step.max_scrolls:
            raise RuntimeError(
                f"Airport text `{step.target_text}` not found after {step.max_scrolls} scrolls"
            )

        x1, y1, x2, y2, duration_ms = self._resolve_swipe(frame, step)
        self.adb.swipe(x1, y1, x2, y2, duration_ms)
        self.airport_scroll_count += 1
        print(
            "[FLOW] pick_airport_text not found "
            f"('{step.target_text}'), scroll {self.airport_scroll_count}/{step.max_scrolls}"
        )
        return False

    def _run_startup_flow_step(self, frame: np.ndarray) -> bool:
        if self.startup_index >= len(self.config.startup_flow):
            return False

        step = self.config.startup_flow[self.startup_index]
        if step.type == "click_template":
            return self._run_click_template_step(frame, step)
        if step.type == "pick_airport":
            return self._run_pick_airport_step(frame, step)
        if step.type == "pick_airport_text":
            return self._run_pick_airport_text_step(frame, step)

        raise ValueError(f"Unsupported startup step type: {step.type}")

    def step(self) -> bool:
        frame = self._decode_frame(self.adb.screenshot_png_bytes())

        if self.startup_index < len(self.config.startup_flow):
            return self._run_startup_flow_step(frame)

        print("[INFO] Startup flow complete. Add post-start logic in step().")
        return False

    def run(self) -> None:
        print("[INFO] Starting bot loop. Press Ctrl+C to stop.")
        while True:
            try:
                self.step()
            except Exception as exc:
                print(f"[ERROR] {exc}")

            sleep_time = self.config.loop_interval_sec + random.uniform(
                0, self.config.jitter_sec
            )
            time.sleep(max(0.05, sleep_time))
