from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .adb_client import AdbClient
from .config import BotConfig, FlowStepConfig
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

        swipe = step.swipe
        if swipe is None:
            raise ValueError("pick_airport step missing swipe config")

        self.adb.swipe(swipe.x1, swipe.y1, swipe.x2, swipe.y2, swipe.duration_ms)
        self.airport_scroll_count += 1
        print(
            f"[FLOW] pick_airport not found, scroll {self.airport_scroll_count}/{step.max_scrolls}"
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
