# ADB Game Bot Skeleton (macOS + Android Emulator)

Minimal starter bot:
- Captures emulator screen via `adb exec-out screencap -p`
- Detects UI elements with OpenCV template matching
- Detects airport text with OCR (Tesseract)
- Executes taps/swipes via `adb shell input ...`
- Runs an ordered startup flow

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract
```

Make sure emulator is running and visible in `adb devices`.

## 2) Configure startup flow

Edit `/Users/xinyuyuan/workspace/CodeX/WOAbot/config.yaml`.

Current startup flow:
1. `click_template` -> `start_button`
2. `pick_airport_text` -> `INN` (scroll list until found by OCR)
3. `click_template` -> `play_button`

`pick_airport_text` requires:
- `target_text` keyword (for example `INN`)
- `min_ocr_confidence`
- `ocr_region_pct` (OCR scan area, 0-1 relative to screen)
- `swipe_pct` (gesture points, 0-1 relative to screen)
- max scroll count

## 3) Add template images

Put cropped images into `/Users/xinyuyuan/workspace/CodeX/WOAbot/templates/`:
- `start_button.png`
- `play_button.png`

Tips:
- Crop tightly around unique UI element/text.
- Keep emulator resolution fixed so coordinates and templates stay stable.

## 4) Run

```bash
python /Users/xinyuyuan/workspace/CodeX/WOAbot/run_bot.py --config /Users/xinyuyuan/workspace/CodeX/WOAbot/config.yaml
```

Stop with `Ctrl+C`.

## 5) Tune

- Template threshold: `0.85` to `0.98`
- `pick_airport_text.ocr_region_pct` and `pick_airport_text.swipe_pct`
- `loop_interval_sec` and `jitter_sec`

Example for right-side floating scrollbar lane:
- `ocr_region_pct`: `x: 0.78, y: 0.12, w: 0.20, h: 0.78`
- `swipe_pct`: `x1: 0.88, y1: 0.86, x2: 0.88, y2: 0.28`

## Notes

- After startup flow completes, `step()` currently logs completion only.
- Respect game terms of service and account safety.
