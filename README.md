# ADB Game Bot Skeleton (macOS + Android Emulator)

Minimal starter bot:
- Captures emulator screen via `adb exec-out screencap -p`
- Detects UI elements with OpenCV template matching
- Executes taps/swipes via `adb shell input ...`
- Runs an ordered startup flow

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure emulator is running and visible in `adb devices`.

## 2) Configure startup flow

Edit `/Users/xinyuyuan/workspace/CodeX/WOAbot/config.yaml`.

Current startup flow:
1. `click_template` -> `start_button`
2. `pick_airport` -> `airport_sfo` (scroll list until found)
3. `click_template` -> `play_button`

`pick_airport` requires:
- airport template image
- swipe coordinates for the list area
- max scroll count

## 3) Add template images

Put cropped images into `/Users/xinyuyuan/workspace/CodeX/WOAbot/templates/`:
- `start_button.png`
- `airport_sfo.png`
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
- `pick_airport.swipe` coordinates and `max_scrolls`
- `loop_interval_sec` and `jitter_sec`

## Notes

- After startup flow completes, `step()` currently logs completion only.
- Respect game terms of service and account safety.
