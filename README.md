# WOABot (macOS + Android Emulator via ADB)

Automation bot for World of Airports running on an Android emulator.

High-level flow:
- Phase 1: enter game (`start -> select airport -> play`)
- Phase 2: loop categories (`processing -> landing -> depart`)

## Requirements

- macOS
- Python 3.12+ (or compatible Python 3)
- Android emulator with ADB access
- Tesseract OCR

## 1) Install dependencies

From `/Users/xinyuyuan/workspace/CodeX/WOAbot`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract
```

## 2) Install/verify ADB

If `adb` is missing:

```bash
brew install android-platform-tools
```

Verify device/emulator:

```bash
adb devices
```

You should see at least one `device`.

Quick screenshot sanity check:

```bash
adb exec-out screencap -p > /tmp/screen.png
ls -lh /tmp/screen.png
```

## 3) Configure bot

Main config:
- `/Users/xinyuyuan/workspace/CodeX/WOAbot/config.yaml`

Main sections:
- `templates`: template image list, threshold, tap action
- `startup_flow`: startup sequence
- `phase2`: runtime loop behavior

Important flags:
- `debug_logging`: verbose debug logs (`true`/`false`)
- `save_debug_screenshots`: save action screenshots (`true`/`false`)
- `phase2.test_mode`: tab/card loop test only (`true`) vs real actions (`false`)

### Template naming for incorrect enabled filters

You can add templates like:
- `button_enabled_incorrect_1`
- `button_enabled_incorrect_2`
- `button_enabled_incorrect_3`

Bot will auto-detect names matching `button_enabled_incorrect_<N>` and click them off between category loops.

Optional explicit list:
- `phase2.incorrect_enabled_templates`
- `phase2.incorrect_enabled_max_passes`

## 4) Run

```bash
source .venv/bin/activate
python3 run_bot.py --config config.yaml
```

Stop with `Ctrl+C`.

## Current behavior summary

### Phase 1
- Click `start_button`
- Find and select airport (image/text flow configured in `startup_flow`)
- Click `play_button`

### Phase 2 setup
- Click grey mode
- Click filter button
- Click actionable filter

### Phase 2 categories

Processing:
- `processing_finish_handling_button` -> click
- `processing_claim_rewards_button` -> click
  - then checks popup buttons:
    - `processing_claim_rewards_popup_confirm_button`
    - `processing_claim_reward_popup_button`
- Assign-crew case:
  - click add first
  - if `processing_not_enough_crew_message` appears before toggle, skip this card
  - else toggle ramp agent, then click start handling

Landing:
- If stand selection state is detected:
  - click leftmost `landing_empty_stand_option`
  - click `landing_confirm_button`
- Otherwise click `landing_clear_to_land_button` (or yellow fallback region)

Depart:
- Click `depart_execute_button_template` if available
- Else click yellow button in `phase2.depart_yellow_button_region_pct`

## OCR notes

Bot supports OCR for:
- `plane_name_region_pct`
- `plane_model_region_pct`

Tune these rectangles in `config.yaml` if plane name/model extraction is unstable.

## Logs

- `[PLANE]`: per-plane action log
- `[PHASE2]`: category/action flow
- `[PHASE2-TEST]`: test mode flow
- `[WARN]`, `[ERROR]`, `[FAIL]`: warning/error/failure paths
- `[DEBUG]`: debug-only messages (`debug_logging: true`)

## Troubleshooting

- `Screenshot bytes are empty or not a valid PNG stream`:
  - verify `adb devices`
  - run `adb exec-out screencap -p > /tmp/screen.png`
- OCR not detecting expected text:
  - enable `debug_logging: true`
  - tune OCR region rectangles in `phase2`
- Wrong taps:
  - re-capture/update template PNGs
  - adjust template thresholds in `config.yaml`
