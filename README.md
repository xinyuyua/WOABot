# WOABot (macOS + Android Emulator via ADB)

Automation bot for World of Airports running on an Android emulator.

Note:
- Tested using BlueStacks Air with a resolution of `1600 x 900`; different resolutions may result in incorrect behavior and need tuning.

High-level flow:
- Phase 1: enter game (`start -> select airport -> play`)
- Phase 2: loop categories (`processing -> landing -> depart`)

## Quick Demo

![WOABot demo](docs/demo.gif)

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
- `test_mode`: defaults to `true` (test mode). Set to `false` for production/real actions.
- `phase2.parse_plane_info`: toggle OCR for plane ID/model (`true`/`false`).
- `phase2.stop_on_unhandled_processing_state`: if `true`, stop bot on unknown processing state; if `false`, warn and continue.

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

### Production mode

Edit `/Users/xinyuyuan/workspace/CodeX/WOAbot/config.yaml`:

```yaml
test_mode: false
```

Then run the same command:

```bash
python3 run_bot.py --config config.yaml
```

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
    - `processing_claim_rewards_and_upgrade_popup_button`
    - `processing_claim_rewards_and_extend_popup_button`
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

Loop order and delay:
- Always try in order: `processing -> landing -> depart`
- If any category performs an action, restart from `processing`
- Delay after action cycle: `phase2.action_cycle_delay_sec` (default `0.5`)
- If no action across all 3 categories, wait `phase2.idle_cycle_delay_sec` (default `2.0`)
- Every sleep includes a random jitter offset in `[-0.1, +0.1]` seconds

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

- `adb devices` shows no emulator/device:
  - restart adb server:
    ```
    adb kill-server
    adb start-server
    adb devices
    ```
  - if still missing, restart emulator and run `adb devices` again
  - if multiple devices are connected, set `serial` in `config.yaml`
- `Screenshot bytes are empty or not a valid PNG stream`:
  - verify `adb devices`
  - run `adb exec-out screencap -p > /tmp/screen.png`
- OCR not detecting expected text:
  - enable `debug_logging: true`
  - tune OCR region rectangles in `phase2`
- Wrong taps:
  - re-capture/update template PNGs
  - adjust template thresholds in `config.yaml`
