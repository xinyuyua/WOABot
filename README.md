# ADB Game Bot Skeleton (macOS + Android Emulator)

Two-phase automation structure:
- Phase 1: startup flow (`start -> airport -> play`)
- Phase 2: in-game action loop (`processing -> landing -> depart`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract
```

## Run

```bash
python run_bot.py --config config.yaml
```

## Config model

Main file: `/Users/xinyuyuan/workspace/CodeX/WOAbot/config.yaml`

Sections:
- `templates`: all matchable UI elements and thresholds
- `startup_flow`: launch sequence
- `phase2`: in-game loop behavior and template mappings

## Phase 2 behavior

`phase2.test_mode: true` behavior:
- Refresh actionable filter
- Cycle tabs only: `processing -> landing -> depart`
- Optional card loop mode: click card slots one-by-one in configured list region
- No execution actions

`phase2.test_mode: false` behavior:
1. Ensure grey mode is enabled.
2. Click filter button to open filter panel.
3. Ensure actionable filter is enabled.
4. Try `processing` category.
5. Try `landing` category.
6. Try `depart` category.

Timing:
- One-time delay after startup enters game: `phase2.post_start_delay_sec` (default `2.0`)
- Delay between setup/category clicks: `phase2.inter_click_delay_sec` (default `0.3`)
- If any action happened while sweeping all categories: sleep `phase2.action_cycle_delay_sec` (default `2.0`)
- If no category action happened: sleep `phase2.idle_cycle_delay_sec` (default `5.0`)

Logs:
- `[PHASE2]` for category/button actions
- `[PHASE2-TEST]` for test-mode category loop traces
- `[PLANE]` for plane-level action records
- `[TRACE]` and `[DEBUG]` for matching diagnostics

Landing flow (real mode):
- Case 1: `landing_clear_to_land_button` visible -> click it.
- Case 2: `landing_select_stand_disabled_button` / stand selection shown:
  click `landing_empty_stand_option`, then click `landing_confirm_button`.

Test mode card loop tuning:
- `phase2.card_key_icon_template`: optional template name for icon-driven card detection (for example yellow `!`, default `card_alert_icon`)
- `phase2.card_list_region_pct`: panel region for card list
- visible card count is auto-derived from start/step values
- `phase2.card_anchor_x_pct`: horizontal click anchor inside region
- `phase2.card_start_y_pct`: first slot vertical position (region-local)
- `phase2.card_step_y_pct`: vertical distance between slots

## Notes

- Plane memory is tracked in-process (`name -> last category/action/time`) for future expansion.
- Crew number logic uses optional OCR regions if configured under `phase2`.


Depart fallback:
- If template action is missing, detect any yellow button in `phase2.depart_yellow_button_region_pct` and click it.

Processing flow (real mode):
- Case 1: `processing_claim_rewards_button` visible -> click it.
- Case 2: `processing_assign_crew_disabled_button` visible ->
  click `processing_add_enabled_button` until disabled, click `processing_ramp_agent_toggle_button`, then click `processing_start_handling_button`.
