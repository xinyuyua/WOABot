# WOAbot Coding Agent Guidelines

This document contains rules, guidelines, and behavioral constraints for coding agents working on the WOAbot project.

## Architecture Guidelines

- **Category Tab Switching**: The category tabs (`processing`, `landing`, and `departing`) in the game support multi-select. To prevent multiple tabs from being active simultaneously, always use the `self._switch_category_tab()` helper in [bot.py](file:///Users/xinyuyuan/workspace/CodeX/WOAbot/src/gamebot/bot.py) when switching tabs. Do not use direct template clicking. Note: Since tab buttons change appearance/darken when selected, template matching will fail to find the active tab for deselection. `_switch_category_tab` therefore uses cached screen coordinates of the active tab (`self.current_category_tab_xy`) to perform deselect clicks directly.
- **Inactivity Timeout**: The bot tracks inactivity via `self.last_action_monotonic`. Any automated step that successfully performs a tap or swipe must return `True` so that the inactivity timer is reset in the main loop.
- **Run-Time Limits**: CLI parameter `--run-time` overrides `BotConfig.run_time_limit_sec`. Any modifications to the main execution loop in `GameBot.run()` must preserve the check for the run-time limit.

## Coding Style & Patterns

- **Tap Jittering**: Tab buttons and other stable elements may bypass tap jittering. However, in Phase 2, click locations on active cards and buttons should use `self._apply_tap_jitter()` to simulate human interactions and prevent detection.
- **Time delays**: Respect configured delays like `phase2.inter_click_delay_sec` and `phase2.action_cycle_delay_sec`. Do not introduce hardcoded sleep values without a strong reason.
