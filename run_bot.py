#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess

from src.gamebot.bot import GameBot
from src.gamebot.config import BotConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal ADB game automation bot")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--save-debug-screenshots",
        action="store_true",
        help="Enable all debug/warning/failure screenshot generation for this run.",
    )
    parser.add_argument(
        "--airport-image",
        default="",
        help="Override airport image filename for pick_airport_image step(s), e.g. airport_inn.png",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--no-take-off-mode",
        action="store_true",
        help="Skip depart category and only run processing + landing.",
    )
    mode_group.add_argument(
        "--take-off-mode",
        action="store_true",
        help="Force normal mode including depart category.",
    )
    mode_group.add_argument(
        "--take-off-at-last-mode",
        dest="take_off_at_last_mode",
        action="store_true",
        help=(
            "Run processing+landing first, then after idle threshold switch to "
            "depart-only phase for configured duration."
        ),
    )
    return parser.parse_args()


def restart_adb_server(adb_path: str) -> None:
    try:
        devices = subprocess.run(
            [adb_path, "devices"], check=False, capture_output=True, text=True
        )
        if devices.returncode == 0:
            lines = [line.strip() for line in devices.stdout.splitlines() if line.strip()]
            device_rows = [line for line in lines[1:] if "\t" in line]
            if device_rows:
                print("[INFO] ADB device detected. Skipping adb server restart.")
                return
    except FileNotFoundError:
        print(f"[WARN] adb not found at '{adb_path}'. Skipping adb restart.")
        return
    except Exception as exc:
        print(f"[WARN] Failed to run 'adb devices': {exc}")

    print("[INFO] Restarting ADB server...")
    try:
        subprocess.run([adb_path, "kill-server"], check=False, capture_output=True, text=True)
    except Exception as exc:
        print(f"[WARN] Failed to run 'adb kill-server': {exc}")

    try:
        start = subprocess.run(
            [adb_path, "start-server"], check=False, capture_output=True, text=True
        )
        if start.returncode == 0:
            print("[INFO] ADB server started.")
        else:
            err = (start.stderr or start.stdout or "").strip()
            print(
                f"[WARN] 'adb start-server' returned non-zero exit code ({start.returncode}). {err}"
            )
    except Exception as exc:
        print(f"[WARN] Failed to run 'adb start-server': {exc}")


def apply_runtime_overrides(cfg: BotConfig, args: argparse.Namespace) -> None:
    if args.save_debug_screenshots:
        cfg.save_debug_screenshots = True
        print("[INFO] Runtime override applied: save_debug_screenshots=true")

    if args.airport_image:
        patched = False
        for step in cfg.startup_flow:
            if step.type == "pick_airport_image":
                step.image = args.airport_image
                patched = True
        if patched:
            print(f"[INFO] Runtime override applied: pick_airport_image='{args.airport_image}'")
        else:
            print("[WARN] --airport-image provided but no pick_airport_image step found in startup_flow.")

    if args.no_take_off_mode:
        cfg.phase2.no_take_off_mode = True
        cfg.take_off_at_last_mode = False
        print("[INFO] Runtime override applied: phase2.no_take_off_mode=true")
    elif args.take_off_mode:
        cfg.phase2.no_take_off_mode = False
        cfg.take_off_at_last_mode = False
        print("[INFO] Runtime override applied: phase2.no_take_off_mode=false")
    elif args.take_off_at_last_mode:
        cfg.take_off_at_last_mode = True
        cfg.phase2.no_take_off_mode = False
        print("[INFO] Runtime override applied: take_off_at_last_mode=true")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_runtime_overrides(cfg, args)
    restart_adb_server(cfg.adb_path)
    bot = GameBot(cfg)
    caffeinate_proc: subprocess.Popen[bytes] | None = None
    try:
        if platform.system() == "Darwin" and shutil.which("caffeinate"):
            # Keep macOS awake while this process is alive.
            caffeinate_proc = subprocess.Popen(
                ["caffeinate", "-dimsu", "-w", str(os.getpid())]
            )
            print("[INFO] macOS sleep prevention enabled via caffeinate.")
        bot.run()
    finally:
        if caffeinate_proc is not None and caffeinate_proc.poll() is None:
            caffeinate_proc.terminate()


if __name__ == "__main__":
    main()
