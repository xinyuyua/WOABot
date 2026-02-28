#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess

from src.gamebot.bot import GameBot
from src.gamebot.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal ADB game automation bot")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    restart_adb_server(cfg.adb_path)
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
