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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
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
