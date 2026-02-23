#!/usr/bin/env python3
from __future__ import annotations

import argparse

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
    bot.run()


if __name__ == "__main__":
    main()
