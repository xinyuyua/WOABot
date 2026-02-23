from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
IEND_CHUNK = b"IEND\xaeB`\x82"


@dataclass
class AdbClient:
    adb_path: str = "adb"
    serial: str = ""

    def _base_cmd(self) -> list[str]:
        cmd = [self.adb_path]
        if self.serial:
            cmd.extend(["-s", self.serial])
        return cmd

    def run(self, *args: str, timeout: Optional[float] = 10) -> subprocess.CompletedProcess:
        cmd = self._base_cmd() + list(args)
        return subprocess.run(cmd, capture_output=True, check=True, timeout=timeout)

    def _extract_png(self, raw: bytes) -> bytes:
        # First try exact PNG stream in raw bytes.
        start = raw.find(PNG_MAGIC)
        if start != -1:
            end = raw.find(IEND_CHUNK, start)
            if end == -1:
                return raw[start:]
            return raw[start : end + len(IEND_CHUNK)]

        # Fallback for environments where adb output has CRLF translated globally.
        data = raw.replace(b"\r\n", b"\n")
        alt_magic = b"\x89PNG\n\x1a\n"
        start = data.find(alt_magic)
        if start == -1:
            return b""
        end = data.find(IEND_CHUNK, start)
        if end == -1:
            return data[start:]
        return data[start : end + len(IEND_CHUNK)]

    def screenshot_png_bytes(self) -> bytes:
        proc = self.run("exec-out", "screencap", "-p", timeout=15)
        return self._extract_png(proc.stdout)

    def tap(self, x: int, y: int) -> None:
        self.run("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        self.run(
            "shell",
            "input",
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            str(duration_ms),
        )
