from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class MatchResult:
    name: str
    confidence: float
    x: int
    y: int
    w: int
    h: int


@dataclass
class Template:
    name: str
    image: np.ndarray
    threshold: float


def load_template(path: str, name: str, threshold: float) -> Template:
    p = Path(path)
    image = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Template image not found or unreadable: {p}")
    return Template(name=name, image=image, threshold=threshold)


def match_template(frame: np.ndarray, template: Template) -> Optional[MatchResult]:
    result = cv2.matchTemplate(frame, template.image, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < template.threshold:
        return None
    h, w = template.image.shape[:2]
    x, y = max_loc
    return MatchResult(
        name=template.name,
        confidence=float(max_val),
        x=int(x),
        y=int(y),
        w=int(w),
        h=int(h),
    )
