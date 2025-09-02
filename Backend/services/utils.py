
# Backend/services/utils.py
from __future__ import annotations
import math
from typing import Any

def to_float(x: Any, default: float) -> float:
    """문자열/None/NaN이 섞여도 float로 안전 변환, 실패 시 default."""
    if x is None:
        return default
    try:
        v = float(str(x).strip())
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def to_str(x: Any, default: str = "") -> str:
    return default if x is None else str(x)

def safe_conf(x: Any, default: float = 0.0, lo: float = 0.0, hi: float = 1.0) -> float:
    """컨피던스/엔트로피 등 실측 수치 정규화 (+범위 클램핑)."""
    v = to_float(x, default)
    if v < lo: v = lo
    if v > hi: v = hi
    return v

def is_blank(x: Any) -> bool:
    return x is None or (isinstance(x, str) and x.strip() == "")
