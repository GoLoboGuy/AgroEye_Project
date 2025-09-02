
# Backend/services/guard.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import os
from .utils import to_float, safe_conf, to_str

@dataclass
class GuardConfig:
    gate_min: float = 0.70
    delta_max: float = 0.45
    pick_override: float = 0.98

    @classmethod
    def from_env(cls) -> "GuardConfig":
        return cls(
            gate_min=to_float(os.getenv("LEAF_GATE_MIN"), 0.70),
            delta_max=to_float(os.getenv("LEAF_DELTA_MAX"), 0.45),
            pick_override=to_float(os.getenv("PICK_OVERRIDE"), 0.98),
        )

@dataclass
class GuardInput:
    mn_label: str
    mn_conf: float
    rn_label: str
    rn_conf: float
    ens_label: str
    ens_conf: float
    picked_model: str
    picked_label: str
    picked_conf: float
    entropy_mn: Optional[float] = None
    entropy_rn: Optional[float] = None
    entropy_ens: Optional[float] = None

@dataclass
class GuardDecision:
    is_unknown: bool
    reason: str
    info: Optional[Dict[str, Any]] = None  # ← 추가: 부가 정보(임계값/컨피던스 등)

    # 3-값 언패킹 지원: (is_unknown, reason, info)
    def __iter__(self):
        yield self.is_unknown
        yield self.reason
        yield self.info

    def __len__(self):
        return 3

    def __bool__(self):
        return self.is_unknown

    def as_tuple(self, n: int = 3):
        t = (self.is_unknown, self.reason, self.info)
        return t if n >= 3 else t[:2]

class ClassGuard:
    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg

    def evaluate(self, gi: GuardInput) -> GuardDecision:
        mn_conf = safe_conf(gi.mn_conf, 0.0)
        rn_conf = safe_conf(gi.rn_conf, 0.0)
        ens_conf = safe_conf(gi.ens_conf, 0.0)
        picked_conf = safe_conf(gi.picked_conf, 0.0)

        info = {
            "thresholds": {
                "gate_min": self.cfg.gate_min,
                "delta_max": self.cfg.delta_max,
                "pick_override": self.cfg.pick_override,
            },
            "conf": {
                "mn": mn_conf,
                "rn": rn_conf,
                "ens": ens_conf,
                "picked": picked_conf,
            },
            "labels": {
                "mn": gi.mn_label,
                "rn": gi.rn_label,
                "ens": gi.ens_label,
                "picked": gi.picked_label,
                "model": gi.picked_model,
            }
        }

        # override 우선
        if (mn_conf >= self.cfg.pick_override
                or rn_conf >= self.cfg.pick_override
                or picked_conf >= self.cfg.pick_override):
            return GuardDecision(False, f"override: >= {self.cfg.pick_override}", info)

        reasons = []
        # 두 분류기 모두 0.70 이하이거나, 차이가 0.45 이상 → Unknown
        if (mn_conf <= self.cfg.gate_min) and (rn_conf <= self.cfg.gate_min):
            reasons.append(f"both<=gate_min({self.cfg.gate_min}) mn={mn_conf:.3f}, rn={rn_conf:.3f}")
        delta = abs(mn_conf - rn_conf)
        if delta >= self.cfg.delta_max:
            reasons.append(f"delta>=delta_max({self.cfg.delta_max}) |mn-rn|={delta:.3f}")

        if reasons:
            return GuardDecision(True, "; ".join(reasons), info)
        return GuardDecision(False, "ok", info)

    def __call__(self, *args, **kwargs) -> GuardDecision:
        """
        지원 시그니처:
        - guard(GuardInput(...))
        - guard(mn_label, mn_conf, rn_label, rn_conf, ens_label, ens_conf, picked_model, picked_label, picked_conf, [entropy_mn, entropy_rn, entropy_ens])
        - guard(**kwargs) 또는 guard(dict)  # 키: mn_label, mn_conf, rn_label, rn_conf, ens_label, ens_conf, picked_model, picked_label, picked_conf, entropy_mn, entropy_rn, entropy_ens
        """
        # 1) GuardInput 1개
        if len(args) == 1 and isinstance(args[0], GuardInput):
            return self.evaluate(args[0])

        # 2) dict 1개
        if len(args) == 1 and isinstance(args[0], dict):
            d = args[0]
            gi = GuardInput(
                mn_label=to_str(d.get("mn_label"), ""),
                mn_conf=safe_conf(d.get("mn_conf"), 0.0),
                rn_label=to_str(d.get("rn_label"), ""),
                rn_conf=safe_conf(d.get("rn_conf"), 0.0),
                ens_label=to_str(d.get("ens_label"), ""),
                ens_conf=safe_conf(d.get("ens_conf"), 0.0),
                picked_model=to_str(d.get("picked_model"), ""),
                picked_label=to_str(d.get("picked_label"), ""),
                picked_conf=safe_conf(d.get("picked_conf"), 0.0),
                entropy_mn=d.get("entropy_mn"),
                entropy_rn=d.get("entropy_rn"),
                entropy_ens=d.get("entropy_ens"),
            )
            return self.evaluate(gi)

        # 3) kwargs
        if kwargs:
            gi = GuardInput(
                mn_label=to_str(kwargs.get("mn_label"), ""),
                mn_conf=safe_conf(kwargs.get("mn_conf"), 0.0),
                rn_label=to_str(kwargs.get("rn_label"), ""),
                rn_conf=safe_conf(kwargs.get("rn_conf"), 0.0),
                ens_label=to_str(kwargs.get("ens_label"), ""),
                ens_conf=safe_conf(kwargs.get("ens_conf"), 0.0),
                picked_model=to_str(kwargs.get("picked_model"), ""),
                picked_label=to_str(kwargs.get("picked_label"), ""),
                picked_conf=safe_conf(kwargs.get("picked_conf"), 0.0),
                entropy_mn=kwargs.get("entropy_mn"),
                entropy_rn=kwargs.get("entropy_rn"),
                entropy_ens=kwargs.get("entropy_ens"),
            )
            return self.evaluate(gi)

        # 4) 포지셔널 9~12개 (과거 스타일)
        if len(args) >= 9:
            mn_label, mn_conf, rn_label, rn_conf, ens_label, ens_conf, picked_model, picked_label, picked_conf = args[:9]
            entropy_mn = args[9] if len(args) > 9 else None
            entropy_rn = args[10] if len(args) > 10 else None
            entropy_ens = args[11] if len(args) > 11 else None
            gi = GuardInput(
                mn_label=to_str(mn_label, ""),
                mn_conf=safe_conf(mn_conf, 0.0),
                rn_label=to_str(rn_label, ""),
                rn_conf=safe_conf(rn_conf, 0.0),
                ens_label=to_str(ens_label, ""),
                ens_conf=safe_conf(ens_conf, 0.0),
                picked_model=to_str(picked_model, ""),
                picked_label=to_str(picked_label, ""),
                picked_conf=safe_conf(picked_conf, 0.0),
                entropy_mn=entropy_mn if entropy_mn is None else safe_conf(entropy_mn, 0.0),
                entropy_rn=entropy_rn if entropy_rn is None else safe_conf(entropy_rn, 0.0),
                entropy_ens=entropy_ens if entropy_ens is None else safe_conf(entropy_ens, 0.0),
            )
            return self.evaluate(gi)

        raise TypeError(
            f"ClassGuard.__call__ invalid signature: args={len(args)} kwargs={list(kwargs.keys())}"
        )