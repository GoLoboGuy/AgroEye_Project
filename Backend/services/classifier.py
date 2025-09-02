# Backend/services/classifier.py
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

# ✅ 상대 임포트로 패키지 안정화
from .guard import ClassGuard, GuardConfig

logger = logging.getLogger(__name__)

# model 폴더 경로를 Python 경로에 추가
current_file = Path(__file__)
project_root = current_file.parent.parent.parent  # Backend/services 상위
model_path = project_root / "model"
DEMO_MODE = bool(int(os.getenv("CLASSIFIER_DEMO_MODE", "0")))

if str(model_path) not in sys.path:
    sys.path.insert(0, str(model_path))
    sys.path.insert(0, str(project_root))

try:
    # model 폴더를 sys.path에 올렸으므로 leaf_ensemble가 일반 모듈처럼 import 가능
    try:
        from leaf_ensemble import get_model, LeafEnsemble  # 권장
    except ImportError:
        # 혹시 model이 패키지로 구성된 경우( __init__.py 존재 ) 대비
        from model.leaf_ensemble import get_model, LeafEnsemble
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: leaf_ensemble 모델을 불러올 수 없습니다: {e}")
    MODEL_AVAILABLE = False
    LeafEnsemble = object  # type: ignore


# 튜닝 상수 정의 추가  -----------------------------------------------------------------------------0902

GATE_MIN  = float(os.getenv("GATE_MIN",  "0.70"))  # 기본 0.75 → 0.70로 완화
DELTA_MAX = float(os.getenv("DELTA_MAX", "0.45"))  # 기본 0.30 → 0.45로 완화
AGREE_MIN = float(os.getenv("AGREE_MIN", "0.60"))  # 합의 예외 허들


class Classifier:
    def __init__(self):
        self.loaded: bool = False
        self.model: Optional[LeafEnsemble] = None
        self.model_available: bool = MODEL_AVAILABLE
        self._class_guard: Optional[ClassGuard] = None

    # ✅ 가드 주입(멱등). 항상 self.model에 붙임
    def _bind_guard_to_model(self) -> ClassGuard:
        guard = self._class_guard
        if guard is None:
            guard = ClassGuard(GuardConfig.from_env())
            self._class_guard = guard

        if self.model is not None and (
            not hasattr(self.model, "_class_guard")
            or getattr(self.model, "_class_guard", None) is None
        ):
            setattr(self.model, "_class_guard", guard)
            logger.debug("Injected _class_guard into LeafEnsemble")
        return guard

    def load(self):
        """실제 모델 로드 구현"""
        if not self.model_available:
            print("Warning: 모델을 사용할 수 없습니다. 데모 모드로 실행됩니다.")
            self.loaded = True
            return

        try:
            self.model = get_model()
            self.loaded = True
            print("LeafEnsemble 모델 로드 완료")
            # ✅ 로드 직후 가드 주입
            self._bind_guard_to_model()
        except Exception as e:
            print(f"Error: 모델 로드 실패: {e}")
            self.model_available = False
            self.loaded = True  # 데모 모드로 실행

    # ✅ 로드 보장 + 가드 재주입 보장
    def _ensure_loaded(self):
        if not self.loaded:
            self.load()
        else:
            if self.model is not None:
                self._bind_guard_to_model()

    def classify(self, image_path: str) -> Tuple[str, float]:
        """이미지 경로를 받아 (클래스명, 신뢰도) 반환"""
        self._ensure_loaded()

        if self.model_available and self.model:
            try:
                from PIL import Image
                image = Image.open(image_path).convert("RGB")

                # ✅ 추론 직전에도 한 번 더 보장(다중경로/리로드 대비)
                self._bind_guard_to_model()

                prediction = self.model.predict_one(image)

                # 컷오프 분기 수정 ------------------------------------------------------------------0902
                mobilenet_conf = prediction["mobilenet"]["confidence"]
                resnet50_conf  = prediction["resnet50"]["confidence"]
                mobilenet_lbl  = prediction["mobilenet"]["label"]
                resnet50_lbl   = prediction["resnet50"]["label"]
                agree_same_lbl = (mobilenet_lbl == resnet50_lbl)

                # 완화된 컷오프
                reject_by_gate  = (mobilenet_conf <= GATE_MIN and resnet50_conf <= GATE_MIN)
                reject_by_delta = (abs(mobilenet_conf - resnet50_conf) >= DELTA_MAX)

                if reject_by_gate or reject_by_delta:
                    # ✅ 합의 예외: 두 모델이 같은 클래스에 합의했고, 둘 중 하나가 AGREE_MIN 이상이면 채택
                    if agree_same_lbl and max(mobilenet_conf, resnet50_conf) >= AGREE_MIN:
                        agreed_label = mobilenet_lbl  # 두 라벨이 동일하므로 어느 쪽이든 동일
                        agreed_conf  = max(mobilenet_conf, resnet50_conf)
                        return agreed_label, agreed_conf

                    # 최상위 확신치 예외는 기존 로직 유지
                    if mobilenet_conf >= 0.98:
                        return prediction["mobilenet"]["label"], mobilenet_conf
                    elif resnet50_conf >= 0.98:
                        return prediction["resnet50"]["label"], resnet50_conf
                    else:
                        return "Unknown", 0.0

                # 컷오프에 안 걸리면 기존대로 ensemble picked 사용
                picked = prediction["picked"]
                return picked["label"], picked["confidence"]

            except Exception as e:
                print(f"Error: 모델 예측 실패: {e}")
                # 실패 시 데모 모드로 폴백

        # ------- 데모 모드 -------
        if DEMO_MODE:
            name = Path(image_path).name.lower()
            if "scab" in name:
                return "Apple___Apple_scab", 0.92
            if "black" in name:
                return "Apple___Black_rot", 0.88
            if "rust" in name:
                return "Apple___Cedar_apple_rust", 0.90
            if "healthy" in name:
                return "Apple___healthy", 0.97
            return "Apple___Apple_scab", 0.75
        return "Unknown", 0.0

    def classify_with_details(self, image_path: str) -> Dict:
        """이미지 경로를 받아 상세한 분류 결과 반환"""
        self._ensure_loaded()

        if self.model_available and self.model:
            try:
                from PIL import Image
                image = Image.open(image_path).convert("RGB")

                # ✅ 추론 직전에도 한 번 더 보장
                self._bind_guard_to_model()
#----------0902
                prediction = self.model.predict_one(image)

                mobilenet_conf = prediction["mobilenet"]["confidence"]
                resnet50_conf  = prediction["resnet50"]["confidence"]
                mobilenet_lbl  = prediction["mobilenet"]["label"]
                resnet50_lbl   = prediction["resnet50"]["label"]
                agree_same_lbl = (mobilenet_lbl == resnet50_lbl)

                reject_by_gate  = (mobilenet_conf <= GATE_MIN and resnet50_conf <= GATE_MIN)   # ✅ 환경변수화
                reject_by_delta = (abs(mobilenet_conf - resnet50_conf) >= DELTA_MAX)           # ✅ 환경변수화

                if reject_by_gate or reject_by_delta:
                    # ✅ 합의 예외: 두 모델이 같은 클래스로 합의 & 허들 통과 시 Unknown으로 덮어쓰지 않음
                    if agree_same_lbl and max(mobilenet_conf, resnet50_conf) >= AGREE_MIN:
                        prediction["picked"] = {
                            "model": "AgreeOverride",
                            "label": mobilenet_lbl,  # 두 모델 라벨 동일
                            "confidence": max(mobilenet_conf, resnet50_conf),
                            "reason": "AgreeOverride"
                        }
                        # (선택) meta.reason 갱신하여 디버깅 용이
                        meta = prediction.setdefault("meta", {})
                        meta["reason"] = f"AgreeOverride gate_min={GATE_MIN} delta_max={DELTA_MAX}"
                    else:
                        # 기존 Top1 override 유지
                        if mobilenet_conf >= 0.98:
                            prediction["picked"] = {
                                "model": "MobileNetV2",
                                "label": prediction["mobilenet"]["label"],
                                "confidence": mobilenet_conf,
                                "reason": "Top1Override"
                            }
                        elif resnet50_conf >= 0.98:
                            prediction["picked"] = {
                                "model": "ResNet50",
                                "label": prediction["resnet50"]["label"],
                                "confidence": resnet50_conf,
                                "reason": "Top1Override"
                            }
                        else:
                            prediction["picked"] = {
                                "model": "Unknown",
                                "label": "Unknown",
                                "confidence": 0.0,
                                "reason": f"Gate(delta>{DELTA_MAX} or both<{GATE_MIN})"
                            }

                prediction["image_path"] = image_path
                return prediction

            except Exception as e:
                print(f"Error: 모델 예측 실패: {e}")
                # 실패 시 데모 모드로 폴백

        # ------- 데모 모드 -------
        name = Path(image_path).name.lower()
        if "scab" in name:
            return {
                "mobilenet": {"label": "Apple___Apple_scab", "confidence": 0.92},
                "resnet50": {"label": "Apple___Apple_scab", "confidence": 0.89},
                "ensemble": {"label": "Apple___Apple_scab", "confidence": 0.91, "weights": {"mn": 0.25, "rn": 0.75}},
                "picked": {"model": "MobileNetV2", "label": "Apple___Apple_scab", "confidence": 0.92},
                "image_path": image_path,
                "meta": {"entropy": {"mobilenet": 0.0, "resnet50": 0.0, "ensemble": 0.0}, "inference_ms": 0.0}
            }
        if "black" in name:
            return {
                "mobilenet": {"label": "Apple___Black_rot", "confidence": 0.88},
                "resnet50": {"label": "Apple___Black_rot", "confidence": 0.91},
                "ensemble": {"label": "Apple___Black_rot", "confidence": 0.90, "weights": {"mn": 0.25, "rn": 0.75}},
                "picked": {"model": "ResNet50", "label": "Apple___Black_rot", "confidence": 0.91},
                "image_path": image_path,
                "meta": {"entropy": {"mobilenet": 0.0, "resnet50": 0.0, "ensemble": 0.0}, "inference_ms": 0.0}
            }
        if "rust" in name:
            return {
                "mobilenet": {"label": "Apple___Cedar_apple_rust", "confidence": 0.90},
                "resnet50": {"label": "Apple___Cedar_apple_rust", "confidence": 0.93},
                "ensemble": {"label": "Apple___Cedar_apple_rust", "confidence": 0.92, "weights": {"mn": 0.25, "rn": 0.75}},
                "picked": {"model": "ResNet50", "label": "Apple___Cedar_apple_rust", "confidence": 0.93},
                "image_path": image_path,
                "meta": {"entropy": {"mobilenet": 0.0, "resnet50": 0.0, "ensemble": 0.0}, "inference_ms": 0.0}
            }
        if "healthy" in name:
            return {
                "mobilenet": {"label": "Apple___healthy", "confidence": 0.97},
                "resnet50": {"label": "Apple___healthy", "confidence": 0.98},
                "ensemble": {"label": "Apple___healthy", "confidence": 0.98, "weights": {"mn": 0.25, "rn": 0.75}},
                "picked": {"model": "ResNet50", "label": "Apple___healthy", "confidence": 0.98},
                "image_path": image_path,
                "meta": {"entropy": {"mobilenet": 0.0, "resnet50": 0.0, "ensemble": 0.0}, "inference_ms": 0.0}
            }
        return {
            "mobilenet": {"label": "Apple___Apple_scab", "confidence": 0.75},
            "resnet50": {"label": "Apple___Apple_scab", "confidence": 0.78},
            "ensemble": {"label": "Apple___Apple_scab", "confidence": 0.77, "weights": {"mn": 0.25, "rn": 0.75}},
            "picked": {"model": "ResNet50", "label": "Apple___Apple_scab", "confidence": 0.78},
            "image_path": image_path,
            "meta": {"entropy": {"mobilenet": 0.0, "resnet50": 0.0, "ensemble": 0.0}, "inference_ms": 0.0}
        }


# 싱글톤 인스턴스
classifier = Classifier()
