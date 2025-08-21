# Backend/services/classifier.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

# model 폴더 경로를 Python 경로에 추가
current_file = Path(__file__)
project_root = current_file.parent.parent.parent  # Backend/services 폴더의 상위 디렉토리
model_path = project_root / "model"

if str(model_path) not in sys.path:
    sys.path.insert(0, str(model_path))
    sys.path.insert(0, str(project_root))

try:
    from leaf_ensemble import get_model, LeafEnsemble
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: leaf_ensemble 모델을 불러올 수 없습니다: {e}")
    MODEL_AVAILABLE = False

class Classifier:
    def __init__(self):
        self.loaded = False
        self.model: Optional[LeafEnsemble] = None
        self.model_available = MODEL_AVAILABLE

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
        except Exception as e:
            print(f"Error: 모델 로드 실패: {e}")
            self.model_available = False
            self.loaded = True  # 데모 모드로 실행

    def classify(self, image_path: str) -> Tuple[str, float]:
        """이미지 경로를 받아 (클래스명, 신뢰도) 반환"""
        if not self.loaded:
            self.load()
            
        if self.model_available and self.model:
            try:
                from PIL import Image
                # 이미지 로드 및 예측
                image = Image.open(image_path).convert("RGB")
                prediction = self.model.predict_one(image)
                
                # 신뢰도 검증 수행
                mobilenet_conf = prediction["mobilenet"]["confidence"]
                resnet50_conf = prediction["resnet50"]["confidence"]
                
                # 신뢰도 검증: 두 분류기의 결과가 모두 0.75 이하이거나 차이가 0.3 이상인 경우
                if (mobilenet_conf <= 0.75 and resnet50_conf <= 0.75) or abs(mobilenet_conf - resnet50_conf) >= 0.3:
                    # 예외: 한쪽의 결과가 0.98 이상이면 해당 결과 사용
                    if mobilenet_conf >= 0.98:
                        return prediction["mobilenet"]["label"], mobilenet_conf
                    elif resnet50_conf >= 0.98:
                        return prediction["resnet50"]["label"], resnet50_conf
                    else:
                        # Unknown으로 분류
                        return "Unknown", 0.0
                
                # 선택된 모델의 결과 반환
                picked = prediction["picked"]
                return picked["label"], picked["confidence"]
                
            except Exception as e:
                print(f"Error: 모델 예측 실패: {e}")
                # 실패 시 데모 모드로 폴백
                pass
        
        # 데모 모드: 파일명에 힌트가 있으면 해당 클래스로 간주
        name = Path(image_path).name.lower()
        if "scab" in name:
            return "Apple___Apple_scab", 0.92
        if "black" in name:
            return "Apple___Black_rot", 0.88
        if "rust" in name:
            return "Apple___Cedar_apple_rust", 0.90
        if "healthy" in name:
            return "Apple___healthy", 0.97
        # 기본값
        return "Apple___Apple_scab", 0.75

    def classify_with_details(self, image_path: str) -> Dict:
        """이미지 경로를 받아 상세한 분류 결과 반환"""
        if not self.loaded:
            self.load()
            
        if self.model_available and self.model:
            try:
                from PIL import Image
                # 이미지 로드 및 예측
                image = Image.open(image_path).convert("RGB")
                prediction = self.model.predict_one(image)
                
                # 신뢰도 검증 수행
                mobilenet_conf = prediction["mobilenet"]["confidence"]
                resnet50_conf = prediction["resnet50"]["confidence"]
                
                # 신뢰도 검증: 두 분류기의 결과가 모두 0.75 이하이거나 차이가 0.3 이상인 경우
                if (mobilenet_conf <= 0.75 and resnet50_conf <= 0.75) or abs(mobilenet_conf - resnet50_conf) >= 0.3:
                    # 예외: 한쪽의 결과가 0.98 이상이면 해당 결과 사용
                    if mobilenet_conf >= 0.98:
                        # MobileNet 결과 사용
                        prediction["picked"] = {
                            "model": "MobileNetV2",
                            "label": prediction["mobilenet"]["label"],
                            "confidence": mobilenet_conf
                        }
                    elif resnet50_conf >= 0.98:
                        # ResNet50 결과 사용
                        prediction["picked"] = {
                            "model": "ResNet50",
                            "label": prediction["resnet50"]["label"],
                            "confidence": resnet50_conf
                        }
                    else:
                        # Unknown으로 분류
                        prediction["picked"] = {
                            "model": "Unknown",
                            "label": "Unknown",
                            "confidence": 0.0
                        }
                
                # 결과에 이미지 경로 추가
                prediction["image_path"] = image_path
                return prediction
                
            except Exception as e:
                print(f"Error: 모델 예측 실패: {e}")
                # 실패 시 데모 모드로 폴백
                pass
        
        # 데모 모드
        name = Path(image_path).name.lower()
        if "scab" in name:
            return {
                "mobilenet": {"label": "Apple___Apple_scab", "confidence": 0.92},
                "resnet50": {"label": "Apple___Apple_scab", "confidence": 0.89},
                "picked": {"model": "MobileNetV2", "label": "Apple___Apple_scab", "confidence": 0.92},
                "image_path": image_path
            }
        if "black" in name:
            return {
                "mobilenet": {"label": "Apple___Black_rot", "confidence": 0.88},
                "resnet50": {"label": "Apple___Black_rot", "confidence": 0.91},
                "picked": {"model": "ResNet50", "label": "Apple___Black_rot", "confidence": 0.91},
                "image_path": image_path
            }
        if "rust" in name:
            return {
                "mobilenet": {"label": "Apple___Cedar_apple_rust", "confidence": 0.90},
                "resnet50": {"label": "Apple___Cedar_apple_rust", "confidence": 0.93},
                "picked": {"model": "ResNet50", "label": "Apple___Cedar_apple_rust", "confidence": 0.93},
                "image_path": image_path
            }
        if "healthy" in name:
            return {
                "mobilenet": {"label": "Apple___healthy", "confidence": 0.97},
                "resnet50": {"label": "Apple___healthy", "confidence": 0.98},
                "picked": {"model": "ResNet50", "label": "Apple___healthy", "confidence": 0.98},
                "image_path": image_path
            }
        # 기본값
        return {
            "mobilenet": {"label": "Apple___Apple_scab", "confidence": 0.75},
            "resnet50": {"label": "Apple___Apple_scab", "confidence": 0.78},
            "picked": {"model": "ResNet50", "label": "Apple___Apple_scab", "confidence": 0.78},
            "image_path": image_path
        }

classifier = Classifier()