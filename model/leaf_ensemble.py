# backend/leaf_ensemble.py
from pathlib import Path
import json
import numpy as np
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

WEIGHTS_DIR = Path(__file__).parent / "weights"
MN_PATH = WEIGHTS_DIR / "mobilenet_v2" / "best.pth"
RN_PATH = WEIGHTS_DIR / "resnet50" / "best.pth"
CLS_JSON = WEIGHTS_DIR / "class_to_idx.json"

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet 정규화(코랩과 동일)
to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def _load_class_names(json_path: Path):
    with open(json_path) as f:
        class_to_idx = json.load(f)
    idx2cls = [None] * len(class_to_idx)
    for c, i in class_to_idx.items():
        idx2cls[i] = c
    return idx2cls

def _safe_load_state_dict(model: nn.Module, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # last_checkpoint 형태(학습시): {"model_state": state_dict, "epoch":..., "best_acc":...}
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        sd = ckpt["model_state"]
    else:
        sd = ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] partial load: missing={len(missing)}, unexpected={len(unexpected)}")
    return model

def _build_mobilenet(num_classes: int):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m

def _build_resnet50(num_classes: int):
    r = models.resnet50(weights=None)
    r.fc = nn.Linear(r.fc.in_features, num_classes)
    return r

class LeafEnsemble:
    def __init__(self):
        self.idx2cls = _load_class_names(CLS_JSON)
        self.num_classes = len(self.idx2cls)

        mn = _build_mobilenet(self.num_classes)
        rn = _build_resnet50(self.num_classes)

        _safe_load_state_dict(mn, MN_PATH)
        _safe_load_state_dict(rn, RN_PATH)

        self.mn = mn.to(device).eval()
        self.rn = rn.to(device).eval()

    @torch.no_grad()
    def _predict_probs(self, model: nn.Module, pil: Image.Image):
        x = to_tensor(pil).unsqueeze(0).to(device, non_blocking=True)
        prob = torch.softmax(model(x), dim=1)[0].float().cpu().numpy()
        return prob

    @staticmethod
    def _top1_with_margin(prob: np.ndarray) -> Tuple[int, float, float]:
        top1_idx = int(prob.argmax())
        top1 = float(prob[top1_idx])
        if len(prob) >= 2:
            # top-2
            top2 = float(np.partition(prob, -2)[-2])
        else:
            top2 = 0.0
        margin = top1 - top2
        return top1_idx, top1, margin

    @torch.no_grad()
    def predict_one(self, pil: Image.Image) -> Dict:
        mn_prob = self._predict_probs(self.mn, pil)
        rn_prob = self._predict_probs(self.rn, pil)

        mn_i, mn_conf, mn_margin = self._top1_with_margin(mn_prob)
        rn_i, rn_conf, rn_margin = self._top1_with_margin(rn_prob)

        mn_label = self.idx2cls[mn_i]
        rn_label = self.idx2cls[rn_i]

        # 타이브레이커: conf 동률(사실상) → margin 큰 쪽
        EPS = 1e-6
        if abs(mn_conf - rn_conf) <= EPS:
            if mn_margin >= rn_margin:
                pick = ("MobileNetV2", mn_label, mn_conf)
            else:
                pick = ("ResNet50", rn_label, rn_conf)
        else:
            pick = ("ResNet50", rn_label, rn_conf) if rn_conf > mn_conf else ("MobileNetV2", mn_label, mn_conf)

        return {
            "mobilenet": {"label": mn_label, "confidence": mn_conf},
            "resnet50": {"label": rn_label, "confidence": rn_conf},
            "picked": {"model": pick[0], "label": pick[1], "confidence": pick[2]},
        }

# 싱글톤
_model_singleton: LeafEnsemble | None = None
def get_model() -> LeafEnsemble:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = LeafEnsemble()
    return _model_singleton
