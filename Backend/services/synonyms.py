# Backend/services/synonyms.py
from __future__ import annotations
from typing import List, Dict

# 최소 예시 사전 — 필요시 확장
SYNONYMS: Dict[str, List[str]] = {
    "Apple___Apple_scab": ["Apple scab", "사과 검은무늬병", "Venturia inaequalis", "black spot of apple",],
    "Apple___Black_rot": ["Black rot", "사과 검은썩음병", "Botryosphaeria obtusa",],
    "Apple___Cedar_apple_rust": ["Cedar apple rust", "사과 붉은별무늬병", "Gymnosporangium juniperi-virginianae", "cedar rust",],
    "Apple___healthy": ["healthy apple leaf", "정상 사과 잎", "무병"],
    # 필요 클래스 추가
    "Blueberry___healthy": ["healthy blueberry leaf", "정상 블루베리 잎", "무병"],
    "Cherry___Powdery_mildew": ["Powdery mildew", "체리 흰가루병"],
    "Cherry___healthy": ["healthy cherry leaf", "정상 체리 잎", "무병"],
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": ["Cercospora leaf spot", "Gray leaf spot", "옥수수 갈색무늬병", "Cercospora zeae-maydis"],
    "Corn___Common_rust": ["Common rust", "옥수수 붉은녹병", "Puccinia sorghi"],
    "Corn___Northern_Leaf_Blight": ["Northern Leaf Blight", "옥수수 북부잎마름병", "Exserohilum turcicum"],
    "Corn___healthy": ["healthy corn leaf", "정상 옥수수 잎", "무병"],
    "Cucumber___Downy_mildew": ["Downy mildew", "오이 노균병"],
    "Cucumber___Healthy": ["healthy cucumber leaf", "정상 오이 잎", "무병"],
    "Cucumber___Powdery_mildew": ["Powdery mildew", "오이 흰가루병"],
    "Grape___Black_rot": ["Black rot", "포도 검은썩음병", "Guignardia bidwellii"],
    "Grape___Esca_(Black_Measles)": ["Esca", "Black Measles", "포도 에스카병"],
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ["Leaf blight", "Isariopsis Leaf Spot", "포도 잎마름병"],
    "Grape___healthy": ["healthy grape leaf", "정상 포도 잎", "무병"],
    "Orange___Haunglongbing_(Citrus_greening)": ["Citrus greening", "황룡병"],
    "Peach___Bacterial_spot": ["Bacterial spot", "복숭아 세균성 점무늬병"],
    "Peach___healthy": ["healthy peach leaf", "정상 복숭아 잎", "무병"],
    "Pepper,_bell___Bacterial_spot": ["Bacterial spot", "피망 세균성 점무늬병"],
    "Pepper,_bell___healthy": ["healthy bell pepper leaf", "정상 피망 잎", "무병"],
    "Potato___Early_blight": ["Early blight", "감자 줄기마름병"],
    "Potato___Late_blight": ["Late blight", "감자 역병"],
    "Potato___healthy": ["healthy potato leaf", "정상 감자 잎", "무병"],
    "Raspberry___healthy": ["healthy raspberry leaf", "정상 라즈베리 잎", "무병"],
    "Rice___Bacterial_Leaf_Blight": ["Bacterial Leaf Blight", "벼 세균성 잎마름병"],
    "Rice___Brown_Spot": ["Brown Spot", "벼 갈색무늬병"],
    "Rice___Healthy": ["healthy rice leaf", "정상 벼 잎", "무병"],
    "Rice___Leaf_Blast": ["Leaf Blast", "벼 도열병"],
    "Rice___Leaf_Scald": ["Leaf Scald", "벼 잎마름병"],
    "Rice___Sheath_Blight": ["Sheath Blight", "벼 도열병"],
    "Soybean___healthy": ["healthy soybean leaf", "정상 콩 잎", "무병"],
    "Squash___Powdery_mildew": ["Powdery mildew", "호박 흰가루병"],
    "Strawberry___Leaf_scorch": ["Leaf scorch", "딸기 잎마름병"],
    "Strawberry___healthy": ["healthy strawberry leaf", "정상 딸기 잎", "무병"],
    "Tomato___Bacterial_spot": ["Bacterial spot", "토마토 세균성 점무늬병"],
    "Tomato___Early_blight": ["Early blight", "토마토 잎마름병"],
    "Tomato___Late_blight": ["Late blight", "토마토 역병"],
    "Tomato___Leaf_Mold": ["Leaf Mold", "토마토 잎곰팡이병"],
    "Tomato___Septoria_leaf_spot": ["Septoria leaf spot", "토마토 세포리아 잎반점병"],
    "Tomato___Spider_mites Two-spotted_spider_mite": ["Spider mites", "Two-spotted spider mite", "점박이 응애"],
    "Tomato___Target_Spot": ["Target Spot", "토마토 타겟반점병"],
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ["Tomato Yellow Leaf Curl Virus", "토마토 황화잎말림바이러스"],
    "Tomato___Tomato_mosaic_virus": ["Tomato mosaic virus", "토마토 모자이크바이러스"],
    "Tomato___healthy": ["healthy tomato leaf", "정상 토마토 잎", "무병"],
}

# 라벨 정규화: Apple___Apple_scab → "Apple scab"

def normalize_label(label: str) -> str:
    parts = label.replace("__", "_").split("___")
    if len(parts) == 2:
        return parts[1].replace("_", " ")
    return label.replace("_", " ")


def class_to_query_terms(label: str) -> List[str]:
    base = normalize_label(label)
    cand = [base]
    cand.extend(SYNONYMS.get(label, []))
    # 중복 제거
    uniq = []
    seen = set()
    for t in cand:
        k = t.strip().lower()
        if k and k not in seen:
            uniq.append(t.strip())
            seen.add(k)
    return uniq


def as_boolean_query(terms: List[str]) -> str:
    if not terms:
        return ""
    # ("term1" OR "term2" OR ...)
    quoted = [f'"{t}"' for t in terms]
    return "(" + " OR ".join(quoted) + ")"