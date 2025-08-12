# backend/image_utils.py

import logging
import os
import uuid

import cv2
import numpy as np

from config import get_settings

# 설정 로드
settings = get_settings()

logger = logging.getLogger(__name__)

# 이미지 처리 설정
MAX_IMAGE_SIZE = settings.MAX_IMAGE_SIZE
JPEG_QUALITY = settings.JPEG_QUALITY
SUPPORTED_FORMATS = set(settings.SUPPORTED_FORMATS)

def validate_image_format(filename: str) -> bool:
    """이미지 파일 형식 검증"""
    if not filename:
        return False
    ext = os.path.splitext(filename.lower())[1]
    return ext in SUPPORTED_FORMATS

def resize_image_if_needed(image: np.ndarray, max_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    """이미지가 너무 클 경우 리사이즈"""
    height, width = image.shape[:2]
    
    if height <= max_size and width <= max_size:
        return image
    
    # 비율 유지하면서 리사이즈
    if height > width:
        new_height = max_size
        new_width = int(width * max_size / height)
    else:
        new_width = max_size
        new_height = int(height * max_size / width)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    logger.info(f"이미지 리사이즈: {width}x{height} -> {new_width}x{new_height}")
    return resized

def read_image(data: bytes) -> np.ndarray:
    """바이트 데이터에서 이미지 읽기 (최적화됨)"""
    try:
        # numpy로 직접 읽기
        arr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("이미지를 읽을 수 없습니다")
        
        # 이미지 크기 검증 및 리사이즈
        image = resize_image_if_needed(image)
        
        logger.info(f"이미지 로드 완료: {image.shape[1]}x{image.shape[0]}")
        return image
        
    except Exception as e:
        logger.error(f"이미지 읽기 실패: {str(e)}")
        raise

def encode_image(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """이미지를 바이트로 인코딩 (최적화됨)"""
    try:
        # OpenCV로 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode('.jpg', image, encode_param)
        
        if not success:
            raise ValueError("이미지 인코딩에 실패했습니다")
        
        return buffer.tobytes()
        
    except Exception as e:
        logger.error(f"이미지 인코딩 실패: {str(e)}")
        raise

def save_annotated_image(
    image: np.ndarray, 
    prefix: str = "images/",
    quality: int = JPEG_QUALITY
) -> str:
    """
    주석이 추가된 이미지를 파일로 저장하고 파일 경로를 반환합니다.
    
    Args:
        image: 저장할 이미지 (numpy array)
        prefix: 저장할 디렉토리 경로
        quality: JPEG 품질 (1-100)
    
    Returns:
        저장된 파일의 경로
    """
    try:
        # 디렉토리 생성
        os.makedirs(prefix, exist_ok=True)
        
        # UUID 기반 파일명 생성
        filename = f"{uuid.uuid4().hex}.jpg"
        path = os.path.join(prefix, filename)
        
        # 이미지 저장 (품질 최적화)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success = cv2.imwrite(path, image, encode_param)
        
        if not success:
            raise ValueError("이미지 저장에 실패했습니다")
        
        logger.info(f"이미지 저장 완료: {path}")
        return path
        
    except Exception as e:
        logger.error(f"이미지 저장 실패: {str(e)}")
        raise

def get_image_info(image: np.ndarray) -> dict:
    """이미지 정보 반환"""
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "size_bytes": image.nbytes
    }

def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR을 RGB로 변환 (OpenCV -> PIL 호환)"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def convert_to_bgr(image: np.ndarray) -> np.ndarray:
    """RGB를 BGR로 변환 (PIL -> OpenCV 호환)"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def optimize_image_for_display(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """디스플레이용 이미지 최적화"""
    height, width = image.shape[:2]
    
    if height <= max_size and width <= max_size:
        return image
    
    # 비율 유지하면서 리사이즈
    if height > width:
        new_height = max_size
        new_width = int(width * max_size / height)
    else:
        new_width = max_size
        new_height = int(height * max_size / width)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def cleanup_old_images(directory: str, max_age_days: int = None):
    """오래된 이미지 파일 정리"""
    if max_age_days is None:
        max_age_days = settings.IMAGE_CLEANUP_DAYS
        
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    os.remove(filepath)
                    logger.info(f"오래된 이미지 삭제: {filename}")
                    
    except Exception as e:
        logger.error(f"이미지 정리 중 오류: {str(e)}")
