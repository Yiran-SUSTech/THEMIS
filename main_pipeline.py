import os
import cv2
import json
import time  # 🚀 引入时间模块
import numpy as np

# 引入三个只负责提取客观证据的原子化专家模块
from experts.expert_ocr import ImageTextAuditor
from experts.expert_detector import OpenVocabularyDetector
from experts.expert_classifier import FineGrainedClassifier

# 万能的 NumPy 类型安全转换器
class ThemeEvidenceEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(ThemeEvidenceEncoder, self).default(obj)

print("--> System Initializing, loading expert engines to memory...")
ocr_expert = ImageTextAuditor(det_model_path='models/Multilingual_PP-OCRv3_det_infer.onnx')
detector_expert = OpenVocabularyDetector()
classifier_expert = FineGrainedClassifier()
print("--> All expert engines loaded successfully, pipeline ready.\n" + "="*50)

def run_themis_pipeline(image_path, class_label, text_threshold=0.3):
    print(f"\n[Processing] Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at {image_path}")
        return None

    img_bgr = cv2.imread(image_path)
    expert_responses = []

    # ==================== [专家 1: 纯净 OCR 文本提取] ====================
    print("\n--> [Expert 1] Scanning for any visible texts...")
    t_start = time.time()  # ⏱️ 锚点 1 开始
    ocr_result = ocr_expert.audit(img_bgr)
    ocr_time = (time.time() - t_start) * 1000  # ⏱️ 锚点 1 结束 (转换为毫秒)
    print(f"    └─ Success. Found {ocr_result['raw_metrics']['detected_text_blocks']} text blocks. Cost: {ocr_time:.2f} ms")
    expert_responses.append(ocr_result)

    # ==================== [专家 2: 纯净 DINO 开放域目标定位] ====================
    print(f"\n--> [Expert 2] Detecting bounding boxes for query: '{class_label}'...")
    t_start = time.time()  # ⏱️ 锚点 2 开始
    detector_result = detector_expert.audit(img_bgr, query_text=class_label, threshold=text_threshold)
    detector_time = (time.time() - t_start) * 1000  # ⏱️ 锚点 2 结束 (转换为毫秒)
    print(f"    └─ Success. Located {detector_result['raw_metrics']['detected_count']} matching objects. Cost: {detector_time:.2f} ms")
    expert_responses.append(detector_result)

    # ==================== [专家 3: 纯净 EVA-02 细粒度分类特征] ====================
    print("\n--> [Expert 3] Evaluating fine-grained image classification...")
    t_start = time.time()  # ⏱️ 锚点 3 开始
    classifier_result = classifier_expert.audit(img_bgr)
    classifier_time = (time.time() - t_start) * 1000  # ⏱️ ⏱️ 锚点 3 结束 (转换为毫秒)
    print(f"    └─ Success. Top-1 Feature Candidate: {classifier_result['evidence']['top3_candidates'][0]['label_name']}. Cost: {classifier_time:.2f} ms")
    expert_responses.append(classifier_result)

    # ==================== [客观证据大礼包组装阶段] ====================
    # 顺便把总耗时和各个子耗时塞进元数据里，方便后续分析
    gathered_evidences = {
        "image_metadata": {
            "image_path": image_path,
            "class_label": class_label,
            "image_resolution": f"{img_bgr.shape[1]}x{img_bgr.shape[0]}",
            "performance_ms": {
                "ocr_time_ms": round(ocr_time, 2),
                "detector_time_ms": round(detector_time, 2),
                "classifier_time_ms": round(classifier_time, 2),
                "total_experts_time_ms": round(ocr_time + detector_time + classifier_time, 2)
            }
        },
        "expert_responses": expert_responses
    }
    
    return gathered_evidences

if __name__ == "__main__":
    test_image = "./test_images/hussar monkey2.png"
    
    raw_evidence_report = run_themis_pipeline(
        image_path=test_image, 
        class_label="hussar monkey"
    )
    
    print("\n" + "="*20 + " Final Gathered Raw Evidences " + "="*20)
    print(json.dumps(raw_evidence_report, indent=4, ensure_ascii=False, cls=ThemeEvidenceEncoder))