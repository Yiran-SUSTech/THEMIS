import os
import cv2
import json
import numpy as np  # 🚀 引入 NumPy 用于拦截类型

# 引入三个只负责提取客观证据的原子化专家模块
from experts.expert_ocr import ImageTextAuditor
from experts.expert_detector import OpenVocabularyDetector
from experts.expert_classifier import FineGrainedClassifier

# 🚀 编写一个万能的 NumPy 类型安全转换器
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
    ocr_result = ocr_expert.audit(img_bgr)
    print(f"    └─ Success. Found {ocr_result['raw_metrics']['detected_text_blocks']} text blocks.")
    expert_responses.append(ocr_result)

    # ==================== [专家 2: 纯净 DINO 开放域目标定位] ====================
    print(f"\n--> [Expert 2] Detecting bounding boxes for query: '{class_label}'...")
    detector_result = detector_expert.audit(img_bgr, query_text=class_label, threshold=text_threshold)
    print(f"    └─ Success. Located {detector_result['raw_metrics']['detected_count']} matching objects.")
    expert_responses.append(detector_result)

    # ==================== [专家 3: 纯净 EVA-02 细粒度分类特征] ====================
    print("\n--> [Expert 3] Evaluating fine-grained image classification...")
    classifier_result = classifier_expert.audit(img_bgr)
    print(f"    └─ Success. Top-1 Feature Candidate: {classifier_result['evidence']['top3_candidates'][0]['label_name']}")
    expert_responses.append(classifier_result)

    # ==================== [客观证据大礼包组装阶段] ====================
    gathered_evidences = {
        "image_metadata": {
            "image_path": image_path,
            "class_label": class_label,
            "image_resolution": f"{img_bgr.shape[1]}x{img_bgr.shape[0]}"
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
    # 🚀 在 json.dumps 中指定 cls=ThemeEvidenceEncoder 彻底免疫 int64 报错
    print(json.dumps(raw_evidence_report, indent=4, ensure_ascii=False, cls=ThemeEvidenceEncoder))