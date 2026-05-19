import os
import cv2
import json

# 🚀 引入三个只负责提取客观证据的原子化专家模块
from experts.expert_ocr import ImageTextAuditor
from experts.expert_detector import OpenVocabularyDetector
from experts.expert_classifier import FineGrainedClassifier

print("--> System Initializing, loading expert engines to memory...")
# 1. 批量初始化专家（一次载入，在 10,000 张图的循环外长驻内存）
ocr_expert = ImageTextAuditor(det_model_path='models/Multilingual_PP-OCRv3_det_infer.onnx')
detector_expert = OpenVocabularyDetector()
classifier_expert = FineGrainedClassifier()
print("--> All expert engines loaded successfully, pipeline ready.\n" + "="*50)

def run_themis_pipeline(image_path, class_label, text_threshold=0.3):
    """
    单张图片的中央流水线控制逻辑（当前阶段：纯证据收集版）
    :param image_path: 图片本地路径
    :param class_label: 评测目标的类别标签（如 "hussar monkey"）
    :param text_threshold: DINO 开放域目标检测的置信度阈值
    """
    print(f"\n[Processing] Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at {image_path}")
        return None

    # 从物理磁盘只读取一次，生成内存矩阵，拒绝多余的磁盘重复 I/O
    img_bgr = cv2.imread(image_path)
    
    # 建立本张图片的专家证据收集舱
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
    # 这里不包含任何主观推导的 verdict 结论，全部是铁一样的物理数据
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
    # 拿你 test_images 目录里的赤猴图片作为单张测试源
    test_image = "./test_images/hussar monkey2.png"
    
    # 触发流水线
    raw_evidence_report = run_themis_pipeline(
        image_path=test_image, 
        class_label="hussar monkey"
    )
    
    # 打印查看最终输出的纯净客观报告
    print("\n" + "="*20 + " Final Gathered Raw Evidences " + "="*20)
    print(json.dumps(raw_evidence_report, indent=4, ensure_ascii=False))