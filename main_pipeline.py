import os
import cv2
import json

# 🚀 引入三个完全独立的专家模块
from experts.expert_ocr import ImageTextAuditor
from experts.expert_detector import OpenVocabularyDetector
from experts.expert_classifier import FineGrainedClassifier

print("--> System Initializing, loading experts to memory...")
# 1. 批量初始化专家（一次载入，永不重复）
ocr_expert = ImageTextAuditor()
detector_expert = OpenVocabularyDetector()
classifier_expert = FineGrainedClassifier() # EVA-02 在这里被长驻载入
print("--> All experts loaded successfully, pipeline ready.\n" + "="*50)

def run_themis_pipeline(image_path, class_label=None, target_text=None, expected_count=1):
    print(f"\n[Processing] Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at {image_path}")
        return None

    # 从物理磁盘只读一次，生成内存矩阵
    img_bgr = cv2.imread(image_path)
    expert_responses = []

    # ==================== [专家 1: OCR 审计] ====================
    print("\n--> Calling expert OCR auditor [image_text_auditor]...")
    ocr_result = ocr_expert.audit(img_bgr, target_text=target_text)
    print(f"    [OCR Verdict]: {ocr_result['verdict']}")
    expert_responses.append(ocr_result)

    # ==================== [专家 2: DINO 检测] ====================
    print("\n--> Calling expert DINO detector [open_vocabulary_detector]...")
    detector_result = detector_expert.audit(img_bgr, query_text=class_label, expected_count=expected_count)
    print(f"    [DINO Verdict]: {detector_result['verdict']}")
    expert_responses.append(detector_result)

    # ==================== [专家 3: EVA-02 细粒度分类] ====================
    # 完全解耦运行，不依赖前两个专家的输出，直接做独立的身份审计
    print("\n--> Calling expert EVA-02 classifier [fine_grained_classifier]...")
    classifier_result = classifier_expert.audit(img_bgr, target_class=class_label)
    print(f"    [Classifier Verdict]: {classifier_result['verdict']}")
    print(f"    [Top-1 Prediction]: {classifier_result['metrics']['top1_prediction']} (Logit: {classifier_result['metrics']['top1_logit']})")
    expert_responses.append(classifier_result)


    # ==================== [综合汇总结果阶段] ====================
    gathered_evidences = {
        "image_metadata": {
            "image_path": image_path,
            "class_label": class_label
        },
        "expert_responses": expert_responses  # OCR + DINO + EVA-02 的数据大团圆
    }
    
    return gathered_evidences

if __name__ == "__main__":
    # 拿你 test_images 目录里的某张真实图片试刀
    test_image = "./test_images/hussar monkey2.png"
    
    results = run_themis_pipeline(
        test_image, 
        class_label="hussar monkey", 
        target_text=None, 
        expected_count=1
    )
    
    print("\n" + "="*20 + " Final Gathered Evidences (3 Experts) " + "="*20)
    print(json.dumps(results, indent=4, ensure_ascii=False))