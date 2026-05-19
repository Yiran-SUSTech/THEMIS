import os
import cv2
import json
import time
import numpy as np

# 🚀 引入全套 7 个原子化专家模块
from experts.expert_ocr import ImageTextAuditor
from experts.expert_detector import OpenVocabularyDetector
from experts.expert_classifier import FineGrainedClassifier
from experts.expert_pose import AnimalPoseEstimator
from experts.expert_depth import MonocularDepthEstimator
from experts.expert_sam import SegmentAnythingExpert
from experts.expert_qinsight import QInsightDistortionAnalyzer  # 新引入的 VLM 专家

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
# 批量初始化长驻内存（全套 7 路引擎齐聚）
ocr_expert = ImageTextAuditor(det_model_path='models/Multilingual_PP-OCRv3_det_infer.onnx')
detector_expert = OpenVocabularyDetector()
classifier_expert = FineGrainedClassifier()
pose_expert = AnimalPoseEstimator()
depth_expert = MonocularDepthEstimator()
sam_expert = SegmentAnythingExpert()
qinsight_expert = QInsightDistortionAnalyzer()  # 🚀 7 号专家就位
print("--> All expert engines loaded successfully, pipeline ready.\n" + "="*50)

def run_themis_pipeline(image_path, class_label, text_threshold=0.3):
    print(f"\n[Processing] Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at {image_path}")
        return None

    # 从磁盘读取一次生成内存矩阵，供给需要处理本地矩阵的专家
    img_bgr = cv2.imread(image_path)
    expert_responses = []

    # ==================== [专家 1: OCR 文本] ====================
    print("\n--> [Expert 1] Scanning for any visible texts...")
    t_start = time.time()
    ocr_result = ocr_expert.audit(img_bgr)
    ocr_time = (time.time() - t_start) * 1000
    print(f"    └─ Success. Found {ocr_result['raw_metrics']['detected_text_blocks']} text blocks. Cost: {ocr_time:.2f} ms")
    expert_responses.append(ocr_result)

    # ==================== [专家 2: DINO 检测] ====================
    print(f"\n--> [Expert 2] Detecting bounding boxes for query: '{class_label}'...")
    t_start = time.time()
    detector_result = detector_expert.audit(img_bgr, query_text=class_label, threshold=text_threshold)
    detector_time = (time.time() - t_start) * 1000
    print(f"    └─ Success. Located {detector_result['raw_metrics']['detected_count']} matching objects. Cost: {detector_time:.2f} ms")
    expert_responses.append(detector_result)

    # 提取第一个边界框用于下层联动
    dino_objects = detector_result["evidence"]["detected_objects"]
    first_box = dino_objects[0]["bounding_box"] if len(dino_objects) > 0 else None

    # ==================== [专家 3: EVA-02 分类] ====================
    print("\n--> [Expert 3] Evaluating fine-grained image classification...")
    t_start = time.time()
    classifier_result = classifier_expert.audit(img_bgr)
    classifier_time = (time.time() - t_start) * 1000
    print(f"    └─ Success. Top-1 Feature Candidate: {classifier_result['evidence']['top3_candidates'][0]['label_name']}. Cost: {classifier_time:.2f} ms")
    expert_responses.append(classifier_result)

    # ==================== [专家 4: rtmlib 姿态关键点] ====================
    print("\n--> [Expert 4] Extruding subject keypoints and poses...")
    t_start = time.time()
    pose_result = pose_expert.audit(img_bgr)
    pose_time = (time.time() - t_start) * 1000
    print(f"    └─ Success. Tracked {pose_result['raw_metrics']['detected_instances_count']} skeletal instances. Cost: {pose_time:.2f} ms")
    expert_responses.append(pose_result)

    # ==================== [专家 5: Depth Anything 单目深度] ====================
    print("\n--> [Expert 5] Computing monocular depth maps...")
    t_start = time.time()
    depth_result = depth_expert.audit(img_bgr, original_image_path=image_path)
    depth_time = (time.time() - t_start) * 1000
    print(f"    └─ Success. Depth map generated and saved. Cost: {depth_time:.2f} ms")
    expert_responses.append(depth_result)

    # ==================== [专家 6: SAM 像素级分割] ====================
    print("\n--> [Expert 6] Executing interactive pixel segmentation...")
    t_start = time.time()
    sam_result = sam_expert.audit(img_bgr, original_image_path=image_path, hint_box=first_box)
    sam_time = (time.time() - t_start) * 1000
    print(f"    └─ Success. Mask saved to disk. Cost: {sam_time:.2f} ms")
    expert_responses.append(sam_result)

    # ==================== [专家 7: Q-Insight 图像失真评估] ====================
    print("\n--> [Expert 7] Parsing image degradation with CoT trajectory...")
    t_start = time.time()
    # 🚀 VLM 专家直接传入本地图片路径
    qinsight_result = qinsight_expert.audit(image_path=image_path)
    qinsight_time = (time.time() - t_start) * 1000
    print(f"    └─ Success. Distortion profile: {qinsight_result['evidence']['distortion_class']} ({qinsight_result['evidence']['severity_level']}). Cost: {qinsight_time:.2f} ms")
    expert_responses.append(qinsight_result)


    # ==================== [客观证据大礼包组装] ====================
    gathered_evidences = {
        "image_metadata": {
            "image_path": image_path,
            "class_label": class_label,
            "image_resolution": f"{img_bgr.shape[1]}x{img_bgr.shape[0]}",
            "performance_ms": {
                "ocr_time_ms": round(ocr_time, 2),
                "detector_time_ms": round(detector_time, 2),
                "classifier_time_ms": round(classifier_time, 2),
                "pose_time_ms": round(pose_time, 2),
                "depth_time_ms": round(depth_time, 2),
                "sam_time_ms": round(sam_time, 2),
                "qinsight_time_ms": round(qinsight_time, 2),
                "total_experts_time_ms": round(ocr_time + detector_time + classifier_time + pose_time + depth_time + sam_time + qinsight_time, 2)
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