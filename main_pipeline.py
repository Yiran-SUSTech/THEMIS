import os
import cv2
import json

# 🚀 从 experts 文件夹引入两个完全独立的专家模块
from experts.expert_ocr import ImageTextAuditor
from experts.expert_detector import OpenVocabularyDetector

print("--> System Initializing, loading experts to memory...")
# 1. 初始化专家（模型仅在最外层循环外加载一次，长驻内存）
ocr_expert = ImageTextAuditor(det_model_path='models/Multilingual_PP-OCRv3_det_infer.onnx')
detector_expert = OpenVocabularyDetector()
print("--> Experts loaded successfully, pipeline ready.\n" + "="*50)

def run_themis_pipeline(image_path, class_label=None, target_text=None, expected_count=1):
    """
    单张图片的中央流水线控制逻辑
    :param image_path: 图片本地路径
    :param class_label: 评测目标的类别标签（如 "hussar monkey"）
    :param target_text: 可选，用户 Prompt 中期望渲染的文本
    :param expected_count: 预期目标数量
    """
    print(f"\n[Processing] Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at {image_path}")
        return None

    # 一次性读取图像矩阵到内存，直接供多方专家共同运算，拒绝多余的磁盘 I/O
    img_bgr = cv2.imread(image_path)
    
    # 用来存储本张图片实际运行的所有专家结果的容器
    expert_responses = []

    # ==================== [专家 1: OCR 审计独立调用] ====================
    # 实际项目中，可以通过 if "image_text_auditor" in router_plan_stages: 来控制
    print("\n--> Calling expert OCR auditor [image_text_auditor]...")
    ocr_result = ocr_expert.audit(img_bgr, target_text=target_text)
    
    # 打印查看 OCR 输出结果
    print(f"    [OCR Verdict]: {ocr_result['verdict']}")
    print(f"    [Detected Text Blocks]: {ocr_result['metrics']['detected_text_blocks']}")
    
    # 收集到总容器
    expert_responses.append(ocr_result)


    # ==================== [专家 2: DINO 目标定位独立调用] ====================
    # 保持原子化，不依赖 OCR 结果，直接用最原始的 img_bgr 运行
    print("\n--> Calling expert DINO detector [open_vocabulary_detector]...")
    detector_result = detector_expert.audit(
        img_bgr, 
        query_text=class_label, 
        expected_count=expected_count, 
        threshold=0.3
    )
    
    # 打印查看 DINO 输出结果
    print(f"    [DINO Verdict]: {detector_result['verdict']}")
    print(f"    [Detected Target Count]: {detector_result['metrics']['detected_count']}")
    
    # 收集到总容器
    expert_responses.append(detector_result)


    # ==================== [综合汇总结果阶段] ====================
    gathered_evidences = {
        "image_metadata": {
            "image_path": image_path,
            "class_label": class_label
        },
        "expert_responses": expert_responses  # 这里自动综合了 OCR 和 DINO 的全套结构化结果
    }
    
    return gathered_evidences

if __name__ == "__main__":
    test_image = "./test_images/hussar monkey2.png"
    
    # 运行流水线
    results = run_themis_pipeline(
        test_image, 
        class_label="hussar monkey", 
        target_text=None, 
        expected_count=1
    )
    
    # 打印查看最后综合打包的完整数据，格式完美对齐你的注册表，未来直接喂给 Reflector
    print("\n" + "="*20 + " Final Gathered Evidences " + "="*20)
    print(json.dumps(results, indent=4, ensure_ascii=False))