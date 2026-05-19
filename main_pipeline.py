import os
import cv2
import json

# 🚀 核心修改：由于文件在 experts 文件夹下，使用相对路径包引入
from experts.expert_ocr import ImageTextAuditor

print("--> System Initializing, loading experts to memory...")
# 实例化 OCR 专家（模型仅在这里加载一次，后续循环 10,000 张图时不再重复加载）
ocr_expert = ImageTextAuditor(det_model_path='models/Multilingual_PP-OCRv3_det_infer.onnx')
print("--> Experts loaded successfully, pipeline ready.\n" + "="*50)

def run_themis_pipeline(image_path, class_label=None, target_text=None):
    """
    单张图片的中央流水线控制逻辑
    :param image_path: 图片本地路径
    :param class_label: 评测目标的类别标签（如 "bald eagle"）
    :param target_text: 可选，用户 Prompt 中期望渲染的文本（用于 T2I 文本对齐检查）
    """
    print(f"\n[Processing] Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at {image_path}")
        return None

    # 1. 读取图像矩阵（保存在内存中，直接传递给专家，拒绝多余的磁盘I/O）
    img_bgr = cv2.imread(image_path)
    
    # 2. 模拟 Router 决议（此处我们手动展示如何调用已经就绪的 OCR 专家）
    # 实际运行时，这里会去解析 Router 传过来的 Plan.json 决定调谁
    print("--> Calling expert OCR auditor [image_text_auditor] for text detection and copyright screening...")
    ocr_result = ocr_expert.audit(img_bgr, target_text=target_text)
    
    # 3. 收集专家证词
    gathered_evidences = {
        "image_metadata": {
            "image_path": image_path,
            "class_label": class_label
        },
        "expert_responses": [
            ocr_result  # 未来其他 6 个专家跑完的结果也统统 append 到这个列表里
        ]
    }
    
    # 4. 打印查看专家输出的结果，格式严丝合缝对应你的注册表描述
    print(f"[Expert Verdict]: {ocr_result['verdict']}")
    print(f"[Detected Text Blocks Count]: {ocr_result['metrics']['detected_text_blocks']}")
    print(f"[Full Extracted Text]: {ocr_result['evidence']['full_extracted_text']}")
    
    return gathered_evidences

if __name__ == "__main__":
    # 测试一下单张图的调用
    test_image = "./test_images/hussar monkey2.png"
    
    # 假设这张图原本的 Prompt 并没有要求生成文本，我们把 target_text 设为 None
    # 系统会自动帮你排查有没有“水印污染”或“无意义的融化乱码文字”
    results = run_themis_pipeline(test_image, class_label="hussar monkey", target_text=None)