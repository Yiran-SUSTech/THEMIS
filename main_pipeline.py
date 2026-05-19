import cv2
from expert_ocr import ImageTextAuditor

# 1. 在超级大循环外面初始化，仅加载一次 OCR 模型
ocr_expert = ImageTextAuditor()

# 2. 在循环内处理某张图
img = cv2.imread("./test_images/hussar monkey2.png")

# 假设这是一张 T2I 图片，用户 Prompt 里面没有文字要求，传 None
# 如果 Prompt 是 "A storefront with text 'COFFEE'"，则 target_text="COFFEE"
ocr_result = ocr_expert.audit(img, target_text=None)

# 3. 打印直接拿到的专家证词
print(ocr_result["verdict"])  # 比如输出: "Watermark_Pollution"
print(ocr_result["metrics"]["text_alignment_score"])  # 输出对齐分数