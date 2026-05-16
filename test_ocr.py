from rapidocr import RapidOCR
import cv2

# 1. 明确指向你刚刚转换成功的本地检测模型 ONNX 路径
det_model_path = 'models/Multilingual_PP-OCRv3_det_infer.onnx'

# 2. 初始化 RapidOCR 引擎
# 我们把本地的检测模型传给 'Det.model_path'
# 识别模型（Rec）如果不指定，RapidOCR 会自动下载其默认的官方轻量化模型，从而帮你完成全套“检测+识别”
engine = RapidOCR(params={
    'Det.model_path': det_model_path
})

# 3. 指定你的本地测试图片路径（比如你刚刚上传的那张猴子带水印的图）
# 请替换为你真实的本地图片路径
img_path = "./test_images/hussar monkey2.png" 

# 4. 运行推理，接收标准的 RapidOCROutput 对象
output = engine(img_path)

# 5. 从新版 Dataclass 对象中直接提取对齐的属性字段
boxes = output.boxes   # 形状为 (N, 4, 2) 的坐标数组
txts = output.txts     # 长度为 N 的文本元组
scores = output.scores # 长度为 N 的置信度元组
elapse = output.elapse # 推理耗时

print("--- test ocr result ---")
# 判断是否有检测到文字 (看 boxes 是否为空且不为 None)
if boxes is not None and len(boxes) > 0:
    # 巧妙利用 zip 将检测、识别、置信度打包，按行循环输出
    for idx, (box, text, score) in enumerate(zip(boxes, txts, scores)):
        print(f"text block id: {idx+1}:")
        print(f"  - coordinate position: {box.tolist()}") # 转成 list 方便查看
        print(f"  - text content: {text}")
        print(f"  - confidence score: {score:.4f}")
else:
    print("image has no text.")

# 打印推理耗时
if elapse:
    print(f"\nocr inference time: {elapse:.3f} seconds")

# 6. 可视化结果 (安全健壮的版本)
if boxes is not None and len(boxes) > 0:
    # 使用 os.path.basename 提取纯文件名，比如 "hussar monkey2.png"
    pure_img_name = os.path.basename(img_path)
    
    # 拼接成合理的文件名，如 "vis_hussar monkey2.png"
    save_path = f"vis_{pure_img_name}"
    
    output.vis(save_path)
    print(f"visualized detection result saved to: {save_path}")
