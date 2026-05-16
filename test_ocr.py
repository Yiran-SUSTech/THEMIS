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

# 4. 运行推理 (修改这行：不要用逗号拆包，直接接收整个对象)
output = engine(img_path)

# 5. 从 output 对象中提取 result 和 elapse 属性
result = output.result
elapse = output.elapse

# 5. 打印解析出来的结构化数据
print("--- test ocr result ---")
if result:
    for idx, line in enumerate(result):
        # line 的结构为: [ [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ], "识别文本", 识别置信度分数 ]
        box = line[0]
        text = line[1]
        score = line[2]
        print(f"text block id: {idx+1}:")
        print(f"  - coordinate position: {box}")
        print(f"  - text content: {text}")
        print(f"  - confidence score: {score:.4f}")
else:
    print("image has no text.")

print(f"\nocr inference time: {sum(elapse):.3f} seconds")

# 6. 可视化结果并保存到本地（会在当前目录下生成 vis_hussar_monkey2.jpg）
if result:
    from rapidocr.utils import VisOCR
    vis = VisOCR()
    box_list, text_list, score_list = zip(*result)
    img = cv2.imread(img_path)
    # 绘制检测框和文本
    res_img = vis(img, box_list, text_list, score_list)
    cv2.imwrite(f"vis_{img_path}", res_img)
    print(f"visualized detection result saved to: vis_{img_path}")
