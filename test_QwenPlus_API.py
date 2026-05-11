import os
import base64
from openai import OpenAI

# 1. 图像转 Base64 的工具函数
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 2. 初始化客户端
client = OpenAI(
    api_key="sk-9165cc69015b4a12ab542fb5edc20612",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 3. 准备图片
image_path = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images/hussar monkey.png"
print(f"image_path: {image_path}")
base64_image = encode_image(image_path)
print(f"image is transformed to base64")

# 4. 直接粘贴你的长 Prompt (使用三引号包裹)
prompt_text = """You are evaluating a generated image for image-generation benchmarking.
Ground the judgment jointly in:
- the image itself
- class label: hussar monkey
- prompt: N/A (there is no original text prompt, only the class label)

Please evaluate two things:

1. alignment_score
- Score from 0 to 1
- 1 = the image strongly matches the class label/prompt
- 0 = severe mismatch

2. artifact_score
- Score from 0 to 1
- 1 = minimal visible artifacts, high perceptual quality
- 0 = severe visible artifacts, obvious generation failures

Important instructions:
- Use only visible evidence from the image.
- Distinguish broad category match from fine-grained class/species match.
- For artifact assessment, consider anatomy, boundaries, texture consistency, duplicated/melted parts, implausible structure, and other visible generation artifacts.
- Do not invent hidden metadata.
- Keep reasoning concise but evidence-based.

Return JSON only in this schema:

{
"alignment_reasoning": "string",
"artifact_reasoning": "string",
"alignment_score": 0.0,
"artifact_score": 0.0,
"hard_failure": true,
"confidence": 0.0,
"key_issues": ["string"]
}"""

# 5. 调用 API
try:
    completion = client.chat.completions.create(
        model="qwen3.6-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        response_format={"type": "json_object"}  # 强制让模型返回标准 JSON
    )
    
    # 6. 输出结果
    print(completion.choices[0].message.content)

except Exception as e:
    print(f"发生错误: {e}")