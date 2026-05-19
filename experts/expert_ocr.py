import os
import cv2
import numpy as np
from rapidocr import RapidOCR

class ImageTextAuditor:
    def __init__(self, det_model_path='models/Multilingual_PP-OCRv3_det_infer.onnx'):
        """
        初始化 RapidOCR 引擎，模型仅加载一次，长驻内存
        """
        print(f"[Init] Loading RapidOCR with local det model: {det_model_path}")
        self.engine = RapidOCR(params={
            'Det.model_path': det_model_path
        })
        
        # 定义常见的版权/图库水印关键词，用于自动化筛查
        self.watermark_keywords = [
            'gettyimages', 'istock', 'shutterstock', 'adobe stock', 
            'alamy', 'dreamstime', 'photoforus', 'watermark'
        ]

    def audit(self, img_bgr, target_text=None):
        """
        OCR 审计核心接口
        :param img_bgr: 传入的 OpenCV BGR 图像矩阵 (numpy array)
        :param target_text: 可选，用户 Prompt 中期望渲染的文本内容（用于 T2I 文本对齐检查）
        :return: 符合 expert_registry.json 规范的结构化字典
        """
        # RapidOCR 支持直接传入 numpy array (BGR 格式)
        output = self.engine(img_bgr)
        
        boxes = output.boxes     # 形状为 (N, 4, 2) 的坐标数组
        txts = output.txts       # 长度为 N 的文本元组
        scores = output.scores   # 长度为 N 的置信度元组
        elapse = output.elapse   # 推理耗时

        # 初始化证据容器与诊断标签
        evidence_list = []
        has_watermark_pollution = False
        has_text_melting = False
        text_alignment_score = 1.0  # 默认满分（如果没有文字要求或完美对齐）
        
        all_detected_text = ""

        if boxes is not None and len(boxes) > 0:
            for idx, (box, text, score) in enumerate(zip(boxes, txts, scores)):
                text_str = str(text).strip()
                all_detected_text += " " + text_str.lower()
                
                # 诊断 1：检测低置信度的“文字融化/乱码 (Text_Melting_or_Gibberish)”
                # 对应注册表：recognition score < 0.65
                is_melted = float(score) < 0.65
                if is_melted:
                    has_text_melting = True

                # 诊断 2：筛查平台版权水印污染 (Watermark_Pollution)
                is_watermark = any(kw in text_str.lower() for kw in self.watermark_keywords)
                if is_watermark:
                    has_watermark_pollution = True

                evidence_list.append({
                    "block_id": idx + 1,
                    "text_content": text_str,
                    "confidence_score": round(float(score), 4),
                    "bounding_box": box.tolist(),  # (4, 2) 顶点坐标
                    "is_melted": is_melted,
                    "is_watermark_pollution": is_watermark
                })

        # 诊断 3：文本指令对齐校验 (Text_Alignment_Failure)
        # 如果 T2I 任务明确指定了文字，但图里没识别到，或者识别到的文本和目标相差很大
        if target_text:
            target_clean = str(target_text).strip().lower()
            if target_clean not in all_detected_text:
                # 没完全包含，触发对齐失败降低分数
                text_alignment_score = 0.0 if len(evidence_list) == 0 else 0.3
            else:
                text_alignment_score = 1.0

        # 根据各项硬指标决定最终裁决标签 (Verdict)
        verdict = "Normal"
        if has_watermark_pollution:
            verdict = "Watermark_Pollution"
        elif has_text_melting:
            verdict = "Text_Melting_or_Gibberish"
        elif target_text and text_alignment_score < 0.5:
            verdict = "Text_Alignment_Failure"

        # 返回严格符合注册表预期的标准输出
        return {
            "expert_id": "image_text_auditor",
            "model_name": "RapidOCR_PP-OCR_ONNX_Engine",
            "status": "success",
            "metrics": {
                "detected_text_blocks": len(evidence_list),
                "text_alignment_score": text_alignment_score,
                "inference_time_seconds": round(float(elapse), 4) if elapse else 0.0
            },
            "verdict": verdict,
            "evidence": {
                "full_extracted_text": all_detected_text.strip(),
                "blocks": evidence_list
            }
        }