import os
import cv2
from rapidocr import RapidOCR

class ImageTextAuditor:
    def __init__(self, det_model_path='models/Multilingual_PP-OCRv3_det_infer.onnx'):
        print(f"[Init] Loading RapidOCR with local det model: {det_model_path}")
        self.engine = RapidOCR(params={'Det.model_path': det_model_path})

    def audit(self, img_bgr):
        """
        OCR 证据提取接口：只提取，不裁判
        """
        output = self.engine(img_bgr)
        
        boxes = output.boxes     
        txts = output.txts       
        scores = output.scores   
        elapse = output.elapse   

        blocks_evidence = []
        full_text_list = []

        if boxes is not None and len(boxes) > 0:
            for idx, (box, text, score) in enumerate(zip(boxes, txts, scores)):
                text_str = str(text).strip()
                full_text_list.append(text_str)
                
                blocks_evidence.append({
                    "block_id": idx + 1,
                    "text_content": text_str,
                    "confidence_score": round(float(score), 4),
                    "bounding_box": box.tolist()  # (4, 2) 坐标
                })

        return {
            "expert_id": "image_text_auditor",
            "model_name": "RapidOCR_PP-OCR_ONNX_Engine",
            "status": "success",
            "raw_metrics": {
                "detected_text_blocks": len(blocks_evidence),
                "inference_time_seconds": round(float(elapse), 4) if elapse else 0.0
            },
            "evidence": {
                "full_extracted_text": " ".join(full_text_list),
                "blocks": blocks_evidence
            }
        }