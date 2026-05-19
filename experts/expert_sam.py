import os
import cv2
import numpy as np
import onnxruntime as ort

class SegmentAnythingExpert:
    def __init__(self, 
                 model_dir="new_models/sam1_onnx/machine_learning_models/",
                 output_dir="/mnt/afs/zhengmingkai/zyr/THEMIS/sam_results_v1"):
        
        encoder_path = os.path.join(model_dir, "mobile_sam.encoder_v16.onnx")
        decoder_path = os.path.join(model_dir, "mobile_sam.decoder_v16.onnx")
        
        print(f"[Init] Loading MobileSAM 1 Model to ONNX Engine...")
        print(f"       Encoder: {encoder_path}")
        print(f"       Decoder: {decoder_path}")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保持与测试脚本一致的 CPUProvider 适配环境
        providers = ['CPUExecutionProvider']
        self.encoder_session = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder_session = ort.InferenceSession(decoder_path, providers=providers)

    def _preprocess(self, img_bgr, input_size=1024):
        """SAM 1 标准预处理：等比例缩放 + Padding 到 1024x1024"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        scale = input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        input_img = np.zeros((input_size, input_size, 3), dtype=np.float32)
        input_img[:new_h, :new_w, :] = img_resized
        
        input_img = input_img / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_img = (input_img - mean) / std
        
        input_img = input_img.transpose(2, 0, 1)[None, ...]
        return input_img.astype(np.float32), scale

    def audit(self, img_bgr, original_image_path, hint_box=None):
        """
        SAM 证据提取接口：生成目标物体的像素分割 Mask，保存并返回本地路径
        :param hint_box: [x1, y1, x2, y2] 形式的引导框。若不提供，则默认点选图片中心。
        """
        orig_h, orig_w = img_bgr.shape[:2]
        
        # 1. 运行 Encoder 提取全局 Embedding
        img_input, scale = self._preprocess(img_bgr)
        encoder_inputs = {self.encoder_session.get_inputs()[0].name: img_input}
        image_embeddings = self.encoder_session.run(None, encoder_inputs)[0]
        
        # 2. 决定点选位置（如果有 DINO 的框，就点选框中心；没有就点选图中心）
        if hint_box and len(hint_box) == 4:
            cx = (hint_box[0] + hint_box[2]) // 2
            cy = (hint_box[1] + hint_box[3]) // 2
        else:
            cx, cy = orig_w // 2, orig_h // 2
            
        input_point = np.array([[cx, cy]], dtype=np.float32)
        input_label = np.array([1], dtype=np.float32)  # 1 代表前景正向点
        
        # 3. 准备 Decoder 的张量形状
        onnx_coord = input_point * scale
        onnx_coord = np.concatenate([onnx_coord, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :]
        
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        
        # 4. 运行 Decoder 生成分割掩膜
        decoder_inputs = {
            "image_embeddings": image_embeddings.astype(np.float32),
            "point_coords": onnx_coord.astype(np.float32),
            "point_labels": onnx_label.astype(np.float32),
            "mask_input": onnx_mask_input.astype(np.float32),
            "has_mask_input": onnx_has_mask_input.astype(np.float32),
            "orig_im_size": np.array([orig_h, orig_w], dtype=np.float32)
        }
        
        masks, scores, _ = self.decoder_session.run(None, decoder_inputs)
        
        # 5. 后处理：选择置信度最高的掩膜并二值化
        best_mask_idx = np.argmax(scores[0])
        best_score = float(scores[0][best_mask_idx])
        mask = masks[0, best_mask_idx]
        mask_binary = (mask > 0).astype(np.uint8) * 255
        
        # 6. 保存 Mask 图片到指定目录
        img_name = os.path.basename(original_image_path)
        mask_filename = os.path.splitext(img_name)[0] + "_mask.png"
        saved_mask_path = os.path.join(self.output_dir, mask_filename)
        cv2.imwrite(saved_mask_path, mask_binary)
        
        return {
            "expert_id": "sam_segmentor",
            "model_name": "MobileSAM_V16_ONNX",
            "status": "success",
            "evidence": {
                "saved_mask_path": saved_mask_path,  # 🚀 Mask 分割图的本地路径证据
                "mask_confidence_score": round(best_score, 4),
                "prompt_point_used": [int(cx), int(cy)]
            }
        }