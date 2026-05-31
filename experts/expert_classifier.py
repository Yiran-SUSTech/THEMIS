import os
import json
import time
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image

class FineGrainedClassifier:
    def __init__(self, 
                 model_path="/mnt/afs/zhengmingkai/zyr/THEMIS/new_models/eva02_large_metax_compatible.onnx",
                 label_path="/mnt/afs/zhengmingkai/zyr/THEMIS/imagenet_classes.json",
                 input_size=448,
                 device="maca:0"):
        print(f"[Init] Loading EVA-02 Model (Device: {device})...")
        providers = self._build_providers(device)
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_size = input_size
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._load_labels(label_path)

    @staticmethod
    def _build_providers(device: str):
        if device.startswith("maca:"):
            device_id = int(device.split(":")[1])
            return [('MACAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
        elif device.startswith("cuda:"):
            device_id = int(device.split(":")[1])
            return [('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
        elif device == "cpu":
            return ['CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']

    def _load_labels(self, label_path):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                labels_dict = json.load(f)
            self.imagenet_labels = [labels_dict.get(str(i), f"Class_Index_{i}") for i in range(1000)]
        except Exception as e:
            self.imagenet_labels = [f"Class_Index_{i}" for i in range(1000)]

    def _preprocess(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        w, h = img.size
        min_size = min(w, h)
        img = img.crop(((w - min_size) // 2, (h - min_size) // 2, (w + min_size) // 2, (h + min_size) // 2))
        img = img.resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
        
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_np = img_np.transpose(2, 0, 1)
        return np.expand_dims(img_np, axis=0)

    def audit(self, img_bgr):
        """
        细粒度分类证据提取接口：只吐出前三名预测结果，不判断语义漂移
        """
        input_data = self._preprocess(img_bgr)
        raw_outputs = self.session.run([self.output_name], {self.input_name: input_data})
        logits = raw_outputs[0][0]
        top3_idx = np.argsort(logits)[-3:][::-1]
        
        top_predictions = []
        for rank, idx in enumerate(top3_idx, 1):
            top_predictions.append({
                "rank": rank,
                "class_index": idx,
                "label_name": self.imagenet_labels[idx] if idx < len(self.imagenet_labels) else f"Index_{idx}",
                "logit_score": round(float(logits[idx]), 2)
            })

        return {
            "expert_id": "fine_grained_classifier",
            "model_name": "EVA-02_Large_ONNX_MetaX",
            "status": "success",
            "evidence": {
                "top3_candidates": top_predictions
            }
        }