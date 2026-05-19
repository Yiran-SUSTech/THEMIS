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
                 input_size=448):
        """
        初始化 EVA-02 细粒度分类引擎，一次加载，常驻内存
        """
        print(f"[Init] Loading EVA-02 Model to MetaX MACA engine...")
        self.input_size = input_size
        
        # 🚀 完美的 MetaX 硬件加速加速网络
        providers = ['MACAExecutionProvider', 'CPUExecutionProvider']
        
        start_load = time.time()
        self.session = ort.InferenceSession(model_path, providers=providers)
        print(f"[Init] EVA-02 loaded successfully, cost: {time.time() - start_load:.2f} s")
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 载入 ImageNet 1000类 标签映射
        self._load_labels(label_path)

    def _load_labels(self, label_path):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                labels_dict = json.load(f)
            self.imagenet_labels = [labels_dict.get(str(i), f"Class_Index_{i}") for i in range(1000)]
            print(f"[Init] Loaded {len(self.imagenet_labels)} ImageNet labels.")
        except Exception as e:
            print(f"[-] Warning: Failed to load labels: {e}, fallback to numeric indices")
            self.imagenet_labels = [f"Class_Index_{i}" for i in range(1000)]

    def _preprocess(self, img_bgr):
        """将内存中的 OpenCV BGR 矩阵转化为 EVA-02 标准方案 B 448 格式"""
        # 1. BGR 转 RGB 
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 2. 转换为 PIL Image 进居中裁剪（保持你原测试脚本的纯正物理对齐）
        img = Image.fromarray(img_rgb)
        
        w, h = img.size
        min_size = min(w, h)
        img = img.crop(((w - min_size) // 2, (h - min_size) // 2, (w + min_size) // 2, (h + min_size) // 2))
        img = img.resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
        
        # 3. 归一化与减去 ImageNet 均值方差
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        
        # 4. HWC -> CHW 并增加 batch 维度
        img_np = img_np.transpose(2, 0, 1)
        img_np = np.expand_dims(img_np, axis=0)
        return img_np

    def audit(self, img_bgr, target_class=None):
        """
        细粒度身份验证审计接口
        :param img_bgr: 传入的 OpenCV BGR 图像矩阵 (numpy array)
        :param target_class: 本张图原本的目标真实类别（如 "hussar monkey" 或 "bald eagle"）
        :return: 符合 expert_registry.json 规范的结构化字典
        """
        t0 = time.time()
        input_data = self._preprocess(img_bgr)
        
        # 送入 MetaX (MACA) 推理
        raw_outputs = self.session.run([self.output_name], {self.input_name: input_data})
        elapse_ms = (time.time() - t0) * 1000
        
        logits = raw_outputs[0][0]
        # 提取 Top-3 索引
        top3_idx = np.argsort(logits)[-3:][::-1]
        
        top_predictions = []
        is_target_matched = False
        
        for rank, idx in enumerate(top3_idx, 1):
            score = float(logits[idx])
            label_name = self.imagenet_labels[idx] if idx < len(self.imagenet_labels) else f"Index_{idx}"
            
            prediction_item = {
                "rank": rank,
                "class_index": idx,
                "label_name": label_name,
                "logit_score": round(score, 2)
            }
            top_predictions.append(prediction_item)
            
            # 语义匹配校验：如果预测出来的 Top-1 或 Top-3 名字里包含了我们要的类别先验
            if target_class and (target_class.lower() in label_name.lower()):
                # 只要前三名命中了目标，就算是匹配成功
                is_target_matched = True

        # --- 结合注册表进行诊断评估 ---
        # 如果模型给出的 Top-1 预测结果和我们的 target_class 对不上，触发语义漂移裁决
        verdict = "Normal"
        if target_class:
            # 拿到 Top-1 的预测标签
            top1_label = top_predictions[0]["label_name"]
            if target_class.lower() not in top1_label.lower():
                verdict = "Semantic_Drift_or_Misclassification"

        return {
            "expert_id": "fine_grained_classifier",
            "model_name": "EVA-02_Large_ONNX_MetaX",
            "status": "success",
            "metrics": {
                "top1_prediction": top_predictions[0]["label_name"],
                "top1_logit": top_predictions[0]["logit_score"],
                "target_class_is_matched": is_target_matched,
                "inference_time_ms": round(elapse_ms, 2)
            },
            "verdict": verdict,
            "evidence": {
                "top3_candidates": top_predictions
            }
        }