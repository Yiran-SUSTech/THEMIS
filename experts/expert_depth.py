import os
import cv2
import numpy as np
import onnxruntime as ort

class MonocularDepthEstimator:
    def __init__(self, 
                 model_path="/mnt/afs/zhengmingkai/zyr/THEMIS/new_models/depth_anything_v1_onnx/onnx/model_fp16.onnx",
                 output_dir="/mnt/afs/zhengmingkai/zyr/THEMIS/depth_results_v1"):
        
        print(f"[Init] Loading Depth Anything V1 Model to MetaX MACA engine...")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 绑定国产 MetaX MACA 硬件提供商
        providers = [('MACAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_node = self.session.get_inputs()[0]
        
        # 解析模型所需的固定输入宽高（例如 518x518）
        def get_dim(dim):
            return dim if isinstance(dim, int) and dim > 0 else 518
        self.input_h = get_dim(self.input_node.shape[2])
        self.input_w = get_dim(self.input_node.shape[3])

    def audit(self, img_bgr, original_image_path):
        """
        深度估计证据提取接口：生成并保存深度图，返回存储路径与深度统计分布
        :param img_bgr: 已经读入内存的BGR图像矩阵
        :param original_image_path: 原图路径（用于生成对应的深度图文件名）
        """
        orig_h, orig_w = img_bgr.shape[:2]
        
        # 1. 颜色转换与预处理 Resize
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_w, self.input_h))
        
        # 2. 归一化并指定 float32
        img_data = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        
        # 3. 维度转换 [H, W, C] -> [1, C, H, W]
        img_data = img_data.transpose(2, 0, 1)[None, ...].astype(np.float32)
        
        # 4. MetaX 硬件加速推理
        depth_raw = self.session.run(None, {self.input_node.name: img_data})[0]
        
        # 5. 后处理：恢复至原图比例并归一化到 0-255 灰度空间
        depth_resized = cv2.resize(depth_raw[0], (orig_w, orig_h))
        d_min, d_max = depth_resized.min(), depth_resized.max()
        depth_norm = ((depth_resized - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        
        # 6. 计算深度基础统计学数据（留给 Reflector 研判是否存在深度值异常或空洞）
        avg_depth = float(depth_resized.mean())
        std_depth = float(depth_resized.std())
        
        # 7. 动态拼装输出文件名，确保与原图片一一对应
        img_name = os.path.basename(original_image_path)
        depth_filename = os.path.splitext(img_name)[0] + "_depth.png"
        saved_depth_path = os.path.join(self.output_dir, depth_filename)
        
        # 持久化深度图到本地磁盘
        cv2.imwrite(saved_depth_path, depth_norm)
        
        return {
            "expert_id": "monocular_depth_estimator",
            "model_name": "Depth_Anything_V1_FP16_MACA",
            "status": "success",
            "raw_metrics": {
                "input_resolution": f"{self.input_w}x{self.input_h}",
                "output_resolution": f"{orig_w}x{orig_h}"
            },
            "evidence": {
                "saved_depth_map_path": saved_depth_path,  # 🚀 灰度深度图的本地路径证据
                "depth_statistics": {
                    "min_value": round(float(d_min), 4),
                    "max_value": round(float(d_max), 4),
                    "mean_value": round(avg_depth, 4),
                    "std_deviation": round(std_depth, 4)
                }
            }
        }