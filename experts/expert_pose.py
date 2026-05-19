import os
from rtmlib import Custom

class AnimalPoseEstimator:
    def __init__(self, 
                 model_dir='/mnt/afs/zhengmingkai/zyr/THEMIS/new_models',
                 device='cuda', 
                 backend='onnxruntime'):
        
        det_local = os.path.join(model_dir, 'yolox_s.onnx')
        pose_local = os.path.join(model_dir, 'vitpose-b-apt36k.onnx')
        
        print(f"[Init] Loading rtmlib Custom Pose Model...")
        print(f"       Det: {det_local}")
        print(f"       Pose: {pose_local}")
        
        # 长驻内存初始化
        self.model = Custom(
            det_class='YOLOX',
            det_mode='multiclass',
            det=det_local,
            det_input_size=(640, 640),
            pose_class='ViTPose',
            pose=pose_local,
            pose_input_size=(192, 256),
            backend=backend,
            device=device
        )

    def audit(self, img_bgr):
        """
        姿态估计证据提取接口：只吐出实例数、关键点坐标和置信度，不画图，不判断畸变
        """
        # 推理：keypoints shape: (N, K, 2), scores shape: (N, K)
        keypoints, scores = self.model(img_bgr)
        
        instances_evidence = []
        num_instances = len(keypoints)
        
        if num_instances > 0:
            for idx, (inst_kpts, inst_scores) in enumerate(zip(keypoints, scores)):
                # 将单个实例的关键点包组装起来
                kpt_list = []
                for kpt_idx, (coord, score) in enumerate(zip(inst_kpts, inst_scores)):
                    kpt_list.append({
                        "keypoint_id": kpt_idx,
                        "x": round(float(coord[0]), 2),
                        "y": round(float(coord[1]), 2),
                        "confidence": round(float(score), 4)
                    })
                
                instances_evidence.append({
                    "instance_id": idx + 1,
                    "total_keypoints_polled": len(kpt_list),
                    "average_instance_confidence": round(float(inst_scores.mean()), 4),
                    "keypoints": kpt_list
                })

        return {
            "expert_id": "animal_pose_estimator",
            "model_name": "rtmlib_YOLOX_ViTPose_APT36K",
            "status": "success",
            "raw_metrics": {
                "detected_instances_count": num_instances
            },
            "evidence": {
                "detected_pose_instances": instances_evidence
            }
        }