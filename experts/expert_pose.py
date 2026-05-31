import os
import cv2
from rtmlib import Custom

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (6, 8), (7, 9),
    (8, 10), (11, 12), (11, 13), (12, 14),
    (13, 15), (14, 16), (5, 11), (6, 12),
]


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

    def _draw_visualization(self, img_bgr, instances_evidence):
        vis = img_bgr.copy()

        for inst in instances_evidence:
            keypoints = inst.get("keypoints", [])
            kpt_coords = {}
            for kpt in keypoints:
                kid = kpt["keypoint_id"]
                x, y = int(round(kpt["x"])), int(round(kpt["y"]))
                conf = kpt["confidence"]
                kpt_coords[kid] = (x, y)

                if conf >= 0.3:
                    cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
                else:
                    cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

                label = f"{kid}:{conf:.2f}"
                cv2.putText(
                    vis, label, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (255, 255, 0), 1, cv2.LINE_AA,
                )

            for i, j in COCO_SKELETON:
                if i in kpt_coords and j in kpt_coords:
                    ci = next((k["confidence"] for k in keypoints if k["keypoint_id"] == i), 0)
                    cj = next((k["confidence"] for k in keypoints if k["keypoint_id"] == j), 0)
                    color = (0, 200, 200) if ci >= 0.3 and cj >= 0.3 else (0, 80, 80)
                    cv2.line(vis, kpt_coords[i], kpt_coords[j], color, 1, cv2.LINE_AA)

        return vis

    def audit(self, img_bgr, save_viz=False, viz_output_path=None):
        """
        姿态估计证据提取接口：吐出实例数、关键点坐标和置信度。
        当 save_viz=True 且 viz_output_path 非空时，保存带关键点编号和置信度的可视化图。
        """
        keypoints, scores = self.model(img_bgr)
        
        instances_evidence = []
        num_instances = len(keypoints)
        
        if num_instances > 0:
            for idx, (inst_kpts, inst_scores) in enumerate(zip(keypoints, scores)):
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

        evidence = {
            "detected_pose_instances": instances_evidence
        }

        if save_viz and viz_output_path and instances_evidence:
            vis = self._draw_visualization(img_bgr, instances_evidence)
            os.makedirs(os.path.dirname(viz_output_path), exist_ok=True)
            cv2.imwrite(viz_output_path, vis)
            evidence["saved_pose_viz_path"] = viz_output_path
            print(f"  [SAVED] Pose visualization -> {os.path.basename(viz_output_path)}")

        return {
            "expert_id": "animal_pose_estimator",
            "model_name": "rtmlib_YOLOX_ViTPose_APT36K",
            "status": "success",
            "raw_metrics": {
                "detected_instances_count": num_instances
            },
            "evidence": evidence
        }