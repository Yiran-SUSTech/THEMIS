#!/usr/bin/env python3
"""
Test all downloaded models for proper loading
"""
import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

MODEL_DIR = '/mnt/afs/zhengmingkai/zyr/THEMIS/models'

def test_model(name, test_func):
    """Test a single model"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    try:
        test_func()
        print(f"[PASS] {name} loaded successfully!")
        return True
    except Exception as e:
        print(f"[FAIL] {name} loading failed: {e}")
        return False

def test_qwen_3b():
    """Test Qwen2.5-VL-3B"""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model_path = f'{MODEL_DIR}/Qwen2.5-VL-3B-Instruct'
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        trust_remote_code=True,
        device_map='cpu',
        torch_dtype='auto'
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

def test_qwen_7b():
    """Test Qwen2.5-VL-7B"""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model_path = f'{MODEL_DIR}/Qwen2.5-VL-7B-Instruct'
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        trust_remote_code=True,
        device_map='cpu',
        torch_dtype='auto'
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

def test_clip():
    """Test CLIP"""
    from transformers import CLIPModel, CLIPProcessor
    model_path = f'{MODEL_DIR}/clip-vit-base-patch32'
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path)
    print(f"  Model type: CLIP-ViT-B/32")

def test_efficientnet():
    """Test EfficientNet"""
    import torch
    import timm
    model_path = f'{MODEL_DIR}/efficientnetv2_s_in21k.pth'
    model = timm.create_model('tf_efficientnetv2_s.in21k', num_classes=1000)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    print(f"  Model type: EfficientNetV2-S")

def test_yolo():
    """Test YOLO"""
    from ultralytics import YOLO
    yolo_dir = f'{MODEL_DIR}/yolo'
    models = ['yolo11n.pt', 'yolo11n-pose.pt', 'yolo11m-pose.pt']
    for m in models:
        path = os.path.join(yolo_dir, m)
        if os.path.exists(path):
            model = YOLO(path)
            print(f"  [OK] {m}")
        else:
            print(f"  [MISSING] {m} not found")

def test_places365():
    """Test Places365/ResNet18"""
    import torch
    import torchvision.models as models
    model_path = f'{MODEL_DIR}/places365/resnet18_imagenet.pt'
    model = models.resnet18()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"  Model type: ResNet18 (ImageNet)")

def test_rmbg():
    """Test RMBG-2.0"""
    from transformers import AutoModelForImageSegmentation, AutoImageProcessor
    model_path = f'{MODEL_DIR}/RMBG-2.0'
    processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
    print(f"  Model type: RMBG-2.0 (Background Segmentation)")

def test_iqa():
    """Test IQA models"""
    import pyiqa
    import warnings
    warnings.filterwarnings('ignore')
    
    metrics = ['musiq', 'niqe']
    for metric in metrics:
        try:
            m = pyiqa.create_metric(metric, device='cpu')
            print(f"  [OK] {metric}")
        except Exception as e:
            print(f"  [FAIL] {metric}: {e}")
    
    print("  Testing MANIQA...")
    try:
        m = pyiqa.create_metric('maniqa', device='cpu')
        print(f"  [OK] maniqa")
    except Exception as e:
        print(f"  [FAIL] maniqa: {e}")

def test_gpu():
    """Test GPU"""
    import torch
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

def main():
    print("="*60)
    print("THEMIS Model Test Script")
    print("="*60)
    print(f"Model directory: {MODEL_DIR}")
    
    results = {}
    
    results['Qwen2.5-VL-3B'] = test_model("Qwen2.5-VL-3B-Instruct", test_qwen_3b)
    results['Qwen2.5-VL-7B'] = test_model("Qwen2.5-VL-7B-Instruct", test_qwen_7b)
    results['CLIP'] = test_model("CLIP-ViT-B/32", test_clip)
    results['EfficientNet'] = test_model("EfficientNetV2-S", test_efficientnet)
    results['YOLO'] = test_model("YOLO Models", test_yolo)
    results['Places365'] = test_model("Places365/ResNet18", test_places365)
    results['RMBG-2.0'] = test_model("RMBG-2.0", test_rmbg)
    results['IQA'] = test_model("IQA Models", test_iqa)
    
    test_model("GPU Status", test_gpu)
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\nAll models loaded successfully!")
        return 0
    else:
        print("\nSome models failed to load, please check.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
