#!/usr/bin/env python3
"""
测试所有下载的模型是否能正常加载
"""
import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

MODEL_DIR = '/mnt/afs/zhengmingkai/zyr/THEMIS/models'

def test_model(name, test_func):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print('='*60)
    try:
        test_func()
        print(f"✓ {name} 加载成功!")
        return True
    except Exception as e:
        print(f"✗ {name} 加载失败: {e}")
        return False

def test_qwen_3b():
    """测试 Qwen2.5-VL-3B"""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model_path = f'{MODEL_DIR}/Qwen2.5-VL-3B-Instruct'
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        trust_remote_code=True,
        device_map='cpu',
        torch_dtype='auto'
    )
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

def test_qwen_7b():
    """测试 Qwen2.5-VL-7B"""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model_path = f'{MODEL_DIR}/Qwen2.5-VL-7B-Instruct'
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        trust_remote_code=True,
        device_map='cpu',
        torch_dtype='auto'
    )
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

def test_clip():
    """测试 CLIP"""
    from transformers import CLIPModel, CLIPProcessor
    model_path = f'{MODEL_DIR}/clip-vit-base-patch32'
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path)
    print(f"  模型类型: CLIP-ViT-B/32")

def test_efficientnet():
    """测试 EfficientNet"""
    import torch
    import timm
    model_path = f'{MODEL_DIR}/efficientnetv2_s_in21k.pth'
    model = timm.create_model('tf_efficientnetv2_s.in21k', num_classes=1000)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    print(f"  模型类型: EfficientNetV2-S")

def test_yolo():
    """测试 YOLO"""
    from ultralytics import YOLO
    yolo_dir = f'{MODEL_DIR}/yolo'
    models = ['yolo11n.pt', 'yolo11n-pose.pt', 'yolo11m-pose.pt']
    for m in models:
        path = os.path.join(yolo_dir, m)
        if os.path.exists(path):
            model = YOLO(path)
            print(f"  ✓ {m}")
        else:
            print(f"  ✗ {m} 不存在")

def test_places365():
    """测试 Places365/ResNet18"""
    import torch
    import torchvision.models as models
    model_path = f'{MODEL_DIR}/places365/resnet18_imagenet.pt'
    model = models.resnet18()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"  模型类型: ResNet18 (ImageNet)")

def test_rmbg():
    """测试 RMBG-2.0"""
    from transformers import AutoModelForImageSegmentation, AutoImageProcessor
    model_path = f'{MODEL_DIR}/RMBG-2.0'
    processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
    print(f"  模型类型: RMBG-2.0 (背景分割)")

def test_iqa():
    """测试 IQA 模型"""
    import pyiqa
    import warnings
    warnings.filterwarnings('ignore')
    
    metrics = ['musiq', 'niqe']
    for metric in metrics:
        try:
            m = pyiqa.create_metric(metric, device='cpu')
            print(f"  ✓ {metric}")
        except Exception as e:
            print(f"  ✗ {metric}: {e}")
    
    print("  测试 MANIQA...")
    try:
        m = pyiqa.create_metric('maniqa', device='cpu')
        print(f"  ✓ maniqa")
    except Exception as e:
        print(f"  ✗ maniqa: {e}")

def test_gpu():
    """测试 GPU"""
    import torch
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

def main():
    print("="*60)
    print("THEMIS 模型测试脚本")
    print("="*60)
    print(f"模型目录: {MODEL_DIR}")
    
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
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("\n所有模型加载成功!")
        return 0
    else:
        print("\n部分模型加载失败，请检查")
        return 1

if __name__ == '__main__':
    sys.exit(main())
