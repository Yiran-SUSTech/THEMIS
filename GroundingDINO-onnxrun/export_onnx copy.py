import argparse
import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    
    #modified config
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


def export_onnx(model, output_dir):
    caption = "the running dog ." 
    tokenized = model.tokenizer([caption], return_tensors="pt")
    input_ids = tokenized["input_ids"]
    # 转换为 long 型以增强国产硬件 MACA 算子库的兼容性
    attention_mask = tokenized["attention_mask"].long() 
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).long()
    token_type_ids = torch.zeros_like(input_ids).long()
    
    # 动态生成匹配分词长度的 3D text_token_mask
    B, N = input_ids.shape
    text_token_mask = torch.ones((B, N, N), dtype=torch.long)
    
    img = torch.randn(1, 3, 800, 1200)

    dynamic_axes={
       "img": {0: "batch_size", 2: "height", 3: "width"},
       "input_ids": {0: "batch_size", 1: "seq_len"},
       "attention_mask": {0: "batch_size", 1: "seq_len"},
       "position_ids": {0: "batch_size", 1: "seq_len"},
       "token_type_ids": {0: "batch_size", 1: "seq_len"},
       "text_token_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len"},       
       "logits": {0: "batch_size"},
       "boxes": {0: "batch_size"}
    }

    print("--> Starting ONNX export with Opset 13 for MACA compatibility...")
    #export onnx model
    torch.onnx.export(
        model,
        f=os.path.join(output_dir, "groundingdino.onnx"),
        args=(img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask),
        input_names=["img" , "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
        output_names=["logits", "boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=13 # 关键修改：降级到 13，适配 ONNX Runtime 1.12.0
    )
    print(f"--> Export success! Saved to {os.path.join(output_dir, 'groundingdino.onnx')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export Grounding DINO Model to IR", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    args = parser.parse_args()

    config_file = args.config_file  
    checkpoint_path = args.checkpoint_path  
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    # 保持 cpu_only=True 避开本地没有 NVIDIA 显卡驱动编译 C++ 算子的限制
    model = load_model(config_file, checkpoint_path, cpu_only=True)
    
    export_onnx(model, output_dir)