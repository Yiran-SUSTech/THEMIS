import argparse
import os
import torch
import torch.nn.functional as F

# =====================================================================
# 救星补丁：用基础算子等价替代 F.grid_sample，彻底消灭 aten::grid_sampler 算子
# =====================================================================
def open_source_grid_sample_bilinear(input, grid, align_corners=False):
    """
    一个完全基于基础数学算子实现的双线性插值 grid_sample。
    它不产生 'aten::grid_sampler'，因此可以在 Opset 13/14 下完美导出，
    并且完全兼容旧版本的 ONNX Runtime 1.12 和国产 MACA 算子库。
    """
    N, C, IH, IW = input.shape
    _, OH, OW, _ = grid.shape

    # 将 [-1, 1] 的 grid 映射到图像像素坐标 [0, IW-1] 和 [0, IH-1]
    x = grid[..., 0]
    y = grid[..., 1]

    if align_corners:
        x = ((x + 1.0) * 0.5) * (IW - 1)
        y = ((y + 1.0) * 0.5) * (IH - 1)
    else:
        x = ((x + 1.0) * 0.5 * IW) - 0.5
        y = ((y + 1.0) * 0.5 * IH) - 0.5

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    # 边界裁剪
    x0 = torch.clamp(x0, 0, IW - 1)
    x1 = torch.clamp(x1, 0, IW - 1)
    y0 = torch.clamp(y0, 0, IH - 1)
    y1 = torch.clamp(y1, 0, IH - 1)

    # 计算双线性插值的权重
    wa = ((x1.float() - x) * (y1.float() - y)).unsqueeze(1)
    wb = ((x1.float() - x) * (y - y0.float())).unsqueeze(1)
    wc = ((x - x0.float()) * (y1.float() - y)).unsqueeze(1)
    wd = ((x - x0.float()) * (y - y0.float())).unsqueeze(1)

    # 变相实现高级索引 (用 flatten + index_select 替代原生 index_put 以防国产卡不支持)
    input_reshape = input.view(N, C, IH * IW)
    
    # 动态构建一维索引
    batch_idx = torch.arange(N, device=input.device).view(N, 1, 1) * (IH * IW)
    idx_a = (y0 * IW + x0 + batch_idx).view(N, -1)
    idx_b = (y1 * IW + x0 + batch_idx).view(N, -1)
    idx_c = (y0 * IW + x1 + batch_idx).view(N, -1)
    idx_d = (y1 * IW + x1 + batch_idx).view(N, -1)

    input_flat = input.transpose(0, 1).contiguous().view(C, -1)

    # 使用 gather 算子（ONNX 原生完美支持，且各家硬件都极度优化）
    ia = input_flat.gather(1, idx_a.expand(C, -1)).view(C, N, OH, OW).transpose(0, 1)
    ib = input_flat.gather(1, idx_b.expand(C, -1)).view(C, N, OH, OW).transpose(0, 1)
    ic = input_flat.gather(1, idx_c.expand(C, -1)).view(C, N, OH, OW).transpose(0, 1)
    id = input_flat.gather(1, idx_d.expand(C, -1)).view(C, N, OH, OW).transpose(0, 1)

    return wa * ia + wb * ib + wc * ic + wd * id

# 核心黑魔法：运行时强行偷天换日
F.grid_sample = open_source_grid_sample_bilinear
# =====================================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    
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
    attention_mask = tokenized["attention_mask"].bool() 
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).long()
    token_type_ids = torch.zeros_like(input_ids).long()
    
    B, N = input_ids.shape
    text_token_mask = torch.ones((B, N, N), dtype=torch.bool)
    
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

    print("--> Starting ONNX export with Opset 14 and Patching Grid Sample...")
    torch.onnx.export(
        model,
        f=os.path.join(output_dir, "groundingdino.onnx"),
        args=(img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask),
        input_names=["img" , "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
        output_names=["logits", "boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=14 # 降级为 14，完美配合 ONNX Runtime 1.12
    )
    print(f"--> Export success! Saved to {os.path.join(output_dir, 'groundingdino.onnx')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export Grounding DINO Model to IR", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")

    args = parser.parse_args()

    config_file = args.config_file  
    checkpoint_path = args.checkpoint_path  
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(config_file, checkpoint_path, cpu_only=True)
    export_onnx(model, output_dir)