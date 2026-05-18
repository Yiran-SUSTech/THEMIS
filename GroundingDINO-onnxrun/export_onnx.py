import argparse
import os
import torch
import torch.nn.functional as F

# =====================================================================
# 🚀 救星补丁 v2.0：完美对齐官方接口的数学等价 grid_sample
# =====================================================================
def open_source_grid_sample_bilinear(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    100% 严格等效于 PyTorch 原生 F.grid_sample 的双线性插值实现（padding_mode='zeros'）。
    专为可变形注意力（Deformable Attention）特殊的 4D Tensor 拓扑设计。
    完全由基础 ONNX 算子构成，在 Opset 14 下可无痛导出，且完美兼容沐曦 MACA 算力。
    """
    # 严格获取输入的各个维度
    # input: [B, C, H, W], grid: [B, H_out, W_out, 2]
    shape = input.shape
    N = shape[0]
    C = shape[1]
    IH = shape[2]
    IW = shape[3]
    
    _, H_out, W_out, _ = grid.shape

    x = grid[..., 0]
    y = grid[..., 1]

    # 根据 align_corners 的标准数学定义进行坐标映射
    if align_corners:
        x = ((x + 1.0) * 0.5) * (IW - 1)
        y = ((y + 1.0) * 0.5) * (IH - 1)
    else:
        x = ((x + 1.0) * 0.5 * IW) - 0.5
        y = ((y + 1.0) * 0.5 * IH) - 0.5

    # 计算插值所需的四个邻域像素坐标
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    # 计算双线性权重系数
    wa = ((x1 - x) * (y1 - y)).unsqueeze(1) # [B, 1, H_out, W_out]
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # 构造边界掩码（用于处理 padding_mode='zeros' 的填充边界）
    mask_x0 = (x0 >= 0) & (x0 < IW)
    mask_x1 = (x1 >= 0) & (x1 < IW)
    mask_y0 = (y0 >= 0) & (y0 < IH)
    mask_y1 = (y1 >= 0) & (y1 < IH)

    mask_a = (mask_x0 & mask_y0).unsqueeze(1)
    mask_b = (mask_x0 & mask_y1).unsqueeze(1)
    mask_c = (mask_x1 & mask_y0).unsqueeze(1)
    mask_d = (mask_x1 & mask_y1).unsqueeze(1)

    # 边界安全裁剪，防止下方的高级索引越界崩溃
    x0 = torch.clamp(x0, 0, IW - 1).long()
    x1 = torch.clamp(x1, 0, IW - 1).long()
    y0 = torch.clamp(y0, 0, IH - 1).long()
    y1 = torch.clamp(y1, 0, IH - 1).long()

    # 构造批次一维线性基础索引，用于绕过 ONNX 不支持对多维 Tensor 直接进行多维索引的硬伤
    # 这是最符合可变形注意力多维度输入特性的矩阵展开方式
    batch_idx = torch.arange(N, device=input.device).view(N, 1, 1)
    
    # 抽取邻域特征 (使用 where 算子配合一维展平索引，在 ONNX 1.12.0 / MACA 上是最稳的)
    def gather_feat(y_coords, x_coords, mask):
        # 将 [B, C, H, W] 展平成 [B, C, H*W]
        input_flat = input.view(N, C, IH * IW)
        # 计算平面一维索引
        flat_idx = y_coords * IW + x_coords
        # 扩展索引到通道维度：[B, C, H_out, W_out]
        flat_idx_expanded = flat_idx.unsqueeze(1).expand(-1, C, -1, -1).reshape(N, C, -1)
        # 抽取特征
        feat = input_flat.gather(2, flat_idx_expanded).view(N, C, H_out, W_out)
        # 如果超出原始图像边界，则填充为 0
        return torch.where(mask, feat, torch.zeros_like(feat))

    ia = gather_feat(y0, x0, mask_a)
    ib = gather_feat(y1, x0, mask_b)
    ic = gather_feat(y0, x1, mask_c)
    id = gather_feat(y1, x1, mask_d)

    # 加权融合得到结果
    return wa * ia + wb * ib + wc * ic + wd * id

# 🛠️ 运行时黑魔法偷换
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

    print("--> Starting ONNX export with Opset 14 and Fixed Interface Patching...")
    torch.onnx.export(
        model,
        f=os.path.join(output_dir, "groundingdino.onnx"),
        args=(img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask),
        input_names=["img" , "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
        output_names=["logits", "boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=14 
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