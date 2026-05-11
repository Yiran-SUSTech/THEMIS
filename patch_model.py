import onnx

def force_patch(input_path, output_path):
    model = onnx.load(input_path)
    # 强行把版本声明从 17 改成 16
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            print(f"patching {input_path} to opset {opset.version} to 16")
            opset.version = 16
    onnx.save(model, output_path)

# 修改你目前那两个 MobileSAM 模型
base = "new_models/sam1_onnx/machine_learning_models/"
force_patch(base + "mobile_sam.encoder.onnx", base + "mobile_sam.encoder_v16.onnx")
force_patch(base + "mobile_sam.decoder.onnx", base + "mobile_sam.decoder_v16.onnx")