全本地运行示例：
python -m src.agentic_eval.run_single "/home/ronin/THEMIS/test_images/beacon.png" --class-label "Beacon" --output test_eval_c2i_output1.json --planner-model "/home/ronin/THEMIS/models/Qwen2.5-VL-3B-Instruct" --judge-model "/home/ronin/THEMIS/models/Qwen2.5-VL-3B-Instruct" --reflector-model "/home/ronin/THEMIS/models/Qwen2.5-VL-7B-Instruct"

全本地细粒度类别示例：
python -m src.agentic_eval.run_single "./test_images/hussar monkey.png" --class-label "Hussar monkey" --output test_eval_c2i_output1.json --planner-model "/home/ronin/THEMIS/models/Qwen2.5-VL-3B-Instruct" --judge-model "/home/ronin/THEMIS/models/Qwen2.5-VL-3B-Instruct" --reflector-model "/home/ronin/THEMIS/models/Qwen2.5-VL-7B-Instruct"

