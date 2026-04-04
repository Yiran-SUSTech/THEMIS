用远程 claude-sonnet-4-6：
python -m src.agentic_eval.run_single "/home/ronin/THEMIS/test_images/beacon.png" --class-label "Beacon" --output test_eval_c2i_output1.json --planner-model "claude-sonnet-4-6"

用本地 Qwen 3B：
python -m src.agentic_eval.run_single "./test_images/hussar monkey.png" --class-label "Hussar monkey" --output test_eval_c2i_output1.json --planner-model "/home/ronin/THEMIS/models/Qwen2.5-VL-3B-Instruct"

