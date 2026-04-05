```shellscript
cd /mnt/afs/zhengmingkai/zyr/THEMIS

python -m src.agentic_eval.run_single "./test_images/hussar_monkey.png" \
    --class-label "Hussar monkey" \
    --output ./outputs/result.json \
    --planner-model /mnt/afs/zhengmingkai/zyr/THEMIS/models/Qwen2.5-VL-3B-Instruct \
    --judge-model /mnt/afs/zhengmingkai/zyr/THEMIS/models/Qwen2.5-VL-3B-Instruct \
    --reflector-model /mnt/afs/zhengmingkai/zyr/THEMIS/models/Qwen2.5-VL-3B-Instruct
```

