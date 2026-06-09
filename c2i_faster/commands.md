# 原有无状态模式（完全不变）
python -m c2i.dispatcher --step 1234 --limit 2 --save-pose-viz

# 新的有状态 session 模式
python -m c2i.dispatcher --step 1234 --limit 2 --save-pose-viz --session