# 这是提供参考示例的 RTDETRMM 剪枝脚本
# 与 pruneMM.py 对齐 —— 复用同一个 YAMLPruneEngine，但默认剪枝范围限制在
# 双模态 ResNet 骨干 + 融合投影层（ConvNormLayer / BasicBlock / Blocks），
# 跳过 AIFI 及其后所有 neck/decoder（详见 RTDETRMM.prune docstring）。
from pathlib import Path

from ultralytics import RTDETRMM

if __name__ == '__main__':
    # 正式剪枝时，建议直接加载已经训练完成的权重，而不是空白 YAML。
    model = RTDETRMM('/path/to/RTDETRMM-r18-mm-mid/weights/best.pt')

    # 剪枝结果会在该目录下生成：
    # 1. pruned.pt          -> 后续 finetrain/val/predict 应优先使用这个
    # 2. pruned.yaml        -> 剪枝后的参考结构说明，只反映层宽变化
    # 3. prune_report.json  -> 逐层结构化剪枝报告
    prune_save_dir = Path('./RTDETRMM-pruned/')
    prune_save_dir.mkdir(parents=True, exist_ok=True)

    model.prune(
        method='l1',       # 可选: 'l1', 'l2', 'lamp', 'bn', 'random'
        ratio=0.30,        # 剪枝比例（双模态骨干通道），目标裁掉 30%
        save_dir=str(prune_save_dir),
        imgsz=640,         # 用于剪枝后前向验证和 GFLOPs 统计
        round_to=8,        # 保留通道数按 8 对齐，便于后续部署
        min_ch=8,          # 单层至少保留的输出通道数
        save_report=True,  # 输出逐层结构化报告到 save_dir/prune_report.json
    )

    # 剪枝后如需验证精度：
    # model = RTDETRMM(str(prune_save_dir / 'pruned.pt'))
    # model.val(data='/path/to/data.yaml', device='0', imgsz=640)
    #
    # 剪枝后微调请使用 finetrainRTMM.py（model.finetrain()）
