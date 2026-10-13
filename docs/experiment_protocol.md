# 标准实验协议：MNIST 纯相位法 1% 数据

本协议以 `20261013-095617_mnist_1pct_pure_phase` 实验为基准，后续纯相位法对比实验默认复用同一数据比例、训练参数、结果目录结构、数据集格式和报告模板。

## 默认训练协议

| 项目 | 默认值 |
|---|---:|
| 数据集 | MNIST |
| 训练集比例 | 1% |
| 训练集数量 | 600 |
| 验证集比例 | 1% |
| 验证集数量 | 100 |
| 独立测试集数量 | 50 |
| Epochs | 20 |
| 有效 batch size | 32 |
| micro-batch size | 8 |
| 梯度累积步数 | 4 |
| 学习率 | 1e-2 |
| 计算网格 | 2048 x 2048 |
| 层间距 | 5 mm |
| 末层到探测面 | 10 mm |
| 训练/相位种子 | 20261012 |
| 测试集抽样种子 | 20261013 |

`scripts/train.py` 已以上述参数作为默认值，标准训练命令为：

```powershell
python scripts/train.py --device cuda
```

## 数据划分

- 训练集：MNIST train 索引 `0..599`；
- 验证集：MNIST test 索引 `0..99`；
- 测试集：从 MNIST test 索引 `100..9999` 中按种子 `20261013` 随机无放回抽取 50 张；
- 数据集固定保存到 `data/experiments/<run-id>/`，必须包含 train、validation、test 图像与标签、`split_manifest.json` 和 `dataset_card.md`。

基准数据集：

`data/experiments/20261013-095617_mnist_1pct_pure_phase`

## 结果与报告

训练输出必须包含：

- `history.json`；
- `_best.pt` 和 `_last.pt`；
- `training.log`；
- 相位分布图；
- 独立测试集指标 JSON、逐样本 CSV 和混淆矩阵；
- 50 张测试样本的探测面光强分布总图；
- 分类失败和分类成功示例图；
- 完整 Markdown 实验报告。

报告使用 [experiment_report_template.md](./templates/experiment_report_template.md) 的章节结构。探测区域使用青色方框和数字 `0-9` 标注；分类正确时，预测数字对应方框改为亮绿色。

## 基准报告

`docs/20261013-103219_result_20261013-095617_mnist_1pct_pure_phase_template.md`
