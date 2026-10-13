# 实验报告：{{EXPERIMENT_TITLE}}

> 范本说明：本模板以 `20261013-103219_result_20261013-095617_mnist_1pct_pure_phase_template.md` 为基准。后续实验必须保留可复现信息、数据划分、训练结果、独立测试、光强分布和产物清单。

## 0. 摘要

{{SUMMARY}}

## 1. 实验身份

| 项目 | 值 |
|---|---|
| 实验名称 | {{EXPERIMENT_NAME}} |
| 分支 | `{{BRANCH_NAME}}` |
| 运行标识 | `{{RUN_ID}}` |
| 报告生成时间 | {{REPORT_TIME}} |
| 训练输出目录 | `{{TRAINING_OUTPUT_DIR}}` |

## 2. 可复现信息

- Git 分支：`{{GIT_BRANCH}}`
- Git commit：`{{GIT_COMMIT}}`
- 工作区状态：`{{GIT_STATUS}}`
- Python 环境：{{PYTHON_ENV}}
- 训练种子：`{{TRAIN_SEED}}`
- 测试集抽样种子：`{{TEST_SEED}}`
- 关键代码 SHA256：{{CODE_HASHES}}

## 3. 数据划分

{{DATA_SPLIT_TABLE}}

数据文件必须保存到 `data/experiments/{{RUN_ID}}/`，至少包含 train、validation、test 的图像与标签、`split_manifest.json` 和 `dataset_card.md`。

## 4. 模型与训练配置

{{MODEL_CONFIG}}

## 5. 复现命令

```powershell
{{TRAIN_COMMAND}}
```

## 6. 训练结果

{{TRAINING_HISTORY_TABLE}}

{{BEST_TRAINING_SUMMARY}}

## 7. 独立测试结果

测试集必须从未参与训练和验证的样本中按固定种子随机抽取，并记录索引。

{{TEST_METRICS_TABLE}}

### 7.1 逐样本结果

{{TEST_PREDICTIONS_TABLE}}

### 7.2 探测面光强分布总图

![测试样本探测面光强分布]({{DETECTOR_GRID_IMAGE}})

{{DETECTOR_GRID_DESCRIPTION}}

### 7.3 分类错误示例

![分类错误示例]({{FAILURE_EXAMPLE_IMAGE}})

{{FAILURE_EXAMPLE_DESCRIPTION}}

### 7.4 分类成功示例

![分类成功示例]({{SUCCESS_EXAMPLE_IMAGE}})

{{SUCCESS_EXAMPLE_DESCRIPTION}}

## 8. 产物清单

{{ARTIFACT_TABLE}}

## 9. 复现步骤

{{REPRODUCTION_STEPS}}

## 10. 结论与限制

{{CONCLUSION_AND_LIMITATIONS}}
