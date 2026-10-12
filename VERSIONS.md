# Version History

本项目采用“每次修改都是一个独立 Git 提交”的保存策略。已发布的 commit 不允许通过 amend、force-push、rebase 或 reset 覆盖。

## Versioning Policy

1. 每次代码、配置、文档或图的修改都创建新的 commit。
2. 每个完成阶段立即推送到 `origin/main`。
3. 已推送 commit 禁止 amend、force-push、rebase 或 reset。
4. 重要里程碑创建 annotated Git tag。
5. 大型 checkpoint 不直接提交到 Git；如需发布，使用 GitHub Release 或 Git LFS。
6. 当前正式分支为 `main`，远程为 `git@github.com:Vanghan-Su/D2NN.git`。

## Released Versions

| Tag | Commit | 内容 |
|---|---|---|
| `v0.1.0` | `f4c5401` | 初始 D2NN 仿真工程、角谱传播、探测器、NumPy/PyTorch 模型、测试和首轮图表 |
| `v0.2.0` | `69e1a69` | 层间距改为 5 mm，加入可训练振幅，默认联合优化相位和振幅 |
| `v0.2.1` | `e128d68` | 同步相位/振幅模型文档说明 |
| `v0.3.0` | `c18ca7b` | 增加 5 mm、32 样本、10 epoch 的相位/振幅训练结果 |
| `v0.3.1` | `7dbf612` | 将训练结果图更正为探测面输出光强分布，并移除误放的相位/振幅图 |
| `v0.3.2` | current | 增加版本保存政策和版本清单 |
