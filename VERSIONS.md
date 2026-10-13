# Version History

本项目采用“每次修改都是一个独立 Git 提交”的保存策略。已发布的 commit 不允许通过 amend、force-push、rebase 或 reset 覆盖。

## Versioning Policy

1. 每次代码、配置、文档或图的修改都创建新的 commit。
2. 每个完成阶段立即推送到对应上游分支；`main` 仅保留已经确认合入的正式路线。
3. 已推送 commit 禁止 amend、force-push、rebase 或 reset。
4. 只有功能性代码、物理模型、优化行为或重大实验里程碑才创建 annotated Git tag。
5. 文档、图片、说明、版本记录等小修改只做普通 commit + push，不单独创建 tag。
6. 大型 checkpoint 不直接提交到 Git；如需发布，使用 GitHub Release 或 Git LFS。
7. 当前开发分支为 `纯相位法`，远程仓库为 `git@github.com:Vanghan-Su/D2NN.git`；`main` 保存历史联合优化实现。

## Pure-Phase Branch

正式方案分支名：`纯相位法`。

- 只有 5 个衍射层的相位可训练，振幅透过率固定为 1；
- 默认层间距 5 mm，末层到探测面 10 mm；
- 默认计算网格改为 2048 x 2048；
- 训练入口默认关闭振幅优化，输出 `<optimization>` 标记为 `纯相位法`；
- `--optimize-amplitude` 仅用于可选的复振幅预训练，不作为最终器件分支。

## Naming Convention

所有 docs 和 outputs 名称使用可按时间排序的前缀：

```text
YYYYMMDD-HHMMSS_<类别>_<修改内容>
```

- 时间使用本机时区；文件名按字典序即为时间顺序。
- 文档：`YYYYMMDD-HHMMSS_<analysis|review|convergence|training|change|result>_<内容>.md`
- 图片：`YYYYMMDD-HHMMSS_<figure|result>_<实验参数>_<内容>.png`
- 输出目录：`YYYYMMDD-HHMMSS_<dataset>_<geometry>_<optimization>_<samples>samples_<epochs>epochs`
- 默认 `<optimization>`：`纯相位法`
- 仅预训练 `<optimization>`：`复振幅预训练`
- checkpoint：`YYYYMMDD-HHMMSS_<run-description>_<last|best|epochNNN>.pt`
- CSV：`YYYYMMDD-HHMMSS_<analysis>_<parameters>.csv`

## Released Versions

| Tag | Commit | 内容 |
|---|---|---|
| `v0.1.0` | `f4c5401` | 初始 D2NN 仿真工程、角谱传播、探测器、NumPy/PyTorch 模型、测试和首轮图表 |
| `v0.2.0` | `69e1a69` | 层间距改为 5 mm，加入可训练振幅，默认联合优化相位和振幅 |
| `v0.3.0` | `c18ca7b` | 增加 5 mm、32 样本、10 epoch 的相位/振幅训练结果 |

## Untagged Main-branch Changes

以下改动保留在 `main` 的提交历史中，但不单独创建 tag：

- 文档措辞和模型说明修正；
- 输出面光强图替换和错误图片删除；
- `VERSIONS.md` 和版本保存政策更新；
- 后续同类的小型文档、图片和说明修改。

## Untagged Pure-Phase Branch Changes

以下改动属于 `纯相位法` 分支，待验证并确认后再决定是否合入 `main` 或创建版本 tag：

- 将默认训练模式从“相位 + 振幅”恢复为纯相位；
- 将默认计算网格和训练入口默认值调整为 2048；
- 将方案分支名称写入模型元数据和输出命名；
- 保留复振幅优化代码，但明确标记为预训练用途；
- 增加训练进度日志、`history.json` 和最佳 checkpoint 保存。

查看分支上的改动：

```powershell
git log main..纯相位法 --oneline
```
