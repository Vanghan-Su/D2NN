# Version History

本项目采用“每次修改都是一个独立 Git 提交”的保存策略。已发布的 commit 不允许通过 amend、force-push、rebase 或 reset 覆盖。

## Versioning Policy

1. 每次代码、配置、文档或图的修改都创建新的 commit。
2. 每个完成阶段立即推送到 `origin/main`。
3. 已推送 commit 禁止 amend、force-push、rebase 或 reset。
4. 只有功能性代码、物理模型、优化行为或重大实验里程碑才创建 annotated Git tag。
5. 文档、图片、说明、版本记录等小修改只做普通 commit + push，不单独创建 tag。
6. 大型 checkpoint 不直接提交到 Git；如需发布，使用 GitHub Release 或 Git LFS。
7. 当前正式分支为 `main`，远程为 `git@github.com:Vanghan-Su/D2NN.git`。

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

查看最新 tag 之后的全部追加提交：

```powershell
git log v0.3.0..main --oneline
```
