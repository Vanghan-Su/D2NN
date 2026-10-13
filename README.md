# D2NN 自由空间全连接光学神经网络仿真（纯相位法）

当前分支 `纯相位法` 按《自由空间全连接光学神经网络——仿真方案（纯相位版）》实现：

- 5 个衍射层，仅优化相位 `phi_l(m,n)`，振幅透过率固定为 `a_l(m,n) = 1`；
- 层间距 5 mm，第 4 层到探测面距离 10 mm；
- 400 x 400 物理采样，默认计算网格 2048 x 2048；
- 复数角谱传播、中心零填充与有限孔径裁剪；
- 10 个矩形探测区域的逐点积分光强、分类、汇聚效率 eta 和对比度；
- NumPy 参考实现与 PyTorch complex128/float64 可微实现；
- 角谱能量守恒、探测区、NumPy/PyTorch 一致性和梯度测试。

复振幅调制没有作为正式优化路线保留。`--optimize-amplitude` 仅供方案第 9 节建议的“复振幅预训练”使用，预训练结束后仍必须投影到振幅为 1 的纯相位掩模，并单独训练和评估最终纯相位模型。

## 理论修正

方案文档第 4 节将探测量定义为积分光强：

`sk = sum_{(x,y) in Dk} |Udet(x,y)|^2`

第 7 节推导梯度时使用了复振幅之和的模方 `|sum U|^2`。两者在区域内相位不一致时并不相等，因此本代码默认采用逐点光强积分：

`sk = sum |U|^2`

梯度由 PyTorch autograd 从该物理定义直接计算。

## 本地运行环境

已验证可用的环境：

- Python 3.8.2；
- torch 2.2.2+cu121；
- torchvision 0.17.2+cu121；
- NumPy 1.24.4；
- SciPy 1.10.1；
- GPU：NVIDIA GeForce RTX 3060 Laptop，6 GB；
- PyTorch CUDA 可用。

运行：

```powershell
& 'E:\codes\env\STME-main\Scripts\python.exe' -m unittest discover -s tests -v
& 'E:\codes\env\STME-main\Scripts\python.exe' scripts\smoke_test.py
& 'E:\codes\env\STME-main\Scripts\python.exe' scripts\train.py --dataset mnist --data-root data --phase-only --phase-init random --padded-size 2048 --layer-spacing-mm 5 --detector-spacing-mm 10 --limit 8 --max-batches 2 --micro-batch-size 2 --effective-batch-size 16 --epochs 1 --device cuda
```

`--phase-only` 是默认行为，在正式实验中显式写出以固定分支语义。按数据集比例训练时使用 `--dataset-fraction 0.01`，训练集和验证集分别取各自总数的 1%。

标准实验协议见 [docs/experiment_protocol.md](docs/experiment_protocol.md)，后续对比实验应复用其中的数据划分、训练参数、数据集格式和报告模板。

## 目录

- `src/d2nn/`：物理核、探测器、NumPy 和 PyTorch 模型；
- `tests/`：数值、接口和梯度验证；
- `scripts/`：端到端冒烟测试、显存基准、绘图和训练入口；
- `docs/`：方案分析、理论修正、零填充收敛、训练结果和纯相位分支变更记录。

## 填充策略

- `Npad=2048`：纯相位方案的默认值，满足 10 mm 单段传播的两条采样判据；
- `Npad=4096`：有限孔径、多层传播的高精度配置，micro-batch=2，资源允许时用于最终验证；
- `Npad=6144`：更高精度参考配置，micro-batch=1；
- 5 mm 几何的收敛实测见 `docs/20261012-170839_convergence_零填充收敛结论.md`。

## 优化分支

正式训练分支名为 `纯相位法`：

- `TorchD2NN` 默认不创建 `amplitude_logits`；
- `scripts/train.py` 默认关闭振幅优化；
- 训练输出目录中的 `<optimization>` 使用 `纯相位法`；
- 若显式使用 `--optimize-amplitude`，输出标记为 `复振幅预训练`，不能作为最终器件结果。

## 训练结果

- [5 mm 层间距历史相位/振幅训练结果](docs/20261012-172422_result_5mm相位振幅训练结果.md)：属于 `main` 上的历史联合优化结果，不替代纯相位法结果。

## 命名规范

文档和实验输出采用可排序的时间戳前缀：

`YYYYMMDD-HHMMSS_<类别>_<修改内容>`。

- 文档：`docs/YYYYMMDD-HHMMSS_<类别>_<内容>.md`
- 图片：`docs/assets/YYYYMMDD-HHMMSS_<figure|result>_<实验参数>_<内容>.png`
- 输出目录：`outputs/YYYYMMDD-HHMMSS_<dataset>_<geometry>_<optimization>_<samples>samples_<epochs>epochs`
- 默认优化标记：`纯相位法`
- 仅预训练优化标记：`复振幅预训练`
- checkpoint：`YYYYMMDD-HHMMSS_<run-description>_<last|best|epochNNN>.pt`

## 版本保存

每次修改都会创建新的 Git commit 并推送到对应远端分支，不覆盖已发布历史。只有功能性代码、物理模型、优化行为或重大实验里程碑才创建 tag。完整版本记录见 [VERSIONS.md](VERSIONS.md)。
