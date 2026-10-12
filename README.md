# D2NN 自由空间全连接光学神经网络仿真

当前代码实现了第一阶段的可验证仿真内核：

- 复数角谱传播；
- 中心零填充与裁剪；
- 5 层相位-only 前向模型；
- 10 个矩形探测区域的积分光强计算；
- 分类、汇聚效率 eta、对比度；
- NumPy 参考实现和 PyTorch 可微实现；
- 角谱能量守恒、探测区、NumPy/PyTorch 一致性和梯度冒烟测试。

## 理论修正

方案文档第 4 节将第 k 个探测区域的测量量定义为积分光强：

`sk = sum_{(x,y) in Dk} |Udet(x,y)|^2`

第 7 节推导梯度时把同一量写成区域内复振幅之和的模方 `|sum U|^2`。两者只有在区域内所有复振幅同相时才近似一致，一般探测器测量的是逐点光强积分。因此本代码默认采用：

`sk = sum |U|^2`

梯度使用 PyTorch autograd 从该物理定义直接计算。

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
& 'E:\codes\env\STME-main\Scripts\python.exe' scripts\train.py --dataset mnist --data-root data --phase-init zero --padded-size 4096 --limit 8 --max-batches 2 --micro-batch-size 2 --effective-batch-size 4 --epochs 1 --device cuda
```

## 目录

- `src/d2nn/`：物理核、探测器、NumPy 和 PyTorch 模型；
- `tests/`：数值和梯度验证；
- `scripts/`：端到端冒烟测试、显存基准和训练入口；
- `docs/`：方案分析、理论修正、零填充收敛和 MNIST 首轮训练结论。

## 填充策略

- `Npad=2048`：速度优先的历史近似值，有限孔径多层级结果不可信；
- `Npad=4096`：本地训练折中值，micro-batch=2；
- `Npad=6144`：当前本机参考验证值，micro-batch=1；
- 完整推导和实测见 `docs/零填充收敛结论.md`。
