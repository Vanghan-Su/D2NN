# 5 mm 层间距与振幅优化变更

日期：2026-10-12

## 1. 几何参数

- 衍射层数量：5；
- 层间间距：`layer_spacing_m = 5.0e-3`；
- 第 4 层到探测面距离：保持 `detector_spacing_m = 10.0e-3`；
- 完整光路长度：30 mm；
- 默认零填充：4096。

按最大传播角计算，30 mm 路径对应的理论支持网格约为 3463，因此 4096 在 5 mm 层间距下首次成为理论足够的默认值。

5 mm 配置的固定相位收敛扫描：

| Npad | 中心场相对 L2 | 最大 sk 相对误差 | eta 误差 |
|---:|---:|---:|---:|
| 2048 | 6.88e-2 | 3.15e-3 | 2.46e-6 |
| 3072 | 1.60e-2 | 1.55e-3 | 2.01e-6 |
| 4096 | 6.13e-3 | 7.16e-4 | 9.85e-7 |
| 6144 | 0 | 0 | 0 |

4096 已可作为当前本地训练和验证配置；6144 继续作为更高精度参考。

## 2. 振幅优化

`TorchD2NN` 新增可选振幅参数：

- `train_amplitude=False`：保持原有纯相位模型，兼容旧 checkpoint；
- `train_amplitude=True`：每层增加 `amplitude_logits`；
- 物理振幅通过 `sigmoid(amplitude_logits)` 映射到 `(0, 1)`；
- 相位和振幅由同一个优化器联合更新；
- `amplitude_init=1.0` 时振幅初始接近 1，但不再是硬固定值。

训练入口默认同时优化相位和振幅：

```powershell
E:\codes\env\STME-main\Scripts\python.exe scripts\train.py ^
  --dataset mnist ^
  --data-root data ^
  --padded-size 4096 ^
  --layer-spacing-mm 5 ^
  --detector-spacing-mm 10 ^
  --phase-init zero ^
  --micro-batch-size 2 ^
  --effective-batch-size 8
```

如需恢复纯相位实验，增加：

```text
--phase-only
```

## 3. 验证

- 18 项单元测试全部通过；
- NumPy 参考模型与 PyTorch 振幅+相位模型前向一致；
- 相位和振幅梯度均有限且非零；
- 默认层间距 5 mm 的配置测试通过；
- 5 mm + 4096 网格的 MNIST 默认相位+振幅训练入口冒烟通过。
