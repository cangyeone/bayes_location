# pinn_location

基于PINN的地震走时表建模与位置反演工具集。

## 功能概述

- 使用共享主干 + P/S双头输出的多层感知机 (`TauPSNet`) 来预测走时。
- 支持随机傅里叶特征增强的输入表示，可提升复杂速度结构下的拟合能力。
- 通过 `compute_travel_time_table` 高效计算网格化走时表，批量处理多个震源和台站以充分利用 GPU。
- `gibbs_mh_location_sampler` 基于预测走时与观测值执行 Gibbs-MH 采样，允许部分缺失的 P/S 走时。
- 训练数据集中自动忽略缺失走时（NaN），保持训练稳定性。

## 快速开始

```python
import torch
from pinn_location import (
    TauPSNet,
    TauPSNetConfig,
    TraveltimeDataset,
    TrainingConfig,
    train_taunet,
    compute_travel_time_table,
    Observations,
    SamplerConfig,
    gibbs_mh_location_sampler,
    build_feats,
)

# 构造样例数据
stations = torch.rand(128, 3) * 100.0  # 台站坐标
sources = torch.rand(128, 3) * 100.0   # 地震坐标
travel_times = torch.rand(128, 2)      # 对应的 P/S 走时，可包含 NaN 表示缺失

# 训练走时网络
model = TauPSNet(TauPSNetConfig())
dataset = TraveltimeDataset(stations, sources, travel_times)
train_taunet(model, dataset, TrainingConfig(epochs=10))

# 计算走时表
grid = torch.rand(512, 3) * 100.0
p_table, s_table = compute_travel_time_table(model, stations, grid)

# 进行 Gibbs-MH 采样
t_obs = torch.rand(stations.shape[0], 2)
obs = Observations(stations, t_obs, sigma_p=0.1, sigma_s=0.2)
config = SamplerConfig(step_std=torch.tensor([1.0, 1.0, 1.0]), num_iterations=1000)
initial = torch.tensor([50.0, 50.0, 10.0])

samples = gibbs_mh_location_sampler(
    lambda sta, hypo: model(build_feats(sta, hypo)),
    obs,
    initial,
    config,
)
print(samples.shape)
```

## 依赖

- Python 3.10+
- PyTorch 2.x

## 许可证

MIT
