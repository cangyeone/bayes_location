<p align="center">
  <img src="docs/assets/logo.png" alt="SeismicX-System 标志" width="640">
</p>

# Bayes Location：走时模型训练、贝叶斯地震定位与目录质量控制

[English](README.md) | **中文说明**

本项目将监督训练的 P/S 波神经网络走时代理嵌入稳健贝叶斯定位框架，输出震源位置、发震时刻、后验样本和不确定性指标，并据此对候选地震排序和筛选。统一命令入口为 `python -m bayesloc`。

本页介绍论文、复现材料和主要使用流程。完整参数、输入输出字段及实现细节见[英文主说明](README.md)、[算法说明](docs/ALGORITHM.md)、[迁移指南](docs/MIGRATION.md)和[已有验证](docs/VALIDATION.md)。

## 论文与引用

**Robust Bayesian Location and Event-Level Quality Control for Automated Earthquake Catalogs**

- 作者：Ziye Yu、Xin Liu、Yuqi Cai、Zemin Liu、Pingping Wu、Jiayan Tan、Jinqing Sun。
- 期刊：*Bulletin of the Seismological Society of America*，2026。
- DOI：[10.1785/0120260196](https://doi.org/10.1785/0120260196)。

论文结合 Student-t 残差、潜在正常值/异常值指示变量、监督走时代理和基于后验宽度的事件筛选，同时处理定位稳健性、震相可靠性和事件质量。使用该方法开展研究时，请引用：

> Yu, Z., Liu, X., Cai, Y., Liu, Z., Wu, P., Tan, J., and Sun, J. (2026). **Robust Bayesian Location and Event-Level Quality Control for Automated Earthquake Catalogs.** *Bulletin of the Seismological Society of America*. [doi:10.1785/0120260196](https://doi.org/10.1785/0120260196).

```bibtex
@article{yu2026robust,
  author  = {Yu, Ziye and Liu, Xin and Cai, Yuqi and Liu, Zemin and
             Wu, Pingping and Tan, Jiayan and Sun, Jinqing},
  title   = {{Robust Bayesian Location and Event-Level Quality Control for Automated Earthquake Catalogs}},
  journal = {Bulletin of the Seismological Society of America},
  year    = {2026},
  doi     = {10.1785/0120260196},
  url     = {https://doi.org/10.1785/0120260196}
}
```

上述信息依据作者校样，暂不填写仍属占位或临时信息的卷号、期号和页码。[CITATION.cff](CITATION.cff) 提供机器可读引用信息。早期预印本见 [arXiv:2512.06407](https://arxiv.org/abs/2512.06407)。

## 论文可复现材料

论文的可复现材料集中保存在 **[Hugging Face：cangyeone/robust_bayes_location](https://huggingface.co/cangyeone/robust_bayes_location)**。

| 入口 | 用途 |
|---|---|
| 当前 GitHub 仓库 | 持续维护的 Python 包、命令行流程和虚构数据示例，便于在自己的数据上应用方法 |
| [Hugging Face 复现材料仓库](https://huggingface.co/cangyeone/robust_bayes_location) | 论文对应的实验脚本、已公开合成观测与真值、三维速度结构、监督网络和 EikoNet 权重、固定版本的比较方法、配置、诊断与汇总结果 |
| [REPRODUCE.md](https://huggingface.co/cangyeone/robust_bayes_location/blob/main/REPRODUCE.md) | 复现入口、环境与实验配置要求 |
| [ARTIFACT_INDEX.md](https://huggingface.co/cangyeone/robust_bayes_location/blob/main/ARTIFACT_INDEX.md) | 材料索引 |

复现论文实验时，应使用材料仓库记录的版本、配置和检查点。受限真实事件、台站和震相输入不公开分发；相关实验需要自行取得授权数据，具体见[真实数据政策](https://huggingface.co/cangyeone/robust_bayes_location/blob/main/data/REAL_DATA_POLICY.md)。

## 安装与快速示例

需要 Python 3.10 或更高版本。核心依赖为 PyTorch、NumPy、SciPy 和 pyproj；使用 FMM 时还需要 scikit-fmm。

```bash
git clone https://github.com/cangyeone/bayes_location.git
cd bayes_location
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[fmm,test]"
python -m bayesloc --help
```

Windows 下使用 `.venv\Scripts\activate` 激活环境。不需要 FMM 和测试依赖时，可以使用 `python -m pip install -e .`。训练、评估和定位支持 `--device auto|cpu|cuda|mps`；自动模式依次选择 CUDA、Apple MPS 和 CPU。

下面的示例生成虚构台站和事件，不需要真实研究数据：

```bash
python examples/make_demo.py --output work/demo

python -m bayesloc prepare \
  --stations work/demo/stations.csv \
  --events work/demo/training_events.csv \
  --picks work/demo/training_picks.csv \
  --geometry work/demo/geometry.json \
  --output work/demo/observed_pairs.npz

python -m bayesloc train \
  --pairs work/demo/observed_pairs.npz \
  --checkpoint work/demo/time.pt \
  --epochs 3 --hidden-dim 32 --batch-size 128 --device cpu

python -m bayesloc locate \
  --stations work/demo/stations.csv \
  --events work/demo/events.csv \
  --picks work/demo/picks.csv \
  --checkpoint work/demo/time.pt \
  --output work/demo/location \
  --samples 80 --burn 40 --thin 2 --chains 2 \
  --pool-size 6 --refine-top 2 --init-steps 5 --device cpu

python -m bayesloc filter \
  --catalog work/demo/location/catalog.csv \
  --top-k 2 --output work/demo/selected.csv

python -m bayesloc reproject \
  --catalog work/demo/selected.csv \
  --geometry work/demo/geometry.json \
  --output work/demo/selected.geographic.csv

python -m pytest -q
```

**3 轮训练和 80 步采样仅用于检查流程能否运行，不能证明模型精度或后验收敛，也不构成论文复现。** 示例 CSV 使用均匀介质解析走时；另行生成的 `velocity.npz` 是用于 FMM 演示的虚构非均匀速度场。正式应用需要独立验证走时精度，并检查初始化和采样诊断。每次定位应选择新的输出目录。

## 目录与命令

```text
bayesloc/       Python 包与统一命令行接口
examples/       虚构数据生成器与示例配置
tests/          数值和接口测试
docs/           算法、迁移及验证说明
README.md       英文主说明与完整字段定义
README_zh.md    中文说明
CITATION.cff    引用元数据
pyproject.toml  安装配置与依赖
DATA_POLICY.md  数据使用与发布规则
```

| 任务 | 命令 |
|---|---|
| 有参考震源的观测数据转为训练样本 | `python -m bayesloc prepare` |
| 将 SWChinaCVM 速度剖面转换为投影网格 | `python -m bayesloc build-grid` |
| 从三维速度网格生成 FMM 训练标签 | `python -m bayesloc generate-fmm` |
| 训练 P/S 走时网络 | `python -m bayesloc train` |
| 独立走时评估 | `python -m bayesloc evaluate` |
| 由分组到时进行定位 | `python -m bayesloc locate` |
| 对目录排序和筛选 | `python -m bayesloc filter` |
| 将投影坐标转换为经纬度 | `python -m bayesloc reproject` |

各命令均可添加 `--help` 查看参数。

## 坐标、单位和数据约定

| 数据 | 约定 |
|---|---|
| 经纬度 | WGS84，经度在前、纬度在后 |
| 网络输入 | `[receiver_xyz, source_xyz]`，单位 km |
| 网络输出 | `[Tp, Ts]`，单位 s |
| 水平坐标 | AEQD 投影后的局部 x/y，单位 km |
| 深度 | 向下为正，单位 km，使用共同的垂直基准 |
| 台站高程 | 向上为正，输入单位 m；台站 z 为 `-elevation_m/1000` |
| 速度与网格 | 速度 km/s，网格轴 km，速度数组形状 `(Nz,Ny,Nx)` |
| 绝对时间 | 带 `Z` 或明确时区偏移的 ISO 8601 时间，内部转换到 UTC |

投影原点、源和接收器适用范围、单位、网络缩放和高程约定均属于模型配置。仅修改范围配置并不能扩大模型的训练覆盖。网络内部对已经以 km 表示的坐标再除以 1000 是数值缩放，不是单位转换。详见[完整数据约定](README.md#coordinates)。

## 两种走时训练途径

### 使用观测走时

需要台站位置、可靠的参考震源坐标与发震时刻，以及对应的 P/S 到时。训练标签为“绝对到时减参考发震时刻”。这种监督训练目标不需要显式速度模型，但仅有尚未定位的到时不能提供训练所需的震源位置和发震时刻。

输入包括台站表、训练事件表、震相表和投影/适用范围配置。缺失震相应省略对应行，不能填入零到时；重复事件—台站—震相键需要上游明确处理。当前接口使用 P/S 两个目标，不自动合并 Pg/Pn/Sg/Sn。字段和命令见[观测训练流程](README.md#observed-training)。

### 使用 FMM 标签

将三维 Vp/Vs 结构转换为共同投影下的规则网格，通过 FMM 求取不同震源到接收器的 P/S 初至走时，再用相同的监督网络拟合这些标签。FMM 在离线标签生成阶段使用速度信息；网络训练本身不施加 eikonal 方程损失。

网格轴必须严格递增、等间隔；不同方向的间隔可以不同。速度需有限、为正且满足 `vp > vs`。经纬度角度不能直接作为笛卡尔距离。详见 [FMM 流程](README.md#fmm-training)和 [SWChinaCVM-V2.0 示例](docs/SWCHINA_CVM.md)。

两种途径都需要独立验证。默认训练/验证划分按不同震源坐标分组，防止同一震源对应的不同接收器跨集合。模型检查点包含网络结构和适用范围；跨地区或改变高程、震相定义时，不能只凭参数形状相同就直接复用。见[训练、评估与模型复用](README.md#model)。

## 定位、输出与筛选

定位需要台站、按事件或候选时间窗分组的到时，以及每个事件的时间参考。定位事件表不要求已知震源坐标；`reference_time` 可以是近似发震时刻或邻近的窗口起点，算法进一步估计发震时刻偏移。当前接口不会自动完成连续到时流的关联和重叠窗口去重。

`locate` 支持 `gaussian`、`student_t` 和默认的 `student_t_z` 三种残差模式。多起点初始化根据台站和到时构造候选，再用当前走时代理评分与精炼。多个采样链从选中的最佳候选起步，这并不保证探索到远处的其他后验模态。具体配置见[定位说明](README.md#location)。

| 输出 | 内容 |
|---|---|
| `catalog.csv` | 位置、发震时刻、不确定性宽度和链诊断 |
| `chain_00.npz` 等 | 保留的后验样本、震相内点概率和索引映射 |
| `initialization.npz` | 初始候选、时间偏移和评分 |
| `pick_quality.csv` | 实际观测震相的到时和后验内点概率 |
| `rejected_events.csv` | 未达到观测数量要求的事件及原因 |
| `run.json` | 配置、输入摘要、随机种子、环境与计时记录 |

事件筛选使用中央 90% 边际区间的**全宽** `q95-q05`。水平宽度为 `sqrt(Wx²+Wy²)`，并与深度宽度分别转换成目录内百分位排名，最终分数为两者的最大值，越小越优。它是相对排序分数，不是事件真实性概率，也不是可跨地区通用的误差界限。

可以通过 `filter --top-k` 保留固定数量的事件，或增加宽度和 R-hat 条件。保留数量及阈值应结合自己的数据和独立验证选择；论文或示例中的数值不是通用标准。筛选同时生成全量审计表和配置记录。见[震相诊断与事件筛选](README.md#qc)。

`locate` 已输出地理坐标；需要再次转换投影坐标时可使用 `reproject`，并保持与训练和定位一致的投影配置及垂直基准。见[逆投影说明](README.md#projection)。

## 适用边界与数据使用

走时模型是监督回归模型。后验依赖所指定的走时模型、误差模型与先验，不会自动包含完整的速度结构不确定性。较窄区间仍可能来自错误局部解、未充分混合的采样链或前向模型偏差；排序有效也不等于区间覆盖率已经校准。

本仓库维护的通用流程不提供波形拾取、完整自动关联、震级估计、双差重定位、各向异性速度反演或四分支走时模型。遇到输入格式、适用范围或采样问题时，参见[排错与限制](README.md#limitations)。旧模型和旧格式应按[迁移指南](docs/MIGRATION.md)转换。

公开示例与测试使用虚构数据。授权真实输入保存在 Git 忽略的 `private_data/` 中，模型、定位和其他生成结果保存在 `work/` 中；受限真实事件、台站和相关派生数据不能作为公开示例提交。具体规则见 [DATA_POLICY.md](DATA_POLICY.md)。

## 开发者与许可

- **Yuqi Cai**：[caiyuqiming@foxmail.com](mailto:caiyuqiming@foxmail.com)
- **Xin Liu**：[xinliu_geo@outlook.com](mailto:xinliu_geo@outlook.com)
- **Ziye Yu**：[yuziye@cea-igp.ac.cn](mailto:yuziye@cea-igp.ac.cn)
- **Fan Xie**：[xiefan@outlook.com](mailto:xiefan@outlook.com)

开发或借助 Coding agent 修改代码前，请阅读 [AGENTS.md](AGENTS.md)。修改数值流程后应运行相应测试，并如实区分接口检查、数值验证和完整论文复现。

本仓库保留已有的研究用途许可声明。其他使用、再分发或商业用途请联系开发者。Copyright © 2025–2026, Ziye Yu. All rights reserved. Hugging Face 材料的许可和数据使用条件以该材料仓库的说明为准。
