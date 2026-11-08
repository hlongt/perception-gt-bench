# PerceptionGT-Bench

面向**单目/双目深度与 3D 跟踪**的伪真值生产、质量校验与 benchmark 评测系统。

自带确定性合成双目渲染器与精确深度真值，不依赖任何需要注册下载的数据集；核心立体匹配、多视几何三角化、指标计算全部手写实现；C++17 内核可编译并实测加速比，未编译时自动回退 NumPy。

---

## 为什么做这个

深度估计与 3D 跟踪模型的迭代，依赖大量带深度真值的训练与评测数据。真实 LiDAR 真值成本高、覆盖稀疏、难以规模化，因此工程上普遍用**多源伪真值（pseudo-GT）+ 质量校验 + benchmark 闭环**替代。难点不在「多跑几个模型」，而在三个系统问题：伪标签怎么生产与互补、伪标签自己的置信度怎么标定、评测结果怎么变成可行动的迭代信号。

本仓库把这条数据流完整实现，并且可以一键复现验证：

| 系统能力 | 本仓库实现 |
|---|---|
| 真值系统方案设计与数据生产 | `data_ingest` + `pseudo_gt/` 三路伪真值 + `gt_qc` 质量校验；sha256 manifest 去重与断点续跑 |
| SfM 生成伪真值 | `pseudo_gt/sfm.py`：Harris 角点 → 对极扫描线匹配 → 本质矩阵校验 → 批量 DLT 三角化 → 重投影误差置信度 |
| 双目深度 | `pseudo_gt/stereo.py`：SAD 代价体 → WTA → 亚像素抛物线拟合 → 左右一致性 → 峰值比置信度 |
| 离线单目先验（可插拔基础模型） | `pseudo_gt/foundation.py`：`FoundationBackend` 接口 + 内置离线先验实现 + Depth Anything V2 / SAM 集成占位（见「已知限制与适用范围」） |
| 测试数据挖掘 | `benchmark/slices.py` 六类场景切片 + `benchmark/regression.py` 按图块排序的 hard-case 挖掘 |
| 评价指标 | 深度 AbsRel / RMSE / SIlog / δ<1.25；3D 跟踪 MOTA / MOTP / IDF1 / IDSW / DetA / AssA / 单阈值 HOTA |
| benchmark 系统 | 6 种方案自动消融对比、场景切片难度排序、置信度校准、Markdown + HTML 报告、可视化看板 |
| C++ 内核 | `cpp/` C++17 实现（LRC 滤波、置信度、中值后处理、指标计算）+ CMake + ctypes 绑定 + 加速比实测 |

---

## 快速开始

```bash
# 1. 依赖（仅 4 个轻量包，不需要 torch / opencv / CUDA）
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt    # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS

# 2. 可选：编译 C++17 内核（需 g++ / clang++ / MSVC，或 pip install ziglang）
python cpp/build_ctypes.py

# 3. 一键跑通全链路
python scripts/run_all.py

# 4. 单独测试 / 单独测 C++ 加速比
python -m pytest -q
python scripts/bench_native.py
```

产物全部落在 `outputs/`：

| 文件 | 内容 |
|---|---|
| `report.md` / `report.html` | 完整评测报告（方案对比、切片难度、跟踪指标、hard-case、置信度校准、C++ 加速比、已知限制） |
| `results.json` | 全部逐帧指标，便于二次分析 |
| `depth_panel.png` | RGB / 深度真值 / 融合伪真值 / 误差热图四联图 |
| `slice_bars.png` | 各场景切片难度 |
| `variant_bars.png` | 6 种伪真值方案消融对比 |
| `risk_coverage.png` | 覆盖率-精度权衡曲线（置信度阈值扫描） |
| `manifest.json` | 逐帧 sha256，重复运行不会重复处理 |

---

## 项目结构

```
perception-gt-bench/
├── configs/default.json              全部超参（相机、切片、匹配、融合、评测）
├── src/gtbench/
│   ├── data/
│   │   ├── calibration.py            针孔双目模型：投影/反投影/深度↔视差/P1 P2
│   │   ├── synth.py                 向量化 ray-box 渲染器，输出左右视图 + 精确深度 GT + 3D 框
│   │   └── ingest.py                sha256 manifest 去重、断点续跑、KITTIAdapter 接口
│   ├── geometry.py                  本质矩阵、对极残差、DLT 三角化、重投影误差、Harris、积分图
│   ├── pseudo_gt/
│   │   ├── stereo.py                SAD 双目匹配 + 亚像素 + LRC
│   │   ├── sfm.py                   稀疏重建 + 置信度扩散
│   │   ├── foundation.py            可插拔基础模型 backend
│   │   └── fusion.py                MAD 离群点门控 + 置信度加权融合
│   ├── gt_qc.py                     有效覆盖率、误差分布、置信度校准分箱
│   ├── benchmark/
│   │   ├── metrics.py               深度与 3D 跟踪指标
│   │   ├── tracker.py               3D IoU 关联跟踪器
│   │   ├── slices.py                六类场景切片定义
│   │   ├── regression.py            6 方案消融 + hard-case 挖掘 + 回归报告数据
│   │   ├── report.py                Markdown / HTML 报告生成器
│   │   └── viz.py                   可视化
│   ├── native.py                    ctypes 封装 + 纯 NumPy 回退
│   ├── bench_native.py              C++ vs NumPy 加速比实测
│   └── cli.py                       一键入口
├── cpp/                             C++17 内核、CMakeLists、编译脚本
├── scripts/                         run_all.py / bench_native.py
├── tests/                           19 个单元测试（几何、指标、融合、立体、全链路）
└── docs/                            方案设计 / 指标与误差来源 / 复现指南
```

---

## 技术要点

**伪真值三路，互补而非重复**
- 双目支路给出稠密但易在弱纹理/遮挡处出错的度量深度；
- SfM 支路给出稀疏但几何自洽、带重投影误差约束的深度；
- 离线先验支路给出全图覆盖但只有相对尺度、需最小二乘对齐到 SfM 的深度。

三者的误差模式不同，所以融合才有意义。

**融合不是平均**：先按像素取中位数，用 `k × MAD` 判定离群源并降权（不是直接丢弃），再做置信度加权；输出置信度同时考虑权重总和与源间一致性。`pseudo_gt/fusion.py`。

**benchmark 是「能加快迭代」的**，不只是一张指标表：六类场景切片定位失效模式、6 种方案自动消融、按 16×16 图块挖掘 worst-case 并反哺数据采集、置信度分箱校准检查置信度是否可信。

**C++ 内核**：`cpp/src/*.cpp` 用 C ABI 导出，Python 侧 `ctypes` 调用；编译脚本自动探测 `g++` / `clang++` / `cl` / `zig c++`。没有编译器时 `native.py` 静默回退 NumPy，全链路依旧可跑。

---

## 实测结果

运行 `python scripts/run_all.py` 后查看 `outputs/report.md`。报告里的每一个数字都由本地真实运行产生，`results.json` 可逐条追溯；本 README 不预置任何未经实测的指标。

关键输出：
- 6 种伪真值方案的 AbsRel / RMSE / δ<1.25 对比，以及校准融合相对未校准等权融合的改善幅度；
- 六类场景切片的难度排序（哪类场景最容易让算法失效）；
- 3D 跟踪的 MOTA / IDF1 / IDSW / 单阈值 HOTA；
- C++ 内核相对 NumPy 的加速比。

---

## 已知限制与适用范围

1. **评测数据是程序化合成场景，不是 KITTI 真实数据**。`data/ingest.py::KITTIAdapter` 给出了切换到真实数据的完整字段映射说明（内参、velodyne 投影、label_2 3D 框），下游模块无需改动，但本次未接入真实数据。
2. 基础模型支路默认跑的是**手写的离线单目先验**（垂直位置 + 亮度 + 边缘保持平滑），**不是 Depth Anything V2 / SAM**。这两个 backend 是显式集成占位，未下载权重、未参与本报告任何数值。
3. SfM 支路是**双目角点三角化 + 本质矩阵校验 + 重投影误差**，不是完整 COLMAP 增量式 SfM / 3DGS 重建 pipeline。
4. C++ 内核覆盖立体后处理与指标计算，**未覆盖 SAD 代价体本身**。
5. HOTA 为**单一 IoU 阈值**实现，未做官方的多阈值积分，数值口径与论文不完全等同。

这些边界在 `outputs/report.md` 第 9 节会原样重复一遍。

---

## 复现与测试

```bash
python -m pytest -q        # 19 passed
python scripts/run_all.py --seed 42 --out outputs_seed42
python scripts/run_all.py --no-viz
```

数据集完全由 `configs/default.json` 的 `seed` 决定，同 seed 逐字节一致（有单元测试保证）。

详见 `docs/复现指南.md`。
