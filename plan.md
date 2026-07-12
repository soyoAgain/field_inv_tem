# 多测点反演实现计划

## 现状

当前 `point11/` 仅对测点11做反演，流程为：

```
save_waveform.py → save_gated_data.py → inv_dls.py → plot_inv.py
```

16 个测点各有一个 SSA 降噪记录。

## 方案：逐点独立反演

每个测点完全独立——有自己的波形、抽道数据、反演参数——只需要**共享同一套算法代码**。

### 目录结构

```
field_inv_tem/
├── point11/                    # 现有单点目录（保留不动）
├── multi_point/                # 新增：多测点反演
│   ├── config.py               # 全局配置（线圈参数、反演超参）
│   ├── run_all.sh              # 遍历所有测点
│   ├── run_point.py            # 单点反演脚本（命令行参数指定测点）
│   ├── plot_all.py             # 汇总所有测点的 RMS 曲线和最终模型
│   ├── results/                # 每个测点的输出子目录
│   │   ├── 测点1/
│   │   │   ├── data_conf.json
│   │   │   ├── inversion_result.json
│   │   │   └── fig/
│   │   ├── 测点2/
│   │   └── ...
│   └── summary/                # 汇总图
│       ├── rms_all.png         # 16 条 RMS 曲线
│       └── section.png         # 电阻率断面图
```

### 核心思路

**1. `run_point.py` — 单点全流程脚本**

接受测点名称作为命令行参数，复用现有算法模块：

```
python run_point.py --point 测点1
```

流程：
1. 加载测点数据（`data_loader.discover_sy6_records`）
2. 提取波形（`save_waveform` 的检测算法）
3. 对数抽道（`save_gated_data` 的插值逻辑）
4. 反演（`inv_dls` 的 DLS + CG 算法）
5. 绘图（`plot_inv` 的并行渲染）
6. 所有输出写入 `results/测点X/`

**2. `run_all.sh` — 串行遍历**

```bash
for point in 测点1 测点2 ... 测点16; do
    python run_point.py --point "$point"
done
```

串行执行避免 Fortran 二进制竞态（共享 JSON 文件）。

**3. 并行替代方案**

```bash
# 每个 worker 使用独立的工作目录，隔离 Fortran JSON 文件
python run_point.py --point 测点1 --workdir /tmp/tem_1 &
python run_point.py --point 测点2 --workdir /tmp/tem_2 &
...
```

每个 worker 在自己的 `/tmp` 子目录中运行 Fortran 二进制，避免文件覆盖。

### 需要修改的模块

| 模块 | 修改内容 |
|------|---------|
| `save_waveform.py` | 接受测点名参数，不再硬编码 `测点11` |
| `save_gated_data.py` | 同上 |
| `inv_dls.py` | 同上 |
| `tem_wrapper.py` | 接受可配置的 `workdir`，Fortran JSON 写入独立路径 |
| `config.py` | 拆分为全局参数 + 测点特定参数（`SOURCE_DATA_PATH` 变为运行时参数） |

### 汇总可视化 (`plot_all.py`)

- **RMS 收敛曲线**：16 条，区分颜色
- **电阻率断面图**：pcolormesh（测点为 x 轴，深度为 y 轴，颜色为 lg(ρ)）
- **数据拟合曲线**：选取典型测点，绘制观测 vs 正演

### 测点差异参数处理

每个测点波形不同，`TIME_GATE_START` 和 `Vobs_SCALE_FACTOR` **必须手动设置**，不存在自动值。

#### point_params.json

每个测点一个条目，包含必需的两个字段：

```json
{
  "测点1":  { "TIME_GATE_START": 0.204,  "Vobs_SCALE_FACTOR": 2.5e8 },
  "测点2":  { "TIME_GATE_START": 0.203,  "Vobs_SCALE_FACTOR": 1.8e8 },
  ...
  "测点16": { "TIME_GATE_START": 0.205,  "Vobs_SCALE_FACTOR": 3.1e8 }
}
```

`run_point.py` 启动时：

```python
params = json.load("point_params.json")
if point not in params:
    raise ValueError(f"点 {point} 未在 point_params.json 中配置")
cfg = params[point]
TIME_GATE_START = cfg["TIME_GATE_START"]
Vobs_SCALE_FACTOR = cfg["Vobs_SCALE_FACTOR"]
```

- `TIME_GATE_START`：抽道起始时刻，相对采集起始的绝对时间 (s)
- `Vobs_SCALE_FACTOR`：数据标定因子，`d_work = d_obs / Vobs_SCALE_FACTOR`，用于消除仪器增益差异

所有 16 个测点**必须在 point_params.json 中提前配置**，未配置的测点直接报错退出。

### 工作量估计

| 任务 | 时间 |
|------|------|
| 模块参数化（接受测点名） | 20 min |
| `point_params.json` + 自动检测逻辑 | 15 min |
| `run_point.py` 单点脚本 | 15 min |
| `run_all.sh` 调度脚本 | 5 min |
| `plot_all.py` 汇总可视化 | 20 min |
| 测试 2 个测点 | 10 min |
| 全 16 测点运行 | ~30 min（串行，每点 ~2 min） |
