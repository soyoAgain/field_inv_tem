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

每个测点的波形不同 → 关断时刻不同 → `TIME_GATE_START` 和 `Vobs_SCALE_FACTOR` 不一致。采用**自动检测 + 手动覆盖**策略。

#### 1. TIME_GATE_START（抽道起始时间）

**自动**：波形检测输出 `real_axis[3]`（关断结束点，电流首次 < 0）。从 `data_conf.json` 读取：

```
TIME_GATE_START = real_axis[3] + 0.5e-3s   // 关断后 0.5ms 开始抽道
```

此值对每个测点自动计算，无需硬编码。`save_gated_data.py` 已经读取 `wave_start_time_real_axis`，改一行即可。

**手动**：如果自动检测的关断点不准，可以在 `point_params.json` 中覆盖：

```json
{
  "测点3": { "TIME_GATE_START": 0.204 },
  "测点7": { "TIME_GATE_START": 0.205 }
}
```

#### 2. Vobs_SCALE_FACTOR（数据标定因子）

**自动**：反演前用初始均匀半空间计算：

```
cal = median(d_obs / abs(f_forward(50 Ω·m)))
```

`inv_dls.py` 已内置此逻辑（`CALIBRATION_FLAG`），每个测点独立计算。

**手动**：如果系统增益已知或需要跨越测点对比，在 `point_params.json` 中覆盖：

```json
{
  "测点3": { "Vobs_SCALE_FACTOR": 2.5e8 },
  "测点7": { "Vobs_SCALE_FACTOR": 1.8e8 }
}
```

`inv_dls.py` 增加 `--scale` 命令行参数，有则覆盖自动计算的 cal。

#### 3. 测点特定配置文件 `point_params.json`

只放**需要手动覆盖**的测点，大部分测点靠自动检测：

```json
{
  "测点1": { "TIME_GATE_START": 0.2042 },
  "测点5": { "Vobs_SCALE_FACTOR": 3.1e8 },
  "测点11": { "TIME_GATE_START": 0.2032, "Vobs_SCALE_FACTOR": 2.0e8 }
}
```

`run_point.py` 加载时：

```python
def get_param(point, key, default):
    overrides = json.load("point_params.json")
    if point in overrides and key in overrides[point]:
        return overrides[point][key]
    return default
```

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
