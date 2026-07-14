# TEM 1D Field Data Inversion

**瞬变电磁（TEM）现场数据一维阻尼最小二乘反演**

基于 Fortran 正演引擎 + Python+numba 正演引擎，采用对数空间中心差分雅可比矩阵，支持 DLS/CG 迭代求解和多测点批量反演。

---

## 目录结构

```
field_inv_tem/
├── config.py                 # 全局配置（线圈参数、反演超参、层数）
├── point_params.json         # 测点特定参数（采样窗口、标定因子）
├── run.sh                    # 单测点一键运行
├── multi_point.sh            # 全 16 测点批量反演 + 汇总可视化
│
├── save_waveform.py          # 波形 4 点关键点自动检测
├── save_gated_data.py        # 对数抽道 + SSA 降噪数据处理
├── inv_dls.py                # DLS 反演（CG 求解器 + alpha 冷却）
├── plot_inv.py               # 单点迭代可视化（joblib 并行渲染）
├── plot_multi_point.py       # 多测点汇总可视化
│
├── tem_wrapper.py            # 正演封装（Fortran 二进制 + Python+numba 双引擎）
├── tem_forward_1d.py         # 纯 Python + numba 独立正演引擎
├── jacob_cal.py              # 雅可比矩阵计算（joblib 并行）
│
├── _dev/                     # 开发调试脚本
│   ├── compare_forward.py    # Fortran vs Python 正演对比
│   ├── sweep_step.py         # Jacobian 步长扫描
│   └── test_jacob_cal.py     # Jacobian Taylor 验证
│
├── results/                  # 反演输出（git 忽略）
│   └── 测点11/
│       ├── data_conf.json
│       ├── inversion_result.json
│       └── fig_inv/
│
└── README.md
```

---

## 快速开始

### 单测点反演

```bash
./run.sh            # 默认测点11
./run.sh 测点5      # 指定测点
```

流程：波形检测 → 对数抽道 → DLS 反演 → 可视化

### 全部 16 测点反演

```bash
./multi_point.sh
```

每个测点串行执行 `run.sh`，最后生成汇总断面图。

---

## 正演引擎

| 引擎 | 函数 | 速度（64 点） | 说明 |
|------|------|-------------|------|
| Fortran 二进制 | `tem_wrapper.tem_forward()` | 17ms | 调用外部 Fortran 可执行文件 |
| Python+numba | `tem_wrapper.tem_forward_numba()` | **15ms** | 纯 Python，numba JIT 加速 |

两个引擎输出差异 < 10⁻⁵（相关系数 ≈ 1.0）。

### 接口

```python
from tem_wrapper import tem_forward, tem_forward_numba

resp = tem_forward(rho=np.array([50.0]), thickness=np.array([]))
resp = tem_forward_numba(rho=np.array([50.0]), thickness=np.array([]))
```

---

## 雅可比矩阵

```python
from jacob_cal import jacobian, jacobian_numba

J = jacobian(rho, thickness)             # Fortran 引擎
J = jacobian_numba(rho, thickness)       # Python+numba，joblib 并行
```

- 对数空间中心差分：`J[:,j] = (log10(f+) - log10(f-)) / (2·dm)`
- 并行加速：joblib `loky` 后端，20 层 4.0× 提速，50 层 4.6× 提速
- Taylor 验证：`JACOBIAN_STEP=9e-3` 时相对误差 < 0.5%

---

## 反演算法

### 数学模型

$$\Phi(\mathbf{m}) = \big\|\log_{10}\mathbf{d} - \log_{10}\mathbf{f}(\mathbf{m})\big\|^2 + \alpha \cdot \big\|\mathbf{W}_m \mathbf{m}\big\|^2$$

- $\mathbf{m}$：$\log_{10}(\rho)$ 向量（$n$ 层）
- $\mathbf{f}(\mathbf{m})$：正演响应
- $\mathbf{W}_m$：一阶差分粗糙度矩阵（$\mathrm{W}_m[0,0]=0$）
- $\alpha$：阻尼因子（初始值 1.0，每次接受后 ×0.8）

### Gauss-Newton 更新

$$\big(\mathbf{J}^T\mathbf{J} + \alpha \mathbf{W}_e^T\mathbf{W}_e\big) \Delta\mathbf{m} = \mathbf{J}^T \big(\log_{10}\mathbf{d} - \log_{10}\mathbf{f}\big) - \alpha \mathbf{W}_m^T\mathbf{W}_m \mathbf{m}$$

- 求解器：`scipy.sparse.linalg.cg`（共轭梯度，tol=1e-10, maxiter=10000）
- 步长固定：$\mathbf{m}_{k+1} = \mathbf{m}_k + 0.5 \cdot \Delta\mathbf{m}$

### 收敛判断

- 最大迭代次数：`MAX_ITER`（默认 50）
- NaN 保护：`m_iter` 出现 NaN 时终止
- 正向响应非正保护：`Fm <= 0` 时终止
- 模型裁剪：`m_iter ∈ [-10, 10]`，`rho_iter > 1e-30`

---

## 配置说明

### 主配置文件 `config.py`

```python
N_LAYERS = 20          # 层数
LAYER_THICKNESS = 0.5  # 每层厚度 (m)
MAX_ITER = 50          # 最大迭代次数
LAMBDA_INITIAL = 1     # 初始阻尼因子
LAMBDA_DECREASE = 0.8  # 阻尼衰减系数
INITIAL_RHO = 1e-2     # 初始均匀电阻率 (Ω·m)
JACOBIAN_STEP = 9e-3   # 雅可比扰动步长
TARGET_POINT = "测点11" # 当前测点
```

### 测点参数 `point_params.json`

```json
{
  "测点11": {
    "TIME_GATE_START": 3.2e-3,
    "TIME_GATE_END": 20e-3,
    "Vobs_SCALE_FACTOR": 2.04e8
  }
}
```

---

## 测试 API

```bash
# 正演对比
python _dev/compare_forward.py

# Jacobian 步长扫描
python _dev/sweep_step.py

# Jacobian Taylor 验证
python _dev/test_jacob_cal.py
```
