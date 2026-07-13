# ---- Data source ----
import json
from pathlib import Path
import os

_DEFAULT_TARGET_POINT = "测点16"
_DATA_ROOT = Path('/Users/xiechushu/project/EM_app/TEM_app/data/sy6')
TARGET_POINT = os.environ.get("TARGET_POINT", _DEFAULT_TARGET_POINT)  # 当前处理的测点名称，用于日志及结果文件元数据
SOURCE_DATA_PATH = os.environ.get(
    "SOURCE_DATA_PATH",
    str(_DATA_ROOT / TARGET_POINT),
)  # 测点原始及降噪数据所在目录
CURRENT_SCALE = 500.0  # 数据加载时使用的电流缩放系数
DATA_FILE_STEM = "测点11_010"  # 未使用降噪数据时读取的原始数据文件名前缀
USE_DENOISED = True  # True=使用降噪信号，False=直接读取原始 npy 信号

_point_params = json.loads(Path(__file__).resolve().with_name("point_params.json").read_text())
if TARGET_POINT not in _point_params:
    raise KeyError(f"未在 point_params.json 中找到测点配置: {TARGET_POINT}")
if not Path(SOURCE_DATA_PATH).exists():
    raise FileNotFoundError(f"测点数据目录不存在: {SOURCE_DATA_PATH}")

_p = _point_params[TARGET_POINT]
TIME_GATE_START = _p["TIME_GATE_START"]

Vobs_SCALE_FACTOR = _p["Vobs_SCALE_FACTOR"]

TIME_GATE_END = 20e-3  # 对数抽道终止时刻，相对脉冲起点，单位 s
TIME_GATE_COUNT = 64  # 起止时刻之间的对数等间隔抽道数量

# ---- Jacobian ----
JACOBIAN_STEP = 4.89e-03  # 最优绝对扰动 (sweep, spike layer 5, rel_err ~12%)

# ---- Inversion ----
N_LAYERS = 50  # 反演模型总层数，包含最底部半空间
LAYER_THICKNESS = 0.5  # 除最底层外各层的固定厚度，单位 m
MAX_ITER = 50  # DLS 反演允许的最大迭代次数
LAMBDA_INITIAL = 1  # DLS 正则化/阻尼参数的初始值


LAMBDA_DECREASE = 0.8  # 更新被接受后阻尼参数的乘法缩小系数






# ---- Output ----
from pathlib import Path
RESULTS_DIR = Path(__file__).resolve().parent / "results" / TARGET_POINT
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONSTRAINT_TYPE = "DLS"  # 当前反演约束类型，使用大写字符串标识 DLS 约束
# ---- Coil parameters ----
TX_RADIUS = 0.5       # 发射线圈半径 (m)
TX_TURNS = 8          # 发射线圈匝数
TX_HEIGHT = 0.0       # 发射线圈离地高度 (m)
RX_RADIUS = 0.2       # 接收线圈半径 (m)
RX_TURNS = 90         # 接收线圈匝数
RX_HEIGHT = 0.1       # 接收线圈离地高度 (m)
OFFSET = 0.58         # 收发水平距离 (m)
