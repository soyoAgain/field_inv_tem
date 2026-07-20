# ---- Data source ----
import json
from pathlib import Path
import os

# 当未通过环境变量指定测点时，默认处理测点11
_DEFAULT_TARGET_POINT = "测点11"

# 所有测点原始/降噪数据的根目录（每个子目录对应一个测点，如 "测点11"）
_DATA_ROOT = Path('/Users/xiechushu/project/EM_app/TEM_app/data/sy6')

# 当前处理的测点名称，可用环境变量 TARGET_POINT 覆盖
# 该值同时影响日志输出、结果目录命名和默认数据路径
TARGET_POINT = os.environ.get("TARGET_POINT", _DEFAULT_TARGET_POINT)

# 当前测点的原始及降噪数据所在目录（优先读取环境变量 SOURCE_DATA_PATH）
# 若未设置，自动从 _DATA_ROOT 和 TARGET_POINT 拼接
# expanduser + resolve 确保路径无 ~ 和 .. 等简写
SOURCE_DATA_PATH = Path(
    os.environ.get(
        "SOURCE_DATA_PATH",
        str(_DATA_ROOT / TARGET_POINT),
    )
).expanduser().resolve()

# 加载电流数据时应用的缩放系数（原始数据单位 1/500 A，乘以此值后单位为 A）
CURRENT_SCALE = 500.0

# 指定要加载的炮号（文件名前缀），例如 "测点11_013"
# 若不设置，默认选择当前测点目录下唯一带完整降噪数据的炮次
DATA_FILE_STEM = os.environ.get("DATA_FILE_STEM")

# 是否使用降噪后的完整信号进行抽道
# True: 使用 denoised_full.npy；False: 使用原始 rx.npy
USE_DENOISED = True

# 加载各测点的抽道与缩放参数（抽道起始时间、Vobs 缩放系数等）
# point_params.json 中必须包含 TARGET_POINT 对应的条目
_point_params = json.loads(Path(__file__).resolve().with_name("point_params.json").read_text())
if TARGET_POINT not in _point_params:
    raise KeyError(f"未在 point_params.json 中找到测点配置: {TARGET_POINT}")
if not SOURCE_DATA_PATH.is_dir():
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
LAYER_THICKNESS = 0.2  # 除最底层外各层的固定厚度，单位 m
MAX_ITER = 50  # DLS 反演允许的最大迭代次数
LAMBDA_INITIAL = 100  # DLS 正则化/阻尼参数的初始值
CONSTRAINT_TYPE = "OCCAM"  # OCCAM、DLS、MGS

LAMBDA_DECREASE = 0.8 # 更新被接受后阻尼参数的乘法缩小系数
INITIAL_RHO = 1e-2     # 初始模型的均匀电阻率 (Ω·m)
MGS_beta = 0.5*1e-1
LAMBDA_INITIAL_MGS = 1
LAMBDA_DECREASE_MGS = 0.99




# ---- Output ----
from pathlib import Path
RESULTS_DIR = Path(__file__).resolve().parent / "results" / TARGET_POINT
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Coil parameters ----
TX_RADIUS = 0.5       # 发射线圈半径 (m)
TX_TURNS = 8          # 发射线圈匝数
TX_HEIGHT = 0.0       # 发射线圈离地高度 (m)
RX_RADIUS = 0.2       # 接收线圈半径 (m)
RX_TURNS = 90         # 接收线圈匝数
RX_HEIGHT = 0.1       # 接收线圈离地高度 (m)
OFFSET = 0.58         # 收发水平距离 (m)
