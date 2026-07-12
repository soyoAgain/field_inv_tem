# ---- Data source ----
SOURCE_DATA_PATH = '/Users/xiechushu/project/EM_app/TEM_app/data/sy6/测点11'  # 测点原始及降噪数据所在目录
CURRENT_SCALE = 500.0  # 数据加载时使用的电流缩放系数
DATA_FILE_STEM = "测点11_010"  # 未使用降噪数据时读取的原始数据文件名前缀
USE_DENOISED = True  # True=使用降噪信号，False=直接读取原始 npy 信号

# ---- Time gating (log-spaced, relative to wave_start_time origin) ----
TIME_GATE_START = 3.2e-3  # 对数抽道起始时刻，相对脉冲起点，单位 s
TIME_GATE_END = 20e-3  # 对数抽道终止时刻，相对脉冲起点，单位 s
TIME_GATE_COUNT = 64  # 起止时刻之间的对数等间隔抽道数量
Vobs_SCALE_FACTOR = 5*1e-2 # 对感应电压的缩放系数
# ---- Jacobian ----
JACOBIAN_STEP = 4.89e-03  # 最优绝对扰动 (sweep, spike layer 5, rel_err ~12%)

# ---- Inversion ----
N_LAYERS = 20  # 反演模型总层数，包含最底部半空间
LAYER_THICKNESS = 0.5  # 除最底层外各层的固定厚度，单位 m
MAX_ITER = 50  # DLS 反演允许的最大迭代次数
LAMBDA_INITIAL = 1  # DLS 正则化/阻尼参数的初始值


LAMBDA_DECREASE = 0.8  # 更新被接受后阻尼参数的乘法缩小系数



# LAMBDA_INCREASE = 3.0  # 更新被拒绝后阻尼参数的乘法放大系数
# LOG10_RHO_MIN = -4.0  # 电阻率下界的 log10 值，对应 1e-4 Ω·m
# LOG10_RHO_MAX = 5.0  # 电阻率上界的 log10 值，对应 1e5 Ω·m




# DLS_LINESEARCH_MAX_TRIALS = 10  # 每轮 DLS 线搜索允许的最大折半试探次数
# DLS_LINESEARCH_PICK_BEST = True  # True=选择所有试探步长中 RMS 最低者，False=接受首个下降步长
# DLS_DIAGONAL_FLOOR = 1e-6  # 正则矩阵对角稳定项及阻尼参数允许的最小值
TARGET_POINT = "测点11"  # 当前处理的测点名称，用于日志及结果文件元数据

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
