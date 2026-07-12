import json
from pathlib import Path
import numpy as np
import config
import time
from scipy.sparse.linalg import cg
from jacob_cal import jacobian
from tem_wrapper import tem_forward


_LOCAL = Path(__file__).resolve().parent
_RESULTS = config.RESULTS_DIR
with open(_RESULTS / "data_conf.json", "r", encoding="utf-8") as f:
    data_conf = json.load(f)

sample_time = data_conf["gated_time"]
sample_time_abs = data_conf["gated_time_abs"]
sample_Vobs = data_conf["gated_rx"]

sample_Vobs = np.array(sample_Vobs, dtype=float)
sample_Vobs = sample_Vobs * config.Vobs_SCALE_FACTOR
nlayer = config.N_LAYERS


t_st = float(sample_time[0])
t_ed = float(sample_time[-1])
# 使用 gated_time 的首尾时间和采样点数量
nt = len(sample_time)

# 设置反演初始值，每层电阻率为 1e-2 Ω·m
rho_iter = np.full(nlayer, 1e-2, dtype=float)
hh_true = np.full(nlayer - 1, config.LAYER_THICKNESS, dtype=float)

m_iter = np.log10(rho_iter) #应该用 log10还是log?

max_iter = config.MAX_ITER
log_time_sample = np.array(sample_time_abs, dtype=float)

diag_main = np.diag(np.ones(nlayer))
diag_lower = np.diag(np.ones(nlayer-1), k=-1)
DEL = diag_main - diag_lower
DEL[0, 0] = 0
Wm = DEL

# # 打印 Wm 和 m_iter 的维度
# print(f"Wm shape: {Wm.shape}")
# print(f"m_iter shape: {m_iter.shape}")

constraint_type = "DLS"  # config.CONSTRAINT_TYPE if hasattr(config, 'CONSTRAINT_TYPE') else "DLS"


def rms_calc(Vobs: np.ndarray, F: np.ndarray) -> float:
    """log-space RMS between observed and predicted."""
    log_V = np.log10(np.maximum(np.abs(Vobs), 1e-30))
    log_F = np.log10(np.maximum(np.abs(F), 1e-30))
    return float(np.sqrt(np.mean((log_V - log_F) ** 2)))

alpha = config.LAMBDA_INITIAL
best_rms = np.inf
best_m = m_iter.copy()
rms_hist = []
rho_hist = []

for i in range(max_iter):
    print(f"================== 迭代第 {i+1} 次 =================")
    start_time = time.time()
    if np.any(np.isnan(m_iter)):
        print("模型参数出现NaN，终止反演。")
        break
    Jk = jacobian(rho_iter, hh_true)
    Fm = tem_forward(rho_iter, hh_true)
    if np.any(Fm <= 0):
        print("预测数据中出现非正值，终止反演。")
        break
    if constraint_type == 'DLS':
        We = np.eye(Wm.shape[0])
        if i == 0:
            alpha = config.LAMBDA_INITIAL
        alpha = alpha * config.LAMBDA_DECREASE
        print(f'正则化参数:{alpha}')
    A = np.dot(Jk.T, Jk) + alpha * np.dot(We.T, We)
    log_F = np.log10(np.maximum(np.abs(Fm), 1e-30))
    log_V = np.log10(np.maximum(np.abs(sample_Vobs), 1e-30))
    # b = np.dot(Jk.T, log_F - log_V)
    b = np.dot(Jk.T, log_V - log_F)
    deltaM, info = cg(A, b, tol=1e-10, maxiter=10000)
    m_iter = m_iter + 0.5 * deltaM
    rho_iter = 10.0 ** m_iter
    rms = rms_calc(sample_Vobs, tem_forward(rho_iter, hh_true))
    if rms < best_rms:
        best_rms = rms
        best_m = m_iter.copy()
    print(f"  RMS: {rms:.4e}")
    rms_hist.append(rms)
    rho_hist.append(rho_iter.tolist())

# 保存每次迭代的信息
result = {
    "rms_history": rms_hist,
    "rho_history": rho_hist,
    "best_rms": float(best_rms),
    "best_rho": (10.0 ** best_m).tolist(),
    "n_iterations": len(rms_hist),
    "calibration": 1.0,
}
with open(_RESULTS / "inversion_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\nSaved to inversion_result.json, best_rms={best_rms:.4e}")


