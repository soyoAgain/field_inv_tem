import json
from pathlib import Path
import numpy as np
import config
import time
from scipy.sparse.linalg import cg


_LOCAL = Path(__file__).resolve().parent
with open(_LOCAL / "data_conf.json", "r", encoding="utf-8") as f:
    data_conf = json.load(f)

sample_time = data_conf["gated_time"]
sample_time_abs = data_conf["gated_time_abs"]
sample_Vobs = data_conf["gated_rx"]

sample_Vobs = np.array(sample_Vobs, dtype=float)

nlayer = config.N_LAYERS
hh_true = np.full(nlayer - 1, config.LAYER_THICKNESS, dtype=float)

t_st = float(sample_time[0])
t_ed = float(sample_time[-1])
# 使用 gated_time 的首尾时间和采样点数量
nt = len(sample_time)

# 设置反演初始值，每层电阻率为 1e-2 Ω·m
rho_iter = np.full(nlayer, 1e-2, dtype=float)

m_iter = np.log10(rho_iter) #应该用 log10还是log?

max_iter = config.MAX_ITER
log_time_sample = np.array(sample_time, dtype=float)

diag_main = np.diag(np.ones(nlayer))
diag_lower = np.diag(np.ones(nlayer-1), k=-1)
DEL = diag_main - diag_lower
DEL[0, 0] = 0
Wm = DEL

# # 打印 Wm 和 m_iter 的维度
# print(f"Wm shape: {Wm.shape}")
# print(f"m_iter shape: {m_iter.shape}")

constraint_type = config.CONSTRAINT_TYPE




for i in range(max_iter):
    print(f"================== 迭代第 {i+1} 次 =================")
    start_time = time.time()
