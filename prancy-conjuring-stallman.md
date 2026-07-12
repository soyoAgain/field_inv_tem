# Context
当前 `point11` 的雅可比矩阵计算位于 `jacob_cal.py`，采用对 `log10(rho)` 的中心差分：每一层都要调用两次 `tem_forward`，因此总成本随层数线性增长。现在实现是串行的，而 `inv_dls.py` 在每次 DLS 迭代中都会重新计算一次雅可比，这已经成为主要耗时来源。

并行化的关键约束不在雅可比公式本身，而在正演调用链：`tem_wrapper.py` 通过 `forwardprocess.forward_Fortran_TEM_log_time_aligned` 调用外部 Fortran 正演；外部 `forwardprocess.py` 当前把输入写到固定路径 `.../fortran_forward_log_sample_aligned/f_副本.json`，再读取固定路径 `.../forward_results.json`，并调用固定可执行文件。用户给出的 `tem_forward_log_sample_aligned.f90` 还使用了 `COMMON` 共享状态。这意味着当前实现不能直接用线程并发，也不能让多个任务共享同一套输入输出文件，否则会发生覆盖和串扰。

目标是在不改变反演数学定义的前提下，把雅可比按“列”并行化，并先把正演执行改造成可隔离的调用方式，确保结果正确且可复现。

# Recommended approach
1. 先把雅可比列计算从 `jacob_cal.py:21` 中抽成独立的“单列任务”逻辑，保持现有中心差分公式不变：每个任务只负责一个层参数 `j`，内部完成 `m_neg/m_pos` 构造、两次正演调用和一列 `J[:, j]` 的返回值计算。
2. 并行粒度定为“每一列一个进程任务”，使用进程并行而不是线程并行。原因是当前正演路径依赖固定 JSON 文件和 Fortran 共享状态，线程级并发没有安全性保证。
3. 在 `tem_wrapper.py:36` 前增加一个正演运行上下文/runner 抽象，把当前 `tem_forward(rho, thickness)` 背后的执行方式拆成：
   - 串行默认模式：保持现有行为；
   - 可隔离模式：允许为每个 worker 指定独立的输入文件、输出文件、工作目录或运行目录。
4. 配合修改外部 `forwardprocess.py:164`，让 `forward_Fortran_TEM_log_time_aligned(...)` 支持由调用方传入运行时路径，而不是把 `f_副本.json`、`forward_results.json`、可执行文件路径写死在函数内部。如果可执行文件仍依赖绝对路径，则同样要把该层改成可配置；否则不要启用并行。
5. 在 `jacob_cal.py` 中增加并行入口参数，例如 `n_jobs`，当 `n_jobs == 1` 时走原来的串行流程；当 `n_jobs > 1` 时，使用 `ProcessPoolExecutor` 或 `multiprocessing` 把各列分发给独立 worker，并在主进程中按列号组装 `J`。
6. 在 `config.py` 中加入并行配置项，例如 `JACOBIAN_N_JOBS`，并由 `inv_dls.py:69` 显式传给 `jacobian(...)`。`inv_dls.py:70` 这一次单独的 `Fm = tem_forward(...)` 保持串行即可，没有必要额外复杂化。
7. 若外部正演在当前阶段无法支持可隔离路径，则计划中止于“保留串行 fallback + 明确禁止并行模式”，不要为了加速而冒险复用固定文件路径。

# Critical files to modify
- `point11/jacob_cal.py`
  - 抽出单列 Jacobian 计算函数。
  - 增加 `n_jobs` / runner 参数。
  - 保留串行 fallback，并在并行模式下由主进程负责组装 `J`。
- `point11/tem_wrapper.py`
  - 封装正演调用上下文。
  - 让 `tem_forward(...)` 能接受可选的隔离执行参数或 runner。
  - 继续复用现有 `data_conf.json` 中的 `_WAVE_TIMES`、`_WAVE_AMPS`、`_GATED_TIME`、`_GATED_TIME_ABS`。
- `point11/inv_dls.py`
  - 把 Jacobian 并行参数接入反演主循环。
  - 保持现有 DLS / RMS 计算逻辑不变，只替换雅可比获取方式。
- `point11/config.py`
  - 增加 Jacobian 并行开关和 worker 数配置。
- `/Users/xiechushu/project/py_tem_fdem/思源湖数据/思源湖现场反演/forwardprocess.py`
  - 让 `forward_Fortran_TEM_log_time_aligned` 支持可配置的输入/输出路径、工作目录或可执行文件路径。
  - 如果当前实现必须写固定文件名，需要在这一层先消除硬编码。
- 如有必要，再检查 `/Users/xiechushu/project/fortran_forward_log_sample_aligned/tem_forward_log_sample_aligned.f90`
  - 重点确认文件路径和共享状态是否会阻碍多进程隔离运行；只有在 Python 层改不动时才继续向下改。

# Reuse existing code
- 复用 `jacob_cal.py:34-51` 现有的对数空间中心差分公式，只调整执行组织方式，不改数学定义。
- 复用 `tem_wrapper.py:55-75` 当前的 survey / waveform / gate / geometry 参数拼装逻辑，避免在并行改造时重复实现一套正演入参。
- 复用 `inv_dls.py:80-88` 现有的 DLS 解算与 RMS 评估流程，只把 `Jk = jacobian(...)` 的内部实现替换为可选并行版本。

# Verification
1. 先做功能等价性验证：
   - 用同一组 `rho`、`thickness` 分别跑串行 `jacobian` 和并行 `jacobian`；
   - 比较 `shape`、列顺序和数值误差，确认与串行结果一致。
2. 做并发安全验证：
   - 在 `n_jobs > 1` 下重复运行多次 Jacobian；
   - 确认不会出现结果缺列、列错位、JSON 读写冲突或运行中断。
3. 做反演回归验证：
   - 在 `inv_dls.py` 中分别用 `JACOBIAN_N_JOBS=1` 和 `>1` 跑至少 1~3 次迭代；
   - 比较 `Jk`、`deltaM`、`rms_history` 是否一致或在可接受误差内。
4. 做性能验证：
   - 记录单次 Jacobian 墙钟时间，比较串行与 2/4 worker 的耗时；
   - 只有在结果正确的前提下再决定默认 worker 数。
5. 如需人工核查，可运行当前 `inv_dls.py` 主流程，并结合已有测试脚本（如 `test_jacob_cal.py`、`test_tem_wrapper.py`）做辅助验证。
