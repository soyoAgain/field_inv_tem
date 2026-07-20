#!/bin/bash
# ============================================================
# TEM 一维单测点反演运行脚本
# 用途: 对单个测点执行 波形提取→对数抽道→DLS反演→绘图 的完整流程
# 用法: ./run.sh [测点名] [炮号]
#        ./run.sh 测点11                   # 自动选择唯一降噪炮
#        ./run.sh 测点11 测点11_013         # 显式指定炮号
# ============================================================

# 遇错即停，管道命令失败也报错
set -euo pipefail

# 切换到脚本所在目录，确保相对路径正确
cd "$(dirname "$0")"

# -- 命令行参数解析 --
# 第一个参数：测点名称，默认测点11
point_name="${1:-测点11}"
# 第二个参数：炮号（可选），为空时自动选择唯一降噪炮
shot_name="${2:-}"

# -- 设置环境变量，供 config.py 读取 --
export TARGET_POINT="$point_name"
export SOURCE_DATA_PATH="/Users/xiechushu/project/EM_app/TEM_app/data/sy6/$point_name"
if [[ -n "$shot_name" ]]; then
  export DATA_FILE_STEM="$shot_name"
else
  unset DATA_FILE_STEM || true   # 不设炮号时清除该变量，由 config.py 自动选择
fi

# 结果输出目录：results/<测点名>/
results_dir="$(pwd)/results/$point_name"

# -- 打印运行参数 --
echo "=================================="
echo " TEM 1D Inversion — $TARGET_POINT"
echo " SOURCE_DATA_PATH=$SOURCE_DATA_PATH"
echo " DATA_FILE_STEM=${DATA_FILE_STEM:-auto}"
echo " RESULTS_DIR=$results_dir"
echo "=================================="
echo

# -- 步骤 1: 保存波形配置 --
# 读取降噪后的电流数据，提取波形关键点（上升沿、峰值、关断沿），写入 data_conf.json
echo "[1/4] Saving waveform config ..."
python save_waveform.py

# -- 步骤 2: 对数时间抽道 --
# 在关断后的指定时间窗口内，对接收电压做对数等间隔抽取，结果追加到 data_conf.json
echo
echo "[2/4] Log-spaced time gating ..."
python save_gated_data.py

# -- 步骤 3: DLS/OCCAM/MGS 反演 --
# 读取 data_conf.json 中的抽道数据，执行阻尼最小二乘反演，结果保存到 inversion_result.json
echo
echo "[3/4] DLS inversion ..."
python inv_dls.py

# -- 步骤 4: 绘制反演结果 --
# 生成每次迭代的电阻率模型图、数据拟合图和 RMS 收敛曲线
echo
echo "[4/4] Plotting results ..."
python plot_inv.py

echo
echo "=================================="
echo " Done: $TARGET_POINT"
echo "=================================="
