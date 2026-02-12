#!/bin/bash
# ============================================================
# ACT 安全评估脚本（带紧急停止 & 回Home功能）
#
# 键盘快捷键（终端需要焦点）:
#   [Space]  紧急停止（禁用所有电机扭矩）
#   [Enter]  从紧急停止恢复
#   [r]      平滑回到Home位置（2秒）
#   [→]      跳过当前episode
#   [Esc]    退出
# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/robotlab/lerobot

# ── 配置参数 ──
MODEL="FrankYuzhe/act_merged_tissue_spoon_0203_0204_2202"
ROBOT_PORT="/dev/ttyACM0"
ROBOT_ID="hope"
CAMERAS="front:/dev/video0,wrist:/dev/video4"
FPS=30
EPISODE_TIME=200      # 每个episode最长秒数
NUM_EPISODES=10
DEVICE="cuda"
REST_DURATION=2.0     # 回Home的插值时间（秒）

echo "============================================================"
echo "  ACT Safe Evaluation"
echo "  Model     : ${MODEL}"
echo "  Robot     : ${ROBOT_ID} @ ${ROBOT_PORT}"
echo "  Cameras   : ${CAMERAS}"
echo "  Episodes  : ${NUM_EPISODES} × ${EPISODE_TIME}s @ ${FPS}fps"
echo "  Device    : ${DEVICE}"
echo "============================================================"

python eval_act_safe.py \
  --model "${MODEL}" \
  --robot-port "${ROBOT_PORT}" \
  --robot-id "${ROBOT_ID}" \
  --cameras "${CAMERAS}" \
  --fps ${FPS} \
  --episode-time ${EPISODE_TIME} \
  --num-episodes ${NUM_EPISODES} \
  --device "${DEVICE}" \
  --rest-duration ${REST_DURATION}
