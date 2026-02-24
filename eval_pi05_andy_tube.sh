#!/bin/bash
# 评估 Pi0.5 模型脚本 (使用 lerobot-record)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/robotlab/lerobot

# 设置 HuggingFace 用户名
export HF_USER=FrankYuzhe

# 生成带时间戳的数据集名称
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATASET_NAME="${HF_USER}/eval_pi05_${TIMESTAMP}"

echo "===================="
echo "Pi0.5 评估脚本"
echo "模型: FrankYuzhe/pi05_0127_2138"
echo "数据集: ${DATASET_NAME}"
echo "===================="

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=hope \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id=${DATASET_NAME} \
  --dataset.single_task="put the tissue on the orange plate" \
  --dataset.episode_time_s=60 \
  --dataset.num_episodes=10 \
  --policy.path=FrankYuzhe/pi05_merged_tissue_spoon_0203_0210_2239 \
  --policy.device=cuda
