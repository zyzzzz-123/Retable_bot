#!/bin/bash
# 评估 FrankYuzhe/act_andy_tube_106_01_20_5090 模型脚本

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/robotlab/lerobot


# 生成带时间戳的数据集名称
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATASET_NAME="${HF_USER}/eval_act_andy_tube_${TIMESTAMP}"

echo "===================="
echo "评估数据集名称: ${DATASET_NAME}"
echo "===================="

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=hope \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id=${DATASET_NAME} \
  --dataset.single_task="put the tissue on the orange plate" \
  --dataset.episode_time_s=200 \
  --dataset.num_episodes=10 \
  --policy.path=FrankYuzhe/act_merged_tissue_spoon_0203_0204_2202

