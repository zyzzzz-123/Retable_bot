#!/bin/bash
# 录制数据集脚本

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/robotlab/lerobot

# 设置 HuggingFace 用户名
export HF_USER=FrankYuzhe

# 生成带时间戳的数据集名称
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATASET_NAME="${HF_USER}/record_${TIMESTAMP}"

echo "===================="
echo "录制数据集脚本"
echo "数据集: ${DATASET_NAME}"
echo "===================="

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_hope \
    --robot.cameras="{  side: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_oo \
    --display_data=true \
    --dataset.repo_id=${DATASET_NAME} \
    --dataset.num_episodes=20\
    --dataset.single_task="Put the cube on the orange plate"
