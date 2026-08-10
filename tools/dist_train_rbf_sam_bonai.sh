#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
CUDA_VISIBLE_DEVICES='0,1,2,3' \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_rbf_sam_bonai.py --config $CONFIG --launcher pytorch ${@:3}

# bash tools/dist_train_rbf_sam_bonai.sh configs/rbf_sam/rbf_sam_bonai.py 4 --work-dir results/rbf_sam_bonai