#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
WORK_DIRS=$3
PORT=${PORT:-28651}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --launcher pytorch 
