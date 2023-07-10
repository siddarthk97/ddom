#!/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="123 234 345"
# temp="$3"

for seed in $seeds; do
  echo $seed
  echo $TASK
  # python design_baselines/diff/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK
  python design_baselines/diff/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --suffix "max_ds_conditioning"
done
