#!/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="123 234 345 456 567"

for seed in $seeds; do
  echo $seed
  echo $TASK
  python design_baselines/diffusers/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK
  python design_baselines/diffusers/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK
done
