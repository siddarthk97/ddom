#!/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="123 234 345 456 567"

# steps="10 50 100 500 1000 2000 5000 10000"
# steps="20000 50000 100000"
# conditions="0.0 0.5 1.0 2.0 3.0 5.0 6.0 8.0 10.0"
conditions="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

for step in $conditions; do
  for seed in $seeds; do
    echo $seed
    echo $TASK
    # python design_baselines/diff/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK
    python design_baselines/diff/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --lamda $step
  done

  python design_baselines/rvs/parse_results.py --names reweighted_rvs cfd_gaussian_small_noise_reweighted_val > "tables-cfd-noise-$TASK-$step-lambda"
done
