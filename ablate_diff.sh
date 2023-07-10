#!/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="123 234 345"

# steps="10 50 100 500 1000 2000 5000 10000"
# steps="20000 50000 100000"
# conditions="0.0 0.1 0.5 1.0 1.5 2.0 3.0 4.0 5.0 6.0 8.0 10.0"
conditions="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

for step in $conditions; do
  for seed in $seeds; do
    echo $seed
    echo $TASK
    # python design_baselines/diff/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --hidden_size $step --name "cfd_hidden_size_$step"
    python design_baselines/diff/trainer.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --hidden_size $step --name "cfd_hidden_size_$step"
  done
done

# python design_baselines/rvs/parse_results.py --names reweighted_rvs cfd_hidden_size_32 cfd_hidden_size_64 cfd_hidden_size_128 cfd_hidden_size_256 cfd_hidden_size_512 cfd_hidden_size_2048 cfd_hidden_size_4096 cfd_hidden_size_8192 cfd_hidden_size_16384
