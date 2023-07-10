#!/usr/bin/env bash

budgets="1 2 4 8 16 32 64 128 256 512"

for b in $budgets; do
  # python design_baselines/rvs/parse_results.py --names classifier_free_diffusion_vtype_gaussian_small_noise  --seeds 123 234 345 --budget $b | tail -n 4 | head -n 1 >> op.txt
  python design_baselines/rvs/parse_results.py --names cfd_new_weighting --seeds 123 234 345 --budget $b | tail -n 4 | head -n 1 >> op2.txt
done
