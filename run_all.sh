#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cnet

set -e # exit on error

# echo "Running base model"
# python run_experiments.py --epoch 1

# echo "Running base model + self attention"
# python run_experiments.py --epoch 1 --use_attention --attention_type self

# echo "Running base model + cross attention"
# python run_experiments.py --epoch 1 --use_attention --attention_type cross

# echo "Running base model + both attention"
# python run_experiments.py --epoch 1 --use_attention --attention_type both

# echo "Running ESM model"
# python run_experiments.py --epoch 1 --subset_frac 0.001 --use_esm

# echo "Running ESM model + self attention"
# python run_experiments.py --epoch 1 --subset_frac 0.001 --use_esm --use_attention --attention_type self

# echo "Running ESM model + cross attention"
# python run_experiments.py --epoch 1 --subset_frac 0.001 --use_esm --use_attention --attention_type cross

echo "Running ESM model + both attention"
python run_experiments.py --epoch 1 --subset_frac 0.001 --use_esm --use_attention --attention_type both

echo "All experiments finished successfully"
