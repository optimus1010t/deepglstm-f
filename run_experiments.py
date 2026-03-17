import os
import subprocess
import argparse

def run_cmd(cmd):
    print(f"\n========================================")
    print(f"Running: {cmd}")
    print(f"========================================\n")
    subprocess.run(cmd, shell=True, check=True)

def main(args):
    # Make sure data is created first
    run_cmd("python3 data_creation.py --dataset davis")
    run_cmd("python3 data_creation.py --dataset kiba")

    epochs = args.epoch
    subset_flag = f"--subset_frac {args.subset_frac}" if args.subset_frac is not None else ""

    print(f"\n=== Running Experiments with Epochs={epochs}, Subset={args.subset_frac} ===")

    print("\n=== Experiment 1: Compare performance with original DeepGLSTM ===")
    # 1. Base DeepGLSTM model
    run_cmd(f"python3 training.py --dataset davis --epoch {epochs} {subset_flag} --save_file base_davis --model DeepGLSTM")
    # 2. ESM_GCN model (default frozen)
    run_cmd(f"python3 training.py --dataset davis --epoch {epochs} {subset_flag} --save_file esmgcn_frozen_davis --model ESM_GCN --freeze_esm")

    print("\n=== Experiment 2: Evaluate effect of frozen vs finetuned ESM embeddings ===")
    # 3. ESM_GCN finetuned
    run_cmd(f"python3 training.py --dataset davis --epoch {epochs} {subset_flag} --save_file esmgcn_finetune_davis --model ESM_GCN")

    print("\n=== Experiment 3: Test cross-dataset generalization (train Davis test KIBA) ===")
    # 4. We already trained on Davis (frozen esm is generally a good baseline). We test on KIBA.
    run_cmd(f"python3 inference.py --dataset kiba --model ESM_GCN --freeze_esm --load_model pretrained_model/esmgcn_frozen_davis.model")

    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all requested experiments automatically")
    parser.add_argument("--epoch", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--subset_frac", type=float, default=None, help="Fraction of samples to use for training (e.g. 0.3 for 30%)")
    args = parser.parse_args()
    
    main(args)
