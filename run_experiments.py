import os
import subprocess
import argparse

def run_cmd(cmd):
    print(f"\n========================================")
    print(f"Running: {cmd}")
    print(f"========================================\n")
    subprocess.run(cmd, shell=True, check=True)

def main(args):
    esm_flag = "--use_esm" if args.use_esm else ""
    subset_flag = f"--subset_frac {args.subset_frac}" if args.subset_frac is not None else ""
    if args.n_samples is not None:
        subset_flag = f"--n_samples {args.n_samples}"

    attn_flag = f"--use_attention --attention_type {args.attention_type}" if args.use_attention else ""

    datasets_to_run = args.dataset.split(',')

    # Make sure data is created first
    for d in datasets_to_run:
        run_cmd(f"python3 data_creation.py --dataset {d} {esm_flag} {subset_flag}")

    epochs = args.epoch

    print(f"\n=== Running Experiments with Epochs={epochs}, Subset={args.subset_frac} ===")

    print(f"\n=== Run original DeepGLSTM ===")
    for d in datasets_to_run:
        run_cmd(f"python3 training.py --dataset {d} --epoch {epochs} {subset_flag} {attn_flag} --save_file base_{d} --model DeepGLSTM")

    if args.use_esm:
        print("\n=== Experiment 1.5: Compare performance with ESM_GCN model ===")
        # 2. ESM_GCN model (default frozen)
        for d in datasets_to_run:
            run_cmd(f"python3 training.py --dataset {d} --epoch {epochs} {subset_flag} {attn_flag} --save_file esmgcn_frozen_{d} --model ESM_GCN --freeze_esm")

        print("\n=== Experiment 2: Evaluate effect of frozen vs finetuned ESM embeddings ===")
        # 3. ESM_GCN finetuned
        for d in datasets_to_run:
            if d == 'davis': # typically we only want to finetune on one large dataset as it's expensive
                run_cmd(f"python3 training.py --dataset {d} --epoch {epochs} {subset_flag} {attn_flag} --save_file esmgcn_finetune_{d} --model ESM_GCN")

        print("\n=== Experiment 3: Test cross-dataset generalization (train Davis test KIBA) ===")
        # 4. We already trained on Davis (frozen esm is generally a good baseline). We test on KIBA.
        if 'davis' in datasets_to_run and 'kiba' in datasets_to_run:
            run_cmd(f"python3 inference.py --dataset kiba --model ESM_GCN --freeze_esm --load_model pretrained_model/esmgcn_frozen_davis.model {subset_flag} {attn_flag}")

    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all requested experiments automatically")
    parser.add_argument("--dataset", type=str, default='davis,kiba', help="Comma-separated datasets to run (e.g. davis,kiba)")
    parser.add_argument("--epoch", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--subset_frac", type=float, default=None, help="Fraction of samples to use for training (e.g. 0.3 for 30%%)")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--use_esm", action="store_true", help="Whether to run ESM tokenization and ESM_GCN experiments")
    parser.add_argument("--use_attention", action="store_true", help="Use attention mechanism instead of concatenation")
    parser.add_argument("--attention_type", type=str, default="both", choices=["self", "cross", "both"], help="Type of attention to use")
    args = parser.parse_args()

    main(args)
