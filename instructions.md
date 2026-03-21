# Instructions

## 1. Installation

Follow these steps to set up the project environment. These instructions are optimized for a broader environment support including **CUDA 12.1** support.

### 1.1 Clone the Repository
```bash
git clone https://github.com/ayush1298/DeepGLSTM.git
cd DeepGLSTM
```

### 1.2 Install PyTorch (CUDA 12.1)
Install PyTorch, torchvision, and torchaudio compatible with CUDA 12.1.
```bash
python3 -m pip install --force-reinstall --no-cache-dir \
  torch==2.3.1+cu121 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

### 1.3 Install PyTorch Geometric Dependencies
Install optional dependencies for PyG (scatter, sparse, cluster, spline_conv) from the PyG wheel index.
```bash
python3 -m pip install --no-cache-dir \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
```

### 1.4 Install PyTorch Geometric
```bash
python3 -m pip install torch_geometric
```

### 1.5 Install Remaining Requirements
Install other dependencies (NumPy, Pandas, RDKit, etc.) from the requirements file.
```bash
python3 -m pip install -r requirements.txt
```

> [!NOTE]
> **RDKit Compatibility**: We are using the official `rdkit` package instead of `rdkit-pypi`, as the latter does not have a compatible version for this setup.

### 1.6 Verify Installation
Run the following Python script to verify that the correct versions are installed and CUDA is available.

```python
import torch
import torch_geometric

print("Torch:", torch.__version__)
print("CUDA :", torch.version.cuda)
print("PyG  :", torch_geometric.__version__)

# Expected Output:
# Torch: 2.3.1+cu121
# CUDA : 12.1
# PyG  : 2.7.0
```

## 2. Usage Instructions

### 2.1 Create Dataset
To create a PyTorch Geometric compatible dataset, run the `data_creation.py` script. The processed data will be saved in `data/processed/`.

```bash
python3 data_creation.py --dataset davis
```

**Running on other datasets:**
To run on other datasets (e.g., `kiba`, `DTC`, `Metz`, `ToxCast`, `Stitch`), simply change the `--dataset` argument.

```bash
python3 data_creation.py --dataset kiba
```

Default is `davis`. Ensure the raw CSV files for the dataset (e.g., `kiba_train.csv`, `kiba_test.csv`) are present in the `data/` directory.

**Using ESM Embeddings:**
If you plan to use the `ESM_GCN` model, you must also pass the `--use_esm` flag to extract protein representations using ESM-2:
```bash
python3 data_creation.py --dataset davis --use_esm
```
### 2.2 Model Training
Run the `training.py` script to train the model. You can specify the dataset using the `--dataset` argument.

```bash
python3 training.py --dataset davis
```
Default is `davis`.

**Training on other datasets:**
```bash
python3 training.py --dataset kiba
```

### New Arguments
- `--n_samples`: Train on a subset of data (e.g. `--n_samples 100` for debugging).
- `--save_file`: Filename to save the best model (e.g. `--save_file davis`). It will be saved as `pretrained_model/davis.model`.
- Scatter plots of the best model predictions will be saved to `plots/`.

Example:
```bash
python3 training.py --n_samples 100 --save_file davis
```

### 2.3 Inference on Pretrained Model
Run the `inference.py` script to test the model. Make sure to specify the dataset and the path to the trained model.

```bash
python3 inference.py --dataset davis
```
Default is `davis`.

**Inference on other datasets:**
```bash
python3 inference.py --dataset kiba --load_model pretrained_model/kiba.model
```

### Load Trained Model
Use `--load_model` to load the model you trained:
```bash
python3 inference.py --load_model pretrained_model/davis.model
```
Prediction scatter plots will be saved to `plots/`.

### 2.4 Reproducing Paper Tables
We provide a script `reproduce_table.py` to reproduce the ablation studies from the paper (Table 5 and Table 6).

**Prerequisite:** Ensure you have created the dataset files first (see [Create Dataset](#create-dataset)):
```bash
python3 data_creation.py
```

**Table 5: Effectiveness of different components**
```bash
python3 reproduce_table.py --table 5 --epoch 1000
```

**Table 6: Effectiveness of using the power graph**
```bash
python3 reproduce_table.py --table 6 --epoch 1000
```

**Run both experiments:**
```bash
python3 reproduce_table.py --table both --epoch 1000
```
Change `--epoch` to a smaller number (e.g., 10) for testing.

## 3. Newly Added: ESM-2 + GCN Architecture

We recently integrated the ESM-2 pre-trained protein language model (`facebook/esm2_t33_650M_UR50D`) linked with a custom GCN context to improve prediction power! 

### 3.1 Pick-and-Plug Models During Training
To run the standard `DeepGLSTM`, you do not need to change anything since `--model DeepGLSTM` is the default.
If you wish to switch models, use the `--model` flag. For example:
```bash
python3 training.py --dataset davis --model ESM_GCN
```
If you wish to freeze the ESM-2 representations during training (which is faster and uses less memory), add the `--freeze_esm` option:
```bash
python3 training.py --dataset davis --model ESM_GCN --freeze_esm
```

### 3.2 Running on Subset of Data
If you only wish to test the model dynamically on a fraction of the dataset, you can now use the `--subset_frac` argument:
```bash
python3 training.py --dataset davis --subset_frac 0.3 --epoch 400
```
This command trains the model for `400` epochs on just `30%` of the davis data. This argument applies identically across models.

### 3.3 Running Automated Experiments
There is a new script available `run_experiments.py` to trigger full ablations. It accepts arguments dynamically, just like `training.py`:
```bash
# This recreates the datasets internally and starts training the base DeepGLSTM benchmarking models on Davis and KIBA
# E.g. Run 400 epochs on 30% of the dataset
python3 run_experiments.py --subset_frac 0.3 --epoch 400
```

**Testing ESM_GCN Models:**
If you want to extract ESM embeddings during data creation and run the `ESM_GCN` models as well, pass the `--use_esm` flag:
```bash
python3 run_experiments.py --use_esm --subset_frac 0.3 --epoch 400
```
