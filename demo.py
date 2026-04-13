import os
import sys

sys.modules["torchvision"] = None  # Bypass local torchvision C++ error

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from torch_geometric import data as DATA
from torch_geometric.nn import global_mean_pool
from transformers import AutoTokenizer

from data_creation import smile_to_graph
from models.esm_gcn import ESMGCNNet


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "pretrained_model/esmgcn_frozen_davis_attn_both.model"
DATA_PATH = "data/davis_test.csv"
MAX_DROPDOWN_EXAMPLES = 10

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

.gradio-container {
    --app-bg-a: #1a2a52;
    --app-bg-b: #233f66;
    --app-bg-c: #090d18;
    --app-bg-d: #0f1526;
    --text-main: #e6ebff;
    --panel-bg: rgba(17, 24, 42, 0.92);
    --panel-border: #2a3451;
    --heading: #f0f4ff;
    --stat-bg-a: #15203a;
    --stat-bg-b: #11192d;
    --stat-border: #304264;
    --muted: #95accf;
    --badge-bg: #121c31;
    --badge-border: #3a4f78;
    --badge-text: #b7c9e9;
    --legend-text: #a4b7da;
    background:
        radial-gradient(circle at 10% -20%, var(--app-bg-a) 0%, transparent 30%),
        radial-gradient(circle at 95% 0%, var(--app-bg-b) 0%, transparent 24%),
        linear-gradient(180deg, var(--app-bg-c) 0%, var(--app-bg-d) 100%);
    color: var(--text-main);
    font-family: "Space Grotesk", "Avenir Next", sans-serif;
}

.gradio-container.theme-light {
    --app-bg-a: #d9e9ff;
    --app-bg-b: #f6fbff;
    --app-bg-c: #edf5ff;
    --app-bg-d: #f8fbff;
    --text-main: #16253f;
    --panel-bg: rgba(255, 255, 255, 0.9);
    --panel-border: #b8cae3;
    --heading: #10213e;
    --stat-bg-a: #f4f8ff;
    --stat-bg-b: #ebf2ff;
    --stat-border: #bfd0e7;
    --muted: #56739b;
    --badge-bg: #eef4ff;
    --badge-border: #b8cde8;
    --badge-text: #2b4568;
    --legend-text: #46648d;
}

.gradio-container.theme-light {
    color: #1c3253;
}

.panel {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 6px 30px rgba(3, 8, 20, 0.35);
}

.panel h1, .panel h2, .panel h3, .panel h4 {
    color: var(--heading);
}

.gradio-container.theme-light .panel .prose,
.gradio-container.theme-light .panel .prose *,
.gradio-container.theme-light .panel .markdown,
.gradio-container.theme-light .panel .markdown * {
    color: #1d3355 !important;
}

.gradio-container.theme-light .panel .prose h1,
.gradio-container.theme-light .panel .prose h2,
.gradio-container.theme-light .panel .prose h3,
.gradio-container.theme-light .panel .prose h4,
.gradio-container.theme-light .panel .prose strong,
.gradio-container.theme-light .panel .prose b {
    color: #102746 !important;
}

.gradio-container.theme-light .panel .prose code,
.gradio-container.theme-light .panel .markdown code {
    color: #1d3355 !important;
    background: #e8f0fb !important;
    border: 1px solid #c7d7ec !important;
    border-radius: 6px;
    padding: 1px 5px;
}

.gradio-container.theme-light .panel .prose pre,
.gradio-container.theme-light .panel .markdown pre {
    background: #f4f8ff !important;
    border: 1px solid #c7d7ec !important;
}

.gradio-container.theme-light .panel .prose li::marker {
    color: #45658f !important;
}

.gradio-container.theme-light .controls-title .prose,
.gradio-container.theme-light .controls-title .prose * {
    color: #1a3356 !important;
    opacity: 1 !important;
}

.gradio-container.theme-light .model-config .prose,
.gradio-container.theme-light .model-config .prose * {
    color: #1b3559 !important;
    opacity: 1 !important;
}

.gradio-container.theme-light .model-config .prose li::marker {
    color: #3e5f8c !important;
}

.gradio-container.theme-light .model-config .prose code {
    color: #173154 !important;
    background: #e6eef9 !important;
    border: 1px solid #bfd0e7 !important;
}

.mono {
    font-family: "IBM Plex Mono", monospace;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}

.stat-card {
    background: linear-gradient(180deg, var(--stat-bg-a) 0%, var(--stat-bg-b) 100%);
    border: 1px solid var(--stat-border);
    border-radius: 10px;
    padding: 10px 12px;
}

.stat-label {
    font-size: 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
}

.stat-value {
    margin-top: 5px;
    font-size: 22px;
    font-weight: 700;
    color: var(--heading);
}

.badge-row {
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    border: 1px solid var(--badge-border);
    background: var(--badge-bg);
    color: var(--badge-text);
}

.flow-legend {
    margin-top: 8px;
    color: var(--legend-text);
    font-size: 13px;
}

.controls-row {
    display: flex;
    gap: 12px;
    align-items: end;
}

.controls-row > * {
    flex: 1;
}

.controls-row button {
    min-height: 44px;
    font-weight: 700;
}

.topbar-row {
    align-items: stretch;
}

.theme-panel .wrap {
    justify-content: flex-end;
}

.theme-panel label {
    font-weight: 700;
}

.main-split-row {
    display: flex;
    flex-wrap: nowrap !important;
    align-items: flex-start;
    gap: 12px;
    overflow-x: auto;
}

.main-split-row > .gr-column {
    min-width: 0 !important;
}

@media (max-width: 900px) {
    .stats-grid {
        grid-template-columns: 1fr;
    }

    .controls-row {
        flex-direction: column;
        align-items: stretch;
    }

    .theme-panel .wrap {
        justify-content: flex-start;
    }

    .main-split-row {
        flex-wrap: wrap !important;
        overflow-x: visible;
    }
}
"""

THEME_TOGGLE_JS = """
(mode) => {
  const root = document.querySelector('.gradio-container');
  if (!root) return;
  root.classList.remove('theme-dark', 'theme-light');
  if (mode === 'Light') {
    root.classList.add('theme-light');
  } else {
    root.classList.add('theme-dark');
  }
}
"""


def fig_to_numpy(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
    plt.close(fig)
    return image


def select_diverse_examples(df, max_samples=10):
    unique_pairs = df.drop_duplicates(subset=["compound_iso_smiles", "target_sequence"], keep="first")
    selected_indices = []
    seen_smiles = set()
    seen_proteins = set()

    for idx, row in unique_pairs.iterrows():
        smiles = row["compound_iso_smiles"]
        protein = row["target_sequence"]
        if smiles not in seen_smiles and protein not in seen_proteins:
            selected_indices.append(idx)
            seen_smiles.add(smiles)
            seen_proteins.add(protein)
        if len(selected_indices) >= max_samples:
            break

    if len(selected_indices) < max_samples:
        for idx in unique_pairs.index:
            if idx in selected_indices:
                continue
            selected_indices.append(idx)
            if len(selected_indices) >= max_samples:
                break

    return df.loc[selected_indices].copy()


def build_choice_label(slot, source_idx, row):
    smiles = row["compound_iso_smiles"]
    affinity = float(row["affinity"])
    return f"Sample {slot:02d} | row {source_idx} | affinity {affinity:.3f} | {smiles[:22]}..."


def build_stats_html(actual, predicted, prot_len, atom_count):
    abs_error = abs(predicted - actual)
    return f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Actual Affinity</div>
            <div class="stat-value">{actual:.4f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Predicted Affinity</div>
            <div class="stat-value">{predicted:.4f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Absolute Error</div>
            <div class="stat-value">{abs_error:.4f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Protein Length</div>
            <div class="stat-value">{prot_len}</div>
        </div>
    </div>
    <div class="badge-row">
        <span class="badge">Molecule atoms: {atom_count}</span>
        <span class="badge">Attention mode: both (cross + self)</span>
        <span class="badge">Pooling level: single drug/protein token</span>
    </div>
    """


def make_attention_figure(q_heads, k_heads, v_heads, attn_weights, theme_mode="Dark"):
    is_light = str(theme_mode).strip().lower() == "light"
    fig_bg = "#eef4ff" if is_light else "#0f1526"
    ax_bg = "#ffffff" if is_light else "#11192d"
    title_color = "#182845" if is_light else "#f0f4ff"
    tick_color = "#4b6a95" if is_light else "#b3c3e3"
    spine_color = "#b7c8e2" if is_light else "#405273"
    label_color = "#35547a" if is_light else "#c6d4ef"
    edge_color = "#6d87ad" if is_light else "#d9e1f5"

    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.8), facecolor=fig_bg)

    for ax in axes.flatten():
        ax.set_facecolor(ax_bg)
        ax.tick_params(colors=tick_color)
        for spine in ax.spines.values():
            spine.set_color(spine_color)

    q_vec = q_heads[0, 0, 0, :].detach().cpu().numpy().reshape(1, -1)
    k_vec = k_heads[0, 0, 0, :].detach().cpu().numpy().reshape(1, -1)
    v_vec = v_heads[0, 0, 0, :].detach().cpu().numpy().reshape(1, -1)

    sns.heatmap(q_vec, ax=axes[0, 0], cmap="viridis", cbar=False, xticklabels=False, yticklabels=False)
    axes[0, 0].set_title("Q (Drug) | Head 1 | shape [1, 64]", color=title_color, fontsize=11)

    sns.heatmap(k_vec, ax=axes[0, 1], cmap="magma", cbar=False, xticklabels=False, yticklabels=False)
    axes[0, 1].set_title("K (Protein) | Head 1 | shape [1, 64]", color=title_color, fontsize=11)

    sns.heatmap(v_vec, ax=axes[1, 0], cmap="plasma", cbar=False, xticklabels=False, yticklabels=False)
    axes[1, 0].set_title("V (Protein) | Head 1 | shape [1, 64]", color=title_color, fontsize=11)

    weights = attn_weights[0, :, 0, 0].detach().cpu().numpy()
    axes[1, 1].bar(
        range(1, len(weights) + 1),
        weights,
        color=["#f38ba8", "#89b4fa", "#a6e3a1", "#f9e2af"],
        edgecolor=edge_color,
        linewidth=0.8,
    )
    axes[1, 1].set_xticks(range(1, len(weights) + 1))
    axes[1, 1].set_xlabel("Attention head", color=label_color)
    axes[1, 1].set_ylabel("Softmax weight", color=label_color)
    axes[1, 1].set_title("Cross Attention 12 weights (Drug queries Protein)", color=title_color, fontsize=11)
    axes[1, 1].set_ylim(0, 1.1)

    fig.suptitle("Manual Attention Math View (cross_attn_12 only)", color=title_color, fontsize=14, weight="bold")
    plt.tight_layout(pad=1.6)
    return fig_to_numpy(fig)


def draw_box(ax, x, y, w, h, text, color, text_color="#f4f7ff", fontsize=9):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#dce6ff33",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        weight="bold",
        wrap=True,
    )


def draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#d9e3ff")
    ax.add_patch(arrow)
    return arrow


def make_flowchart_figure(theme_mode="Dark"):
    is_light = str(theme_mode).strip().lower() == "light"
    fig_bg = "#eef4ff" if is_light else "#0f1526"
    text_color = "#1d3559" if is_light else "#f0f4ff"
    legend_color = "#4a6791" if is_light else "#a7b9dd"
    arrow_color = "#7f95b5" if is_light else "#d9e3ff"

    fig, ax = plt.subplots(figsize=(11.2, 9.6), facecolor=fig_bg)
    ax.set_facecolor(fig_bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    drug_color = "#1b4e8c"
    protein_color = "#125f52"
    fusion_color = "#5a2a8a"
    output_color = "#3f3f78"
    common_color = "#3a435c"

    w = 0.28
    h = 0.075

    draw_box(ax, 0.06, 0.86, w, h, "Drug input\nSMILES string", common_color)
    draw_box(ax, 0.06, 0.75, w, h, "Molecular graph\nnodes=atoms, edges=bonds", drug_color)
    draw_box(ax, 0.06, 0.64, w, h, "GCNConv x3 + global mean pool", drug_color)
    draw_box(ax, 0.06, 0.53, w, h, "Drug embedding\nshape [1, 256]", drug_color)
    draw_box(ax, 0.06, 0.42, w, h, "proj1 + LayerNorm\nshape [1, 1, 256]", drug_color)

    draw_box(ax, 0.65, 0.86, w, h, "Protein input\nFASTA sequence", common_color)
    draw_box(ax, 0.65, 0.75, w, h, "ESM-2 encoder\nlast_hidden_state [1, L, 1280]", protein_color)
    draw_box(ax, 0.65, 0.64, w, h, "EOS token select + protein_proj", protein_color)
    draw_box(ax, 0.65, 0.53, w, h, "Protein embedding\nshape [1, 256]", protein_color)
    draw_box(ax, 0.65, 0.42, w, h, "proj2 + LayerNorm\nshape [1, 1, 256]", protein_color)

    draw_box(ax, 0.35, 0.27, w, h, "Attention module\ncross_12 + cross_21 + self", fusion_color)
    draw_box(ax, 0.35, 0.16, w, h, "FFN fusion output\nshape [1, 512]", fusion_color)
    draw_box(ax, 0.35, 0.06, w, h, "MLP head\n512 -> 256 -> 128 -> 1\nBinding affinity", output_color)

    draw_arrow(ax, (0.20, 0.86), (0.20, 0.825)).set_color(arrow_color)
    draw_arrow(ax, (0.20, 0.75), (0.20, 0.715)).set_color(arrow_color)
    draw_arrow(ax, (0.20, 0.64), (0.20, 0.605)).set_color(arrow_color)
    draw_arrow(ax, (0.20, 0.53), (0.20, 0.495)).set_color(arrow_color)
    draw_arrow(ax, (0.79, 0.86), (0.79, 0.825)).set_color(arrow_color)
    draw_arrow(ax, (0.79, 0.75), (0.79, 0.715)).set_color(arrow_color)
    draw_arrow(ax, (0.79, 0.64), (0.79, 0.605)).set_color(arrow_color)
    draw_arrow(ax, (0.79, 0.53), (0.79, 0.495)).set_color(arrow_color)
    draw_arrow(ax, (0.20, 0.42), (0.48, 0.35)).set_color(arrow_color)
    draw_arrow(ax, (0.79, 0.42), (0.62, 0.35)).set_color(arrow_color)
    draw_arrow(ax, (0.49, 0.27), (0.49, 0.235)).set_color(arrow_color)
    draw_arrow(ax, (0.49, 0.16), (0.49, 0.135)).set_color(arrow_color)

    ax.text(0.06, 0.965, "Drug branch", color="#4d8dde" if is_light else "#9cc6ff", fontsize=11, weight="bold")
    ax.text(0.65, 0.965, "Protein branch", color="#2b9e89" if is_light else "#8fe3cb", fontsize=11, weight="bold")
    ax.text(
        0.03,
        0.01,
        "Legend: blue=drug path | green=protein path | purple=pair fusion",
        color=legend_color,
        fontsize=9,
    )
    fig.suptitle("Current DeepGLSTM-ESM Pipeline (Actual Code Path)", color=text_color, fontsize=15, weight="bold")
    plt.tight_layout()
    return fig_to_numpy(fig)


def build_explanation_markdown(smiles, protein_seq, actual_affinity, predicted_affinity, shape_map):
    return f"""
### Selected Davis Pair
- **Drug SMILES:** `{smiles}`
- **Protein length:** `{len(protein_seq)}`
- **Actual affinity:** `{actual_affinity:.4f}`
- **Predicted affinity:** `{predicted_affinity:.4f}`

### What the prediction path uses (`attention_type='both'`)
1. Drug embedding from GCN and protein embedding from ESM are both computed.
2. Attention block applies **cross_12** (drug->protein), **cross_21** (protein->drug), then **self-attention** over the two-token sequence.
3. Fused representation is passed to MLP layers for final affinity prediction.

### What the manual math panel shows
1. It visualizes only **cross_12** math: drug as Q, protein as K and V.
2. This is a focused explainer view of one branch, not the entire attention module.
3. At pooled level there is one token per modality, so attention score map has shape `[1, 4, 1, 1]`.

### Tensor Shapes
- `drug_emb`: `{shape_map['drug_emb']}`
- `protein_emb`: `{shape_map['protein_emb']}`
- `x1` and `x2` after projection+norm: `{shape_map['x1']}` and `{shape_map['x2']}`
- `Q_heads`, `K_heads`, `V_heads`: `{shape_map['q_heads']}`, `{shape_map['k_heads']}`, `{shape_map['v_heads']}`
- `attn_scores` and `attn_weights`: `{shape_map['attn_scores']}`, `{shape_map['attn_weights']}`
"""


def load_examples_and_choices():
    full_df = pd.read_csv(DATA_PATH)
    selected_df = select_diverse_examples(full_df, max_samples=MAX_DROPDOWN_EXAMPLES)
    labels = []
    label_to_row_idx = {}
    for slot, (idx, row) in enumerate(selected_df.iterrows(), start=1):
        label = build_choice_label(slot, idx, row)
        labels.append(label)
        label_to_row_idx[label] = idx
    return full_df, labels, label_to_row_idx


tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = ESMGCNNet(device=DEVICE, freeze_esm=True, use_attention=True, attention_type="both").to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

full_df, choices, choice_to_row_idx = load_examples_and_choices()
default_choice = choices[0]

checkpoint_status = "Loaded" if os.path.exists(MODEL_PATH) else "Not found"
model_config_markdown = "\n".join(
    [
        "### Model Configuration",
        f"- **Checkpoint:** `{MODEL_PATH}`",
        f"- **Checkpoint status:** `{checkpoint_status}`",
        "- **ESM model:** `facebook/esm2_t33_650M_UR50D`",
        "- **Drug encoder:** `GCNConv 78 -> 64 -> 128 -> 256`",
        "- **Protein projection:** `1280 -> 256`",
        "- **Attention mode:** `both` (`cross_12 + cross_21 + self`)",
        f"- **Device:** `{DEVICE}`",
    ]
)


def process_example(choice_label, theme_mode):
    row_idx = choice_to_row_idx.get(choice_label, choice_to_row_idx[default_choice])
    row = full_df.loc[row_idx]
    smiles = row["compound_iso_smiles"]
    protein_seq = row["target_sequence"]
    actual_affinity = float(row["affinity"])

    c_size, features, edge_index = smile_to_graph(smiles)
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    esm_out = tokenizer([protein_seq], truncation=True, max_length=1024, padding="max_length", return_tensors="pt")
    esm_ids = esm_out["input_ids"]
    esm_mask = esm_out["attention_mask"]

    data = DATA.Data(x=x, edge_index=edge_index, y=torch.tensor([[actual_affinity]], dtype=torch.float))
    data.target_esm_ids = esm_ids
    data.target_esm_mask = esm_mask
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    data = data.to(DEVICE)

    with torch.no_grad():
        h = model.relu(model.gcn1(data.x, data.edge_index))
        h = model.relu(model.gcn2(h, data.edge_index))
        h = model.gcn3(h, data.edge_index)
        drug_emb = global_mean_pool(h, data.batch)

        esm_res = model.esm_model(input_ids=data.target_esm_ids, attention_mask=data.target_esm_mask)
        last_hidden_state = esm_res.last_hidden_state
        lengths = data.target_esm_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden_state.shape[0], device=DEVICE)
        protein_emb_raw = last_hidden_state[batch_indices, lengths]
        protein_emb = model.protein_proj(protein_emb_raw)

        predicted_affinity = model(data)[0][0].item()

    attn_mod = model.attention
    hidden_dim = 256
    num_heads = attn_mod.cross_attn_12.num_heads
    head_dim = hidden_dim // num_heads

    with torch.no_grad():
        x1 = attn_mod.norm1(attn_mod.proj1(drug_emb).unsqueeze(1))
        x2 = attn_mod.norm2(attn_mod.proj2(protein_emb).unsqueeze(1))

        mha = attn_mod.cross_attn_12
        q_proj = mha.in_proj_weight[:hidden_dim, :]
        k_proj = mha.in_proj_weight[hidden_dim : 2 * hidden_dim, :]
        v_proj = mha.in_proj_weight[2 * hidden_dim :, :]

        q_bias = mha.in_proj_bias[:hidden_dim]
        k_bias = mha.in_proj_bias[hidden_dim : 2 * hidden_dim]
        v_bias = mha.in_proj_bias[2 * hidden_dim :]

        q = torch.matmul(x1, q_proj.t()) + q_bias
        k = torch.matmul(x2, k_proj.t()) + k_bias
        v = torch.matmul(x2, v_proj.t()) + v_bias

        batch_size = q.shape[0]
        q_heads = q.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
        k_heads = k.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
        v_heads = v.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (head_dim**0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

    shape_map = {
        "drug_emb": str(tuple(drug_emb.shape)),
        "protein_emb": str(tuple(protein_emb.shape)),
        "x1": str(tuple(x1.shape)),
        "x2": str(tuple(x2.shape)),
        "q_heads": str(tuple(q_heads.shape)),
        "k_heads": str(tuple(k_heads.shape)),
        "v_heads": str(tuple(v_heads.shape)),
        "attn_scores": str(tuple(attn_scores.shape)),
        "attn_weights": str(tuple(attn_weights.shape)),
    }

    stats_html = build_stats_html(actual_affinity, predicted_affinity, len(protein_seq), c_size)
    explanation_md = build_explanation_markdown(
        smiles=smiles,
        protein_seq=protein_seq,
        actual_affinity=actual_affinity,
        predicted_affinity=predicted_affinity,
        shape_map=shape_map,
    )
    attention_plot = make_attention_figure(q_heads, k_heads, v_heads, attn_weights, theme_mode=theme_mode)
    flowchart_plot = make_flowchart_figure(theme_mode=theme_mode)
    return stats_html, explanation_md, attention_plot, flowchart_plot


def create_ui():
    with gr.Blocks(title="DeepGLSTM ESM-2 Attention Explorer") as demo:
        with gr.Row(elem_classes=["topbar-row"]):
            with gr.Column(scale=8, min_width=760):
                gr.Markdown(
                    """
<div class="panel">
<h1>DeepGLSTM + ESM-2 Attention Explorer</h1>
<p>Explore one Davis drug-protein pair at a time and inspect embedding + attention math in a single dashboard.</p>
</div>
                    """
                )
            with gr.Column(scale=2, min_width=260):
                with gr.Column(elem_classes=["panel", "theme-panel"]):
                    theme_mode = gr.Radio(
                        choices=["Dark", "Light"],
                        value="Dark",
                        label="Theme",
                    )

        with gr.Column(elem_classes=["panel"]):
            gr.Markdown("### Controls", elem_classes=["controls-title"])
            with gr.Row(elem_classes=["controls-row"]):
                example_dropdown = gr.Dropdown(
                    choices=choices,
                    label="Select Davis test pair (diverse sample set)",
                    value=default_choice,
                    scale=3,
                )
                run_button = gr.Button("Compute Embeddings and Attention", variant="primary", scale=1)
            gr.Markdown(model_config_markdown, elem_classes=["mono", "model-config"])

        with gr.Row(elem_classes=["main-split-row"]):
            with gr.Column(scale=2, min_width=420):
                gr.Markdown('<div class="panel"><h3>Prediction and Attention Math</h3></div>')
                stats_output = gr.HTML(value='<div class="panel">Run a sample to populate stats.</div>')
                explanation_output = gr.Markdown(value="Run a sample to view the pair-level explanation.", elem_classes=["panel"])

            with gr.Column(scale=3, min_width=520):
                gr.Markdown('<div class="panel"><h3>Pipeline Flowchart</h3></div>')
                flowchart_output = gr.Image(
                    label="Current code-path diagram",
                    type="numpy",
                    format="png",
                )
                gr.Markdown(
                    """
<div class="panel flow-legend">
This diagram reflects the current implementation path in code:
GCN drug branch + ESM protein branch -> attention fusion -> MLP -> affinity.
</div>
                    """
                )

        with gr.Column():
            gr.Markdown('<div class="panel"><h3>Attention Maps</h3></div>')
            attention_output = gr.Image(
                label="Manual cross_attn_12 math (Q/K/V and per-head weights)",
                type="numpy",
                format="png",
            )

        theme_mode.change(
            fn=None,
            inputs=[theme_mode],
            outputs=None,
            js=THEME_TOGGLE_JS,
            queue=False,
        )

        run_button.click(
            fn=process_example,
            inputs=[example_dropdown, theme_mode],
            outputs=[stats_output, explanation_output, attention_output, flowchart_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860, css=APP_CSS)
