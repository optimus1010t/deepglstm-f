import torch
from torch_geometric.utils import to_dense_adj
import torch_geometric.utils as u
from scipy import sparse
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from models.attention import AttentionModule

class GCNNet(torch.nn.Module):
  def __init__(self, k1, k2, k3, embed_dim, num_layer, device, num_feature_xd=78, n_output=1, num_feature_xt=25, output_dim=128, dropout=0.2, layer_configs=None, use_attention=False, attention_type='both'):
    super(GCNNet, self).__init__()
    self.device = device
    # Smile graph branch
    self.k1 = k1
    self.k2 = k2
    self.k3 = k3
    self.embed_dim = embed_dim
    self.num_layer = num_layer
    self.layer_configs = layer_configs if layer_configs is not None else [3, 2, 1]

    # Define convolutions
    # We maintain the original names for backward compatibility if possible, but the logic changes to be dynamic based on layer_configs
    # The original had Conv1, Conv2, Conv3. We will reuse them and add Conv4 if needed.
    # Note: The original code hardcoded input/output dims for Conv1, Conv2, Conv3
    # Conv1: num_feature_xd -> num_feature_xd
    # Conv2: num_feature_xd -> num_feature_xd*2
    # Conv3: num_feature_xd*2 -> num_feature_xd*4

    self.Conv1 = GCNConv(num_feature_xd, num_feature_xd)
    self.Conv2 = GCNConv(num_feature_xd, num_feature_xd * 2)
    self.Conv3 = GCNConv(num_feature_xd * 2, num_feature_xd * 4)
    # Adding Conv4 for the case where we need 4 layers in a block
    # Only initialize if needed to avoid load_state_dict errors with old models
    if any(l >= 4 for l in self.layer_configs):
        self.Conv4 = GCNConv(num_feature_xd * 4, num_feature_xd * 8) # Assuming doubling dims continue

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

    # Calculate fc_g1 input dimension
    # Original: concat of h3 (from Conv3), h5 (from Conv2), h6 (from Conv1)
    # h3: output of Conv3 -> dim is num_feature_xd*4
    # h5: output of Conv2 -> dim is num_feature_xd*2
    # h6: output of Conv1 -> dim is num_feature_xd
    # Total: 78*4 + 78*2 + 78 = 312 + 156 + 78 = 546. Matches original 546.

    # We need to calculate this dynamically based on which blocks are enabled (k1, k2, k3...) and how many layers each has.
    # k1=1 -> 1st block enabled. Length determined by layer_configs[0]
    # k2=2 -> 2nd block enabled. Length determined by layer_configs[1]
    # k3=3 -> 3rd block enabled. Length determined by layer_configs[2]
    # There could be k4... but the constructor only has k1, k2, k3. The table implies a 4th block might be used in "DeepGLSTM -> ... + 4th block".
    # However, the constructor signature only has k1, k2, k3. We should probably stick to 3 blocks max enabled via k arguments,
    # OR assume that if we are doing the "4th block" experiment, we might need to interpret k params differently or add k4.
    # PROPOSAL: We'll stick to the existing k1, k2, k3 flags to turn on blocks.
    # But wait, Experiment 3 has 4 blocks. The original code doesn't support k4.
    # We should add k4 argument or just iterate based on layer_configs length if we want to be fully dynamic,
    # but the original code uses `if self.k1 == 1`, `if self.k2 == 2`.
    # Let's add k4 to init.

    # Re-calculating fc_g1_input_dim
    fc_g1_input_dim = 0

    # Helper to get output dim of a layer index (1-based)
    def get_layer_out_dim(layer_idx):
        if layer_idx == 1: return num_feature_xd
        if layer_idx == 2: return num_feature_xd * 2
        if layer_idx == 3: return num_feature_xd * 4
        if layer_idx == 4: return num_feature_xd * 8
        return num_feature_xd # Fallback

    # Block 1 (k1=1)
    if self.k1 == 1 and len(self.layer_configs) >= 1:
        layers = self.layer_configs[0]
        fc_g1_input_dim += get_layer_out_dim(layers)

    # Block 2 (k2=2)
    if self.k2 == 2 and len(self.layer_configs) >= 2:
        layers = self.layer_configs[1]
        fc_g1_input_dim += get_layer_out_dim(layers)

    # Block 3 (k3=3)
    if self.k3 == 3 and len(self.layer_configs) >= 3:
        layers = self.layer_configs[2]
        fc_g1_input_dim += get_layer_out_dim(layers)

    # Block 4 (implied if layer_configs has 4 items)
    if len(self.layer_configs) >= 4:
        layers = self.layer_configs[3]
        fc_g1_input_dim += get_layer_out_dim(layers)

    # Wait, existing `training.py` calls it as: `model = modeling[0](k1=1,k2=2,k3=3,embed_dim=128,num_layer=1,device=device).to(device)`
    # It does NOT verify if k4 exists. We can add `k4=None` to `__init__`.

    self.fc_g1 = nn.Linear(fc_g1_input_dim, 1024)
    self.fc_g2 = nn.Linear(1024, output_dim)


    # protien sequence branch (LSTM)
    self.embedding_xt = nn.Embedding(num_feature_xt + 1, embed_dim)
    self.LSTM_xt_1 = nn.LSTM(self.embed_dim, self.embed_dim, self.num_layer, batch_first=True, bidirectional=True)
    self.fc_xt = nn.Linear(1000 * 256, output_dim)

    self.use_attention = use_attention
    if self.use_attention:
        self.attention = AttentionModule(dim1=output_dim, dim2=output_dim, hidden_dim=output_dim, attention_type=attention_type)

    # combined layers
    self.fc1 = nn.Linear(2 * output_dim, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.out = nn.Linear(512, n_output)

  def forward(self, data, hidden, cell):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    adj = to_dense_adj(edge_index)
    target = data.target

    block_outputs = []

    # Helper to run layers
    def run_layers(x_in, edge_idx, num_layers):
        h = x_in
        # Layer 1
        if num_layers >= 1:
            h = self.Conv1(h, edge_idx)
            h = self.relu(h)
        # Layer 2
        if num_layers >= 2:
            h = self.Conv2(h, edge_idx)
            h = self.relu(h)
        # Layer 3
        if num_layers >= 3:
            h = self.Conv3(h, edge_idx)
            h = self.relu(h)
        # Layer 4
        if num_layers >= 4:
            h = self.Conv4(h, edge_idx)
            h = self.relu(h)
        return h

    # Block 1
    if self.k1 == 1 and len(self.layer_configs) >= 1:
        out = run_layers(x, edge_index, self.layer_configs[0])
        block_outputs.append(out)

    # Block 2
    if self.k2 == 2 and len(self.layer_configs) >= 2:
         # Power of adj: A^2
        edge_index_square, _ = torch_sparse.spspmm(edge_index, None, edge_index, None, adj.shape[1], adj.shape[1], adj.shape[1], coalesced=True)
        out = run_layers(x, edge_index_square, self.layer_configs[1])
        block_outputs.append(out)

    # Block 3
    if self.k3 == 3 and len(self.layer_configs) >= 3:
        # We need A^2 to compute A^3? No, we can compute A^3 from A^2 * A
        # Recalculate A^2 if needed or reuse if k2 was enabled.
        # But to be safe and independent:
        if 'edge_index_square' not in locals():
             edge_index_square, _ = torch_sparse.spspmm(edge_index, None, edge_index, None, adj.shape[1], adj.shape[1], adj.shape[1], coalesced=True)

        edge_index_cube, _ = torch_sparse.spspmm(edge_index_square, None, edge_index, None, adj.shape[1], adj.shape[1], adj.shape[1], coalesced=True)
        out = run_layers(x, edge_index_cube, self.layer_configs[2])
        block_outputs.append(out)

    # Block 4 - Assuming simple expansion if we had a k4 flag or just implied by layer_configs length > 3?
    # The original code only passed k1, k2, k3.
    # The table has "4th block". This implies A^4.
    # Let's support it if layer_configs has 4 elements AND we modify how we call it.
    if len(self.layer_configs) >= 4:
        # We need A^4 = A^2 * A^2 or A^3 * A
        if 'edge_index_square' not in locals():
             edge_index_square, _ = torch_sparse.spspmm(edge_index, None, edge_index, None, adj.shape[1], adj.shape[1], adj.shape[1], coalesced=True)

        # A^4 = A^2 * A^2
        edge_index_quad, _ = torch_sparse.spspmm(edge_index_square, None, edge_index_square, None, adj.shape[1], adj.shape[1], adj.shape[1], coalesced=True)
        out = run_layers(x, edge_index_quad, self.layer_configs[3])
        block_outputs.append(out)

    if not block_outputs:
        # Fallback if no blocks enabled? Should not happen in proposed experiments
        raise ValueError("No GCN blocks enabled but forward called")

    concat = torch.cat(block_outputs, dim=1)

    x = gmp(concat, batch)  # global_max_pooling

    # flatten
    x = self.relu(self.fc_g1(x))
    x = self.dropout(x)
    x = self.fc_g2(x)
    x = self.dropout(x)

    # LSTM layer
    embedded_xt = self.embedding_xt(target)
    LSTM_xt, (hidden, cell) = self.LSTM_xt_1(embedded_xt, (hidden, cell))
    xt = LSTM_xt.contiguous().view(-1, 1000 * 256)
    xt = self.fc_xt(xt)

    # fusion
    if hasattr(self, 'use_attention') and self.use_attention:
        xc = self.attention(x, xt)
    else:
        xc = torch.cat((x, xt), 1)

    # add some dense layers
    xc = self.fc1(xc)
    xc = self.relu(xc)
    xc = self.dropout(xc)
    xc = self.fc2(xc)
    xc = self.relu(xc)
    xc = self.dropout(xc)
    out = self.out(xc)
    return out

  def init_hidden(self, batch_size):
    hidden = torch.zeros(2, batch_size, self.embed_dim).to(self.device)
    cell = torch.zeros(2, batch_size, self.embed_dim).to(self.device)
    return hidden, cell
