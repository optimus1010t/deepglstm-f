import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModel
from models.attention import AttentionModule

class ESMGCNNet(torch.nn.Module):
    def __init__(self, device, num_features_xd=78, freeze_esm=True, use_attention=False, attention_type='both'):
        super(ESMGCNNet, self).__init__()
        self.device = device
        self.freeze_esm = freeze_esm

        self.use_attention = use_attention
        if self.use_attention:
            self.attention = AttentionModule(dim1=256, dim2=256, hidden_dim=256, attention_type=attention_type)

        # ---------------- Drug Encoder (GCN) ----------------
        # 3 GCN layers per instructions
        self.gcn1 = GCNConv(num_features_xd, 64)
        self.gcn2 = GCNConv(64, 128)
        self.gcn3 = GCNConv(128, 256)

        self.relu = nn.ReLU()
        # Output dim is 256 for drug

        # ---------------- Protein Encoder (ESM-2) ----------------
        # facebook/esm2_t33_650M_UR50D
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

        if self.freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False

        # ESM-2 650M embedding size is 1280
        self.protein_proj = nn.Linear(1280, 256)

        # ---------------- Prediction Head (MLP) ----------------
        self.fc1 = nn.Linear(256 + 256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, data, hidden=None, cell=None):
        # We accept hidden and cell parameters for API compatibility with original GCNNet, which uses LSTM

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Drug Encoder
        # GCN(64) -> ReLU -> GCN(128) -> ReLU -> GCN(256)
        h = self.gcn1(x, edge_index)
        h = self.relu(h)
        h = self.gcn2(h, edge_index)
        h = self.relu(h)
        h = self.gcn3(h, edge_index)

        # GlobalMeanPooling
        drug_emb = global_mean_pool(h, batch) # [batch_size, 256]

        # 2. Protein Encoder
        # Input: ESM token ids and attention mask
        esm_ids = data.target_esm_ids
        esm_mask = data.target_esm_mask

        # If freeze_esm is True, we can use torch.no_grad() to save memory
        if self.freeze_esm:
            with torch.no_grad():
                esm_out = self.esm_model(input_ids=esm_ids, attention_mask=esm_mask)
        else:
            esm_out = self.esm_model(input_ids=esm_ids, attention_mask=esm_mask)

        # Extract features (use pooler_output or mean of last hiddens? Often the <eos> token or mean is used)
        # Using the sequence embeddings mean pooling for ESM, because it's standard, OR <eos> token.
        # ESM masks include padding. Mean over non-padding is typical.
        # Instruction says: "Extract final token embedding" - usually ESM uses the <eos> token which is the last non-padded token.
        # Or it just means the output embeddings for the entire sequence pooled.
        # Actually esm models from huggingface typically use the first token <cls> for sentence level representation,
        # but since instruction said "final token embedding", we can use the <eos> token.

        last_hidden_state = esm_out.last_hidden_state # [batch_size, seq_len, 1280]

        # EOS token index is effectively where the attention mask ends.
        # To get the EOS token representation, we can compute lengths
        lengths = esm_mask.sum(dim=1) - 1 # -1 because length includes the <cls> token (assuming 1-based indexing for length)
        # However, huggingface transformers usually guarantees <eos> is at `lengths` index (0-indexed).

        batch_size = last_hidden_state.shape[0]
        protein_emb_raw = last_hidden_state[torch.arange(batch_size, device=self.device), lengths] # [batch_size, 1280]

        # Project
        protein_emb = self.protein_proj(protein_emb_raw) # [batch_size, 256]

        # 3. Fusion Layer
        if hasattr(self, 'use_attention') and self.use_attention:
            fused = self.attention(drug_emb, protein_emb)
        else:
            fused = torch.cat([drug_emb, protein_emb], dim=1) # [batch_size, 512]

        # 4. Prediction Head
        h_f = self.fc1(fused)
        h_f = self.relu(h_f)
        h_f = self.fc2(h_f)
        h_f = self.relu(h_f)
        h_f = self.fc3(h_f)
        h_f = self.relu(h_f)
        out = self.out(h_f)

        return out

    def init_hidden(self, batch_size):
        # Compatibility with existing training scrip
        return None, None
