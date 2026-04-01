import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, dim1, dim2, hidden_dim=256, num_heads=4, dropout=0.1, attention_type='both'):
        super(AttentionModule, self).__init__()
        self.attention_type = attention_type
        
        if self.attention_type not in ['self', 'cross', 'both']:
            raise ValueError("attention_type must be 'self', 'cross', or 'both'")
        
        # Project both embeddings to the same hidden dimension, for compatibility
        self.proj1 = nn.Linear(dim1, hidden_dim) if dim1 != hidden_dim else nn.Identity()
        self.proj2 = nn.Linear(dim2, hidden_dim) if dim2 != hidden_dim else nn.Identity()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        if self.attention_type in ['self', 'both']:
            self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.norm_self = nn.LayerNorm(hidden_dim)
            
        if self.attention_type in ['cross', 'both']:
            self.cross_attn_12 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.cross_attn_21 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.norm_cross1 = nn.LayerNorm(hidden_dim)
            self.norm_cross2 = nn.LayerNorm(hidden_dim)
            
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, emb1, emb2):
        """
        emb1: [batch_size, dim1]
        emb2: [batch_size, dim2]
        """
        
        x1 = self.proj1(emb1).unsqueeze(1) # [batch_size, 1, hidden_dim]
        x2 = self.proj2(emb2).unsqueeze(1) # [batch_size, 1, hidden_dim]
        
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        
        out1, out2 = x1, x2
        
        if self.attention_type in ['cross', 'both']:
            # Cross attention: emb1 queries emb2 (out1 queries out2)
            attn12, _ = self.cross_attn_12(query=out1, key=out2, value=out2)
            out1 = self.norm_cross1(out1 + attn12)
            
            # Cross attention: emb2 queries emb1 (out2 queries out1)
            attn21, _ = self.cross_attn_21(query=out2, key=out1, value=out1)
            out2 = self.norm_cross2(out2 + attn21)
            
        if self.attention_type in ['self', 'both']:
            # Self attention: over the combined tokens sequence [out1, out2]
            combined = torch.cat([out1, out2], dim=1) # [batch_size, 2, hidden_dim]
            attn_combined, _ = self.self_attn(combined, combined, combined)
            combined = self.norm_self(combined + attn_combined)
            out1, out2 = combined[:, 0:1, :], combined[:, 1:2, :]
            
        # Flatten and fuse into a single vector the same way old torch.cat did contextually
        fused = torch.cat([out1.squeeze(1), out2.squeeze(1)], dim=1) # [batch_size, hidden_dim * 2]
        
        # Passes it through the final projection to ensure same final shape as previous concatenation
        out = self.ffn(fused)
        return out
