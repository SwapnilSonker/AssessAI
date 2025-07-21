import torch
import torch.nn as nn
import math

torch.manual_seed(42)

# Step 0: Initial hidden vector (1 token with 4 features)
hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
print(f"🔹 Step 0 - Hidden Input:\n{hidden}\n")

# Step 1: First LayerNorm
layernorm1 = nn.LayerNorm(4)
norm1 = layernorm1(hidden)
print(f"🔹 Step 1 - After LayerNorm:\n{norm1}\n")

# Step 2: Linear projections for Q, K, V
Wq = nn.Linear(4, 4, bias=False)
Wk = nn.Linear(4, 4, bias=False)
Wv = nn.Linear(4, 4, bias=False)

q = Wq(norm1)
k = Wk(norm1)
v = Wv(norm1)
print(f"🔹 Step 2 - Query (Q):\n{q}")
print(f"🔹 Step 2 - Key (K):\n{k}")
print(f"🔹 Step 2 - Value (V):\n{v}\n")

# Step 3: Attention Scores and Softmax
attn_scores = q @ k.transpose(-2, -1) / math.sqrt(4)
print(f"🔹 Step 3 - Attention Scores (QKᵀ / √d_k):\n{attn_scores}")

attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
print(f"🔹 Step 3 - Attention Weights (after Softmax):\n{attn_weights}\n")

# Step 4: Apply attention weights to values
attn_output = attn_weights @ v
print(f"🔹 Step 4 - Attention Output (Weighted V):\n{attn_output}\n")

# Step 5: Output projection Wo
Wo = nn.Linear(4, 4, bias=False)
projected = Wo(attn_output)
print(f"🔹 Step 5 - Output Projection (Wo * attn_output):\n{projected}\n")

# Step 6: First Residual Connection
residual1 = hidden + projected
print(f"🔹 Step 6 - First Residual Connection:\n{residual1}\n")

# Step 7: Second LayerNorm + MLP
layernorm2 = nn.LayerNorm(4)
norm2 = layernorm2(residual1)
print(f"🔹 Step 7 - Second LayerNorm:\n{norm2}\n")

mlp = nn.Sequential(
    nn.Linear(4, 8),
    nn.SiLU(),
    nn.Linear(8, 4)
)
mlp_out = mlp(norm2)
print(f"🔹 Step 7 - MLP Output:\n{mlp_out}\n")

# Step 8: Final Residual
final_out = residual1 + mlp_out
print(f"✅ Step 8 - Final Output After Second Residual:\n{final_out}")
