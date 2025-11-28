# save as task_2/make_figs.py, then: python3 make_figs.py
import os, numpy as np, matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)

# 1) Positional encoding curves
d = 32
pos = np.arange(0, 200)
i = np.arange(0, d//2)
angles = pos[:, None] / (10000 ** (2*i/d))

pe_sin = np.sin(angles)
pe_cos = np.cos(angles)

plt.figure()
for k in [0, 1, 2, 3, 8, 9]:
    plt.plot(pos, pe_sin[:, k], alpha=0.9)
plt.title("Positional Encoding — sin components (sampled dims)")
plt.xlabel("position"); plt.ylabel("value")
plt.tight_layout(); plt.savefig("figs/positional_encoding.png"); plt.close()

# 2) Toy attention heatmap (e.g., head attending locally + one long jump)
n = 30
attn = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        attn[i, j] = np.exp(-((i-j)**2)/(2*(2.0**2)))  # local window
attn[10, 25] += 1.5  # a long-range spike (e.g., URL ↔ CTA)
attn = attn / attn.sum(axis=1, keepdims=True)

plt.figure()
plt.imshow(attn, cmap="viridis", aspect="auto")
plt.colorbar(label="attention weight")
plt.xlabel("Key/Value index (j)"); plt.ylabel("Query index (i)")
plt.title("Toy Self-Attention Heatmap")
plt.tight_layout(); plt.savefig("figs/fake_attention_heatmap.png"); plt.close()

print("Saved figs/positional_encoding.png and figs/fake_attention_heatmap.png")
