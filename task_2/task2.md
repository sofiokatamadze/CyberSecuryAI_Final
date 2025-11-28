# Task 2 — Transformer Networks and Their Applications in Cybersecurity

**Course:** AI and ML for Cybersecurity
**Student:** Sofio Katamadze

---

## 1  Overview of the Transformer Architecture

A **Transformer** is a deep neural network architecture that models sequential data using self-attention instead of recurrence.
Proposed by Vaswani et al. (2017) in *Attention Is All You Need*, it eliminates the sequential processing bottleneck of RNNs and LSTMs, allowing parallel computation and efficient learning of long-range dependencies.

Each encoder layer consists of:

* **Multi-Head Self-Attention (MHSA):** Learns contextual relationships among tokens from different perspectives.
* **Feed-Forward Network (FFN):** Applies non-linear transformations to each position.
* **Residual Connections + Layer Normalization:** Improve training stability and gradient flow.

Transformers are especially powerful for cybersecurity tasks involving long, complex sequences such as log entries, network traffic, emails, and binary byte streams.

---

## 2  Mathematical Foundations

### Self-Attention

Given embeddings **X ∈ ℝⁿˣᵈ**:

**Q = XW_Q**, **K = XW_K**, **V = XW_V**

The attention operation is defined as:

**Attention(Q, K, V) = softmax((QKᵀ) / √dₖ) × V**

Each token creates query (**Q**), key (**K**), and value (**V**) vectors.  
The attention scores measure how strongly each token relates to others.  
Multi-head attention repeats this in parallel to learn multiple contextual patterns.

---

### Positional Encoding

To retain order information, sinusoidal *positional encodings* are added to token embeddings:

**PE(pos, 2i) = sin(pos / 10000^(2i/d))**  
**PE(pos, 2i+1) = cos(pos / 10000^(2i/d))**

These signals let the model infer both absolute and relative positions of tokens even without recurrence.


## 3  Visualization of Key Mechanisms

Both figures below were generated using [`make_figs.py`](./make_figs.py) in this project’s root folder. They are saved under `figs/`.

### 3.1 Self-Attention Heatmap

The heatmap shows how each token ( *y-axis* ) attends to others ( *x-axis* ).
Bright cells represent strong dependencies, demonstrating how attention connects distant tokens (e.g., link ↔ CTA).

![Toy Self-Attention Heatmap](./figs/fake_attention_heatmap.png)

---

### 3.2 Positional Encoding Curves

This plot illustrates the sinusoidal positional encodings across several sample dimensions.
Different frequencies encode positions at multiple scales, giving the model a sense of order.

![Positional Encoding — sin components (sampled dims)](./figs/positional_encoding.png)

---

## 4  Applications in Cybersecurity

Transformers enable context-aware sequence analysis for various security domains:

| Cyber Area                      | Transformer Use Case                                                              |
| :------------------------------ | :-------------------------------------------------------------------------------- |
| **Phishing / Spam Detection**   | BERT-style models understand context and link structures in emails.               |
| **Network Intrusion Detection** | Self-attention captures packet dependencies over long flows for attack detection. |
| **Log Anomaly Detection**       | LogBERT learns normal log patterns and flags unusual event sequences.             |
| **Malware Analysis**            | Byte/Opcode-level Transformers learn malicious code patterns and obfuscations.    |
| **Threat Intelligence Mining**  | LLM Transformers extract IOCs and summarize cyber reports automatically.          |

By leveraging attention to weigh critical tokens, Transformers achieve state-of-the-art accuracy on heterogeneous security datasets.

---

## 5  Summary

Transformers introduced a paradigm shift in sequence modeling for cybersecurity applications.
Their self-attention mechanism captures context and relationships across entire datasets, while positional encoding preserves temporal order.
This architecture now underpins leading solutions for **phishing detection**, **intrusion detection**, and **malware classification**, enabling deep contextual understanding and automation in modern cyber defense systems.


✅ **Project Structure**

```
task_2/
├── task2.md
├── make_figs.py
└── figs/
    ├── fake_attention_heatmap.png
    └── positional_encoding.png
```
