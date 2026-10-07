# Transformer, Block by Block 🧠📐

> **An interactive, visual, and mathematically rigorous course exploring the Transformer neural network architecture — from raw tokens to GPT, BERT, and modern reasoning models.**

[![Live Demo](https://img.shields.io/badge/Live%20Website-GitHub%20Pages-success?style=for-the-badge&logo=github)](https://rajeevranjanpandey.github.io/learn-transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Vanilla%20JS)-orange?style=for-the-badge)](index.html)
[![Pure Browser Engine](https://img.shields.io/badge/Runtime-In--Browser%20Matrix%20Math-purple?style=for-the-badge)](index.html)

---

## 🌟 What Is This Project?

**Transformer, Block by Block** is an interactive, browser-native educational platform and reference implementation designed to demystify modern Artificial Intelligence. 

Instead of treating the Transformer (Vaswani et al., 2017) as an opaque black box or relying on generic high-level diagrams, this project computes **every single numerical value live in your browser** with 2-decimal reproducibility. You follow a real, trained sentence step by step from discrete word tokens, through positional wave encodings and multi-head attention projections, to output logits, backpropagation gradients, and KV cache generation loops.

---

## 👥 Who Can Use It?

| Audience | How It Helps | Recommended View Mode |
| :--- | :--- | :--- |
| **University Students (Undergrad & Grad)** | Visualizes abstract linear algebra and tensor projections in real-time. Bridges the gap between PyTorch code and theoretical equations. | **Standard / Expert** |
| **Professors & Lecturers** | Ready-made curriculum resource. Includes 3 modular lesson plans (50m, 2x90m, reading groups), a classroom roleplay script, and diagnostic mastery rubrics. | **Standard / Expert** + Teacher Guide |
| **Machine Learning Engineers** | Demystifies low-level operations (causal masking, pre-norm vs. post-norm, KV caching memory economics, multi-query vs. grouped-query attention). | **Expert** |
| **AI Researchers** | Features an in-browser live **Ablation Lab**, 8 structured active research avenues, and a reproducible CPU training pipeline (`train.py`). | **Expert** + Research Lab |
| **Curious Tech Professionals** | Intuitive classroom analogies, interactive vector playgrounds, and zero-math plain English summaries. | **Beginner** |

---

## 🚀 How To Use It

### 1. Online Access (Zero Installation)
Visit the live website directly from any modern desktop or mobile browser:
👉 **[https://rajeevranjanpandey.github.io/learn-transformers/](https://rajeevranjanpandey.github.io/learn-transformers/)**

### 2. Difficulty View Toggles
Use the bottom toolbar to calibrate mathematical density to your background:
- 🟢 **Beginner**: Focuses on high-level intuition, classroom analogies, and visual tensor flow. Formulas are hidden.
- 🟡 **Standard**: The balanced sweet spot. Combines conceptual explanation, interactive step animation, and key tensor shapes.
- 🔴 **Expert**: Unfolds deep-dive mathematical panels (`∑ The math`), explicit vector coordinates, symbol definitions, and research trivia.

### 3. Interactive Playgrounds
- **Attention as Arrows (Step 7)**: Drag the green Query arrow in real time to witness dot-product projections, scaled softmax re-weighting, and chained value vector additions live.
- **Attention on Real Sentences (Chapter F)**: Switch context endings ("tired" vs. "wide") to observe how ambiguous pronouns (*"it"*) alter their attention distribution between subjects.
- **Gradient Descent Playground (Step 22c)**: Drop optimization marbles across loss landscapes to visualize learning rate overshoot, momentum escape, and local minima traps.
- **In-Browser Ablation Lab**: Systematically toggle Positional Encoding, Cross-Attention, or Feed-Forward networks to quantify exact performance drops on trained data.
- **Try Your Own Sentence**: Input custom 4-word sequences into the browser's trained Transformer and inspect its generation confidence scores.

### 4. Running Locally / Offline
Clone this repository and open `index.html` in any web browser — no build steps, bundlers, or package managers required:
```bash
git clone https://github.com/rajeevranjanpandey/learn-transformers.git
cd learn-transformers
# Open directly in browser
open index.html
# Or serve via Python:
python3 -m http.server 8080
```

---

## 📚 What Is Covered? (Step-by-Step Architecture)

The course is organized into five structured progression parts spanning **30 visual modules**:

### Part 1: From Text to Numbers
- **Step 0: The Job** — Sequence-to-sequence mapping and why token order alters semantic meaning.
- **Step 1: Why Not Only Recurrence?** — Sequential RNN bottleneck vs. $O(1)$ parallel access.
- **Step 2: Tokens & Vocabulary** — Subword segmentation, byte fallbacks, and discrete token ID lookups.
- **Step 3: Embeddings ($E$)** — Transforming discrete IDs into continuous metric manifolds.
- **Step 4: Positional Encoding ($PE$)** — Injecting order through multi-frequency sinusoidal waves and rotary position (RoPE).
- **Step 5: The Input Matrix ($X$)** — Stacking batch and sequence tensors into unified computing matrices.

### Part 2: Attention Mechanism
- **Step 6: The Three Roles ($Q, K, V$)** — Query questions, Key addresses, and Value communication vectors.
- **Step 7: Match Scores ($S = QK^T$)** — Vector dot-products and directional alignment.
- **Step 8: Scale ($\div \sqrt{d_k}$)** — Controlling variance to prevent extreme softmax saturation.
- **Step 9a & 9b: Masks (Padding & Causal)** — Enforcing batch safety and future-token isolation ($-\infty$).
- **Step 10: Softmax Shares ($A$)** — Exponentiation and row-wise normalization into percentage distributions.
- **Step 11: Mix the Values ($O = AV$)** — Blending communicated content along attention coefficients.
- **Step 12: Multi-Head Attention ($MHA$)** — Parallel representation subspaces merged via $W_o$.

### Part 3: The Encoder Block
- **Step 13: Residual Addition ($X + MHA$)** — Gradient expressways enabling deep signal propagation.
- **Step 14: Layer Normalization ($LN$)** — Per-token mean-centering and variance stabilization (Post-LN vs. Pre-LN).
- **Step 15: Position-wise Feed-Forward Network ($FFN$)** — Channel expansion ($d_{\text{ff}}$), ReLU / SwiGLU gating, and projection.
- **Step 16: Full Encoder Stack ($\times N$)** — Multi-layer hierarchical feature refinement.

### Part 4: The Decoder & Full Model
- **Step 17: Masked Decoder Self-Attention** — Strict autoregressive past-token conditioning.
- **Step 18: Cross-Attention** — Querying encoder memory representations from decoder states.
- **Step 19: Full Encoder-Decoder Stack** — The complete Vaswani et al. (2017) translation architecture.
- **Step 20: Architectural Halves (GPT vs. BERT)** — Decoder-only generative models vs. Encoder-only representations.
- **Step 21: Output Head & Sampling** — Logits, temperature scaling, Top-$k$, and nucleus sampling.

### Part 5: Optimization, Inference & Modern Enhancements
- **Step 22: Training & Teacher Forcing** — Cross-entropy loss across all sequence positions simultaneously.
- **Step 22b: Backpropagation & Chain Rule** — Reverse automatic differentiation through multi-layer graphs.
- **Step 22c: Gradient Descent Landscape** — Rolling down loss surfaces with adaptive step sizes.
- **Step 22d: Chat Tuning & Alignment (SFT & RLHF/DPO)** — Transforming web-text completion into assistant behavior.
- **Step 23: Autoregressive Inference Loop** — Sequential token generation and stopping conditions.
- **Step 23b: KV Cache Optimization** — Eliminating $O(n^2)$ computational redundancy during token generation.
- **Step 24: Modern Innovations** — RMSNorm, RoPE, Grouped-Query Attention (GQA), and Mixture of Experts (MoE).

---

## 🎯 Key Pedagogical & Technical Benefits

1. **Exact Mathematical Coherence**: Every tensor shown on screen reflects genuine, computed floating-point outputs from an internally consistent, PyTorch-trained 2+2 block network.
2. **Zero "Hand-Waving"**: Every formula links directly to its mathematical definition, tensor shape change, and computed numbers.
3. **Interactive Misconception Rubric**: Includes targeted oral diagnostic questions for educators to quickly evaluate deep structural understanding versus superficial equation memorization.
4. **Reproducible Python Baseline (`train.py`)**: Downloadable self-contained PyTorch script that reproduces the exact weights and loss trajectory on any CPU within 60 seconds.
5. **High Accessibility & Performance**: Built purely with accessible semantic HTML5, SVG graphics, responsive CSS, and Vanilla JavaScript. Fast load times, responsive mobile rendering, and printer-friendly PDF worksheet export.

---

## 📖 Landmark Academic Papers Covered

The repository contextualizes modern AI research through 10 peer-reviewed landmark publications:
1. **Attention Is All You Need** (Vaswani et al., *NeurIPS 2017*)
2. **BERT: Bidirectional Transformers** (Devlin et al., *NAACL 2019*)
3. **Language Models are Few-Shot Learners — GPT-3** (Brown et al., *NeurIPS 2020*)
4. **Vision Transformer (ViT)** (Dosovitskiy et al., *ICLR 2021*)
5. **InstructGPT / RLHF** (Ouyang et al., *NeurIPS 2022*)
6. **FlashAttention** (Dao et al., *NeurIPS 2022*)
7. **Grouped-Query Attention (GQA)** (Ainslie et al., *EMNLP 2023*)
8. **DeepSeek-R1: Incentivizing Reasoning via RL** (DeepSeek-AI, *Nature 2025*)
9. **Gated Attention for LLMs** (Qiu et al., *NeurIPS 2025 Best Paper*)
10. **Transformers are Inherently Succinct** (Bergsträßer et al., *ICLR 2026 Outstanding Paper*)

---

## 🔬 Eight Active Research Directions
Explore laptop-scale starter experiments across active deep learning research frontiers:
1. **Mechanistic Interpretability** (Circuit tracing & induction heads)
2. **Efficient Attention & Infinite Context** (Attention sinks & sub-quadratic operators)
3. **Length Generalization** (RoPE, ALiBi & relative frequency decay)
4. **Preference Learning** (DPO vs. PPO reward models)
5. **Empirical Scaling Laws** (Compute-optimal Chinchilla budgets)
6. **Sub-Quadratic & Sparse Architectures** (State-space models & sparse MoE routers)
7. **Reasoning & Test-Time Compute** (Chain-of-thought verification & reinforcement learning)
8. **Multimodal Cross-Attention** (Cross-modal visual token projections)

---

## 📝 How to Cite

If you use this educational interactive guide in academic courses, workshops, or research, please cite:

```bibtex
@misc{pandey2026transformerblockbyblock,
  author = {Pandey, Rajeev},
  title = {Transformer, Block by Block: An Interactive Visual Deep Learning Guide},
  year = {2026},
  howpublished = {\url{https://rajeevranjanpandey.github.io/learn-transformers/}},
  note = {Interactive educational reference based on Vaswani et al. (2017)}
}
```

---

## 📄 License
This educational project is distributed under the [MIT License](LICENSE). Contributions, translations, and pedagogical extensions are welcomed!
