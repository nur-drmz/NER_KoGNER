# KoGNER — Knowledge Graph Distillation for Biomedical NER

Biomedical Named Entity Recognition for **GENE**, **DISEASE**, **CHEMICAL** entities using an adapted KoGNER architecture (arXiv:2503.15737) with BioBERT, span-entity matching, KG distillation, knowledge fusion, CRF decoding, and **adaptive multi-task loss weighting** (Kendall et al., 2018).

## Status: **READY TO RUN**

**The notebook is ready to run on Google Colab with A100 GPU.**

## Key Features

| Feature | Description |
|---------|-------------|
| **Adaptive Loss Weighting** | Learns per-task uncertainty σ to automatically balance 7 losses: `L = Σ (1/(2σ²)·L_i + log(σ))` |
| **Loss Randomization** | Stochastically drops non-essential losses (10% prob every 50 steps) for stability analysis |
| **3-Stage Training** | Warmup (5ep) → Structural (5ep) → Refinement (7ep) with stage-specific active losses |
| **Live Visualization** | Full training dashboard: loss curves, F1 scores, adaptive weights, gradient norms |
| **A100 Optimization** | bf16 autocast + TF32 matmul for 2x speedup |
| **Google Drive Checkpoints** | Auto-save after every epoch, resume on disconnect |

## Architecture

```
Input tokens
    │
    ▼
┌──────────────────────────────┐
│  1. BioBERT Text Encoder      │  [KoGNER §2.1]
└──────────┬───────────────────┘
           │ token_repr [B, L, 768]
           │
           ├──→ 2. Span Representation Layer [KoGNER §2.1]
           │       span_repr [B, num_spans, 768]
           │           │
           │           ├──→ 2b. Span Pruning [v11.3]
           │           │       top-k spans → pruned_repr [B, k, 768]
           │           │
           │           ├──→ 3. Entity Type Encoder [KoGNER §2.1]
           │           │       entity_type_repr [num_types, 768]
           │           │
           │           ├──→ 4. Span-Entity Matcher [v11.1 — stabilized]
           │           │       matching_logits = LayerNorm(s·n'/τ)
           │           │       → BCE Language Loss
           │           │
           │           ├──→ 4b. Evidence Aggregation [v11.4]
           │           │       logsumexp over overlapping spans
           │           │       → span_token_logits [B, L, 7]
           │           │
           │           └──→ 5. KG Teacher Distillation [KoGNER §2.3]
           │                   MSE(MLP(span), MLP(H=[Z,Z',Z'']))
           │
           │  ┌─────────────────────────────┐
           ├──│ Knowledge Encoder (FAISS)   │
           │  └──────────┬──────────────────┘
           │             │ knowledge_repr [B, L, 768]
           ▼             ▼
┌──────────────────────────┐
│  6. Knowledge Fusion     │  concat / gated / attention
└──────────┬───────────────┘
           │ fused [B, L, 768]
           ▼
┌──────────────────────────┐
│  7. Multi-task Heads     │  NER + Boundary + Type
└──────────┬───────────────┘
           │ ner_logits [B, L, 7]
           ▼
┌──────────────────────────────────┐
│  7b. Gated Span-Token Fusion     │  [v11.2]
│  gate = σ(W·[ner; span])         │
│  final = gate·ner + (1-gate)·span │
└──────────┬───────────────────────┘
           │ fused_logits [B, L, 7]
           ▼
┌──────────────────────────┐
│  8. CRF Decoder          │  Viterbi + BIO constraints
└──────────┬───────────────┘
           │ predictions [B, L]
           ▼
     Final Predictions
```

## Project Structure

```
NER_KoGNER/
├── src/                          # Model source code
│   ├── models/                   #   BioBERT encoder, span repr, entity type encoder,
│   │                             #   span-entity matcher, KG teacher, CRF decoder,
│   │                             #   span pruning, evidence aggregation, span-token fusion
│   ├── retrieval/                #   FAISS knowledge store + entries + filter + type priors
│   ├── knowledge_encoder/        #   Knowledge embedding projection
│   ├── fusion/                   #   CrossAttention + Gated fusion
│   ├── reasoning/                #   BoundaryRefiner, TypeConsistency, SelfFeedback,
│   │                             #   BioPortal reasoner, second-pass reasoning
│   ├── integration/              #   BioPortal REST API client
│   ├── calibration/              #   Per-entity threshold search
│   ├── trainer/                  #   Training loop, composite loss, data pipeline,
│   │                             #   adaptive loss weighting, 3-stage training schedule
│   ├── evaluator/                #   Entity-level metrics
│   └── interface/                #   Ablation runner
├── configs/
│   └── config.yaml               # Experiment configuration
├── notebooks/
│   └── NER_Thesis_Colab.ipynb    # Google Colab notebook
├── scripts/
│   ├── train_local.py            #   Local training
│   ├── remote_train.py           #   Remote A100 training via SSH
│   ├── checkpoint_manager.py     #   Auto-resume checkpoints
│   ├── experiment_tracker.py     #   Experiment logging
│   └── fix_v13_notebook.py       #   Notebook bug fix script
├── analysis/
│   └── performance_debugger.py   # Diagnostics module
├── monitor/                      # Colab monitoring + resilience
├── docs/
│   └── architecture.md           # Full architecture documentation
├── data/                         # Datasets (downloaded at runtime)
├── results/                      # Evaluation outputs
├── .windsurf/prompts/            # Architecture prompt for Cascade
├── requirements.txt
└── README.md
```

## Datasets

- **BC2GM**: Gene/protein mention corpus
- **Few-NERD** (medical subset): DISEASE, CHEMICAL, GENE entities
- **Synonym augmentation**: ~1000 augmented minority examples

## Label Schema

| Index | Label | Boundary | Entity Type |
|:-----:|-------|:--------:|:-----------:|
| 0 | O | O | O |
| 1 | B-GENE | B | GENE |
| 2 | I-GENE | I | GENE |
| 3 | B-CHEMICAL | B | CHEMICAL |
| 4 | I-CHEMICAL | I | CHEMICAL |
| 5 | B-DISEASE | B | DISEASE |
| 6 | I-DISEASE | I | DISEASE |

## Loss Functions

### Adaptive Multi-Task Loss (Kendall et al., 2018)

**Total Loss:** `L = Σ_i [ (1/(2σ²_i))·L_i + (1/2)·log(σ²_i) ]`

Where σ_i is a **learnable uncertainty parameter** for each task. The model automatically balances 7 losses:

| Loss | Description |
|------|-------------|
| NER (Focal/CRF) | Primary classification (γ=2.0, per-class α) |
| Boundary | Span boundary detection (3-class) |
| Entity Type | Entity type classification (4-class) |
| Knowledge Alignment | Token-knowledge cosine alignment |
| BCE Language | Span-entity matching [KoGNER §2.2] |
| KG Distillation | MSE KG teacher distillation [KoGNER §2.3] |
| CRF | Negative log-likelihood from CRF decoder |

**Loss Randomization:** Non-essential losses (boundary, type, alignment, distillation) are stochastically dropped with 10% probability every 50 steps to analyze training stability.

### 3-Stage Training Schedule

| Stage | Epochs | Active Losses | LR Multiplier |
|-------|:------:|---------------|:-------------:|
| 1 (warmup) | 5 | NER + CRF + language | 1.0 |
| 2A (structural) | 5 | + boundary + type + distillation + alignment | 1.0 |
| 2B (refinement) | 7 | NER + CRF + language + boundary + type + alignment | 1.0 |

## Feature Flags

| Flag | Effect when `True` |
|------|-------------------|
| `use_span_representation` | SpanRepresentationLayer |
| `use_knowledge_fusion` | KnowledgeFusionLayer (configurable method) |
| `use_crf_decoder` | CRF decoder with BIO constraints |
| `use_contrastive_loss` | ContrastiveEntityLoss |
| `use_distillation_loss` | KnowledgeDistillationLoss |
| `use_biencoder` | Bi-encoder span-entity matching [KoGNER] |
| `use_gated_span_fusion` | Gated span-token fusion |
| `use_span_pruning` | Top-k span pruning |
| `fusion_method` | concat / gated / attention |

## Usage

Training runs on **Google Colab with A100 GPU** (optimized for bf16 + TF32):

1. Upload `notebooks/NER_Thesis_Colab.ipynb` to Google Colab
2. Select **A100 GPU** runtime (Runtime → Change runtime type → A100 GPU)
3. Mount Google Drive (for checkpoints)
4. Run all cells sequentially


## Installation (local)

```bash
pip install -r requirements.txt
```
