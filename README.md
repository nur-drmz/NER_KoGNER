# Adapted KoGNER — Knowledge Graph Distillation for Biomedical NER (v11.6)

Biomedical Named Entity Recognition for **GENE**, **DISEASE**, **CHEMICAL** entities using an adapted KoGNER architecture (arXiv:2503.15737) with BioBERT, span-entity matching, KG distillation, knowledge fusion, CRF decoding, and contrastive learning.

## Architecture (v11.6)

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
NER_v11_KoGNER/
├── src/                          # v11 model source code
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
│   │                             #   v11 losses, 3-stage training schedule
│   ├── evaluator/                #   Entity-level metrics
│   └── interface/                #   Version comparison + ablation runner
├── configs/
│   └── v11_config.yaml           # Full experiment configuration
├── notebooks/
│   └── NER_Thesis_Colab_v11.ipynb # Colab notebook
├── scripts/                      # v11.6 infrastructure
│   ├── train_local.py            #   Local training
│   ├── remote_train.py           #   Remote A100 training via SSH
│   ├── checkpoint_manager.py     #   Auto-resume checkpoints
│   └── experiment_tracker.py     #   Experiment logging
├── analysis/
│   └── performance_debugger.py   # Diagnostics module
├── monitor/                      # Colab monitoring + resilience
├── docs/
│   └── v11.6_architecture.md     # Full architecture documentation
├── logs/model_versions/          # Version changelogs (v11.1–v11.6)
├── data/                         # Datasets (downloaded at runtime)
├── results/                      # Evaluation outputs
├── run_v11_colab.py              # Standalone Colab runner script
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

## Loss Functions (Composite, 7 terms — staged activation)

| Loss | λ | Description | Stage |
|------|:-:|-------------|:-----:|
| NER (Focal/CRF) | 1.0 | Primary classification (γ=2.0, per-class α) | 1+ |
| Boundary | 0.3 | Span boundary detection | 2A+ |
| Entity Type | 0.3 | Entity type classification | 2A+ |
| Knowledge Alignment | 0.1 | Token-knowledge cosine alignment | 2A+ |
| BCE Language | 0.5 | Span-entity matching [KoGNER §2.2] | 1+ |
| KG Distillation | 0.15 | MSE KG teacher distillation [KoGNER §2.3] | 2A |
| Contrastive | 0.2 | Supervised contrastive on entity embeddings | 2B |

### 3-Stage Training Schedule (v11.5)

| Stage | Epochs | Active Losses |
|-------|--------|---------------|
| 1 (warmup) | 3 | NER + CRF |
| 2A (structural) | 5 | + boundary + type + distillation |
| 2B (contrastive) | 7 | + contrastive (no distillation) |

## Feature Flags

| Flag | Effect when `True` | Version |
|------|-------------------|---------|
| `use_span_representation` | SpanRepresentationLayer | v11 |
| `use_knowledge_fusion` | KnowledgeFusionLayer (configurable method) | v11 |
| `use_crf_decoder` | CRF decoder with BIO constraints | v11 |
| `use_contrastive_loss` | ContrastiveEntityLoss | v11 |
| `use_distillation_loss` | KnowledgeDistillationLoss | v11 |
| `use_biencoder` | Bi-encoder span-entity matching [KoGNER] | v11 |
| `use_gated_span_fusion` | Gated span-token fusion | v11.2 |
| `use_span_pruning` | Top-k span pruning | v11.3 |
| `fusion_method` | concat / gated / attention | v11 |


## Usage

Training runs on **Google Colab with T4/A100 GPU**:

1. Push to GitHub
2. Open on Colab: `notebooks/NER_Thesis_Colab_v11.ipynb`
3. Select GPU runtime and run cells sequentially

## Installation (local)

```bash
pip install -r requirements.txt
```
