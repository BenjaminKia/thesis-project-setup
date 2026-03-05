# When Do Language Models Learn to Trade?

Sentiment Trading with LLMs using Transfer Learning, Ablation, and Model Resilience.

**Author:** Benjamin Kia | **Spring 2026**

## Quick Start

```bash
# 1. Clone and enter project
git init thesis-llm-trading
cd thesis-llm-trading

# 2. Copy all files from the setup package into this directory

# 3. Create conda environment
conda env create -f environment.yml
conda activate thesis

# 4. Verify GPU
python -m src.utils.check_gpu

# 5. Set up WRDS credentials (first time only)
# You'll be prompted for username/password on first run
python -c "import wrds; db = wrds.Connection(); db.close()"

# 6. Download CRSP data
python -m src.data.download_crsp
```

## Project Structure

```
thesis-llm-trading/
├── config.yaml              # All hyperparameters and paths
├── environment.yml          # Conda environment
├── data/
│   ├── raw/
│   │   ├── news/            # Financial news articles
│   │   └── returns/         # CRSP stock returns
│   ├── processed/           # Cleaned, tokenized data
│   └── labels/              # Market-informed labels (1d, 3d, 5d, 7d)
├── models/
│   ├── checkpoints/         # Saved model weights
│   └── configs/             # Model-specific configs
├── src/
│   ├── data/
│   │   ├── download_crsp.py     # CRSP return pipeline
│   │   ├── download_news.py     # News data pipeline (TBD)
│   │   └── match_news_returns.py # Text ↔ return matching
│   ├── models/
│   │   ├── fine_tune.py         # Training loop
│   │   └── predict.py           # Inference / scoring
│   ├── evaluation/
│   │   ├── backtest.py          # Portfolio construction & SR
│   │   └── metrics.py           # Accuracy, F1, financial metrics
│   ├── interpretability/
│   │   ├── attribution.py       # Token attribution (RQ1)
│   │   ├── attention.py         # Attention rollout (RQ1)
│   │   └── probing.py           # Layer-wise probes (RQ2)
│   └── utils/
│       ├── check_gpu.py         # GPU verification
│       └── helpers.py           # Shared utilities
├── notebooks/                # Exploration & visualization
├── results/                  # Experiment outputs
├── scripts/                  # One-off scripts
├── thesis/
│   ├── chapters/             # Draft text
│   ├── figures/              # Publication-quality plots
│   └── tables/               # Result tables
└── logs/                     # Training logs
```

## Research Questions

| RQ | Question | Method |
|----|----------|--------|
| RQ1 | Which linguistic features drive trading decisions? | Token attribution, attention rollout, feature masking |
| RQ2 | When do LLMs learn financial skill? | Frozen/partial/full fine-tuning, layer probes |
| RQ3 | How sensitive are results to design choices? | Sharpe surfaces, breakpoint analysis |
| RQ4 | Do modern architectures + ensembles improve performance? | RoBERTa, majority vote, weighted ensemble |

## Hardware

- **Primary:** NVIDIA RTX 4060 (8GB VRAM) — fits BERT/FinBERT/RoBERTa at batch=16
- **Backup:** Google Colab Pro — for OPT-1.3B if needed
