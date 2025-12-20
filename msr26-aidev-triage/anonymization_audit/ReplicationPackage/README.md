# 🔬 Replication Package: Early-Stage Prediction of Review Effort in AI-Generated Pull Requests

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-MSR%202026-blue)](https://zenodo.org/[TO_BE_FILLED])
[![Dataset](https://img.shields.io/badge/Dataset-AIDev-green)](https://huggingface.co/datasets/hao-li/AIDev)

> **Anonymized for double-blind review** | MSR 2026 Mining Challenge Submission

This repository contains the complete replication package for our MSR 2026 paper investigating the "hidden cost" of AI coding agents: a high rate of abandoned pull requests that waste maintainer effort.

## 📊 Key Findings

- **Two-Regime Reality**: 32.6% of AI-generated PRs merge instantly, but the remainder face a 64.5% ghosting risk
- **Near-Perfect Triage**: Simple complexity signals (patch size, file types) yield AUC 0.94
- **Efficient Gatekeeping**: Flagging top 20% riskiest PRs captures 82.8% of high-effort drains
- **98.4% Oracle Performance**: Achieves near-optimal effort coverage with zero-delay prediction

---

## 📂 Repository Structure

```
├── src/                       # Core package
│   ├── config.py              # Configuration constants
│   ├── features.py            # Feature engineering
│   ├── models.py              # LightGBM training
│   ├── metrics.py             # Evaluation metrics
│   └── utils.py               # Shared utilities
├── scripts/                   # Reproduction pipeline
│   ├── 01_process_data.py     # Data loading
│   ├── 02_engineer_features.py # Feature extraction (T0/T1)
│   ├── 03_train_models.py     # Model training & evaluation
│   ├── 04-25_*.py             # Analyses for paper
│   └── ...                    # (28 scripts total)
├── notebooks/                 # Interactive analysis
│   └── paper_analysis.py      # Main notebook
├── outputs/                   # Generated figures & tables
├── data/                      # Dataset directory (not included)
├── run_pipeline.py            # Master orchestrator
├── Makefile                   # Automation commands
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── CITATION.cff               # Citation metadata
└── DATASET.md                 # Dataset download instructions
```

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.9 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: ~10GB (dataset ~4GB + outputs)
- **OS**: Linux, macOS, or Windows

### 1. Clone & Setup

```bash
# Clone this repository
git clone [REPOSITORY_URL]
cd msr26-aidev-triage

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

See [`DATASET.md`](DATASET.md) for detailed instructions. Quick version:

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Download AIDev dataset (~4GB)
mkdir -p data/raw
huggingface-cli download hao-li/AIDev --repo-type dataset --local-dir data/raw/
```

### 3. Run Full Pipeline

```bash
# Execute entire reproduction pipeline (~30 minutes)
python run_pipeline.py
```

This will:
1. Process raw data → `data/processed/`
2. Engineer T0 (creation-time) features
3. Train LightGBM model
4. Generate all paper figures → `outputs/`
5. Compute all reported metrics

### 4. Verify Results

Expected outputs in `outputs/`:

| Output File | Description | Paper Reference |
|------------|-------------|-----------------|
| `model_performance.png` | AUC/PR-AUC curves | Figure 2 |
| `topk_coverage.png` | Top-K utility curve | Figure 2(a) |
| `calibration_high_cost.png` | Calibration plot | Figure 2(b) |
| `ghosting_analysis.png` | Ghosting rates by agent | Figure 4 |
| `paper_stats.json` | All paper metrics | Tables 1-5 |

**Key Metrics** (from `paper_stats.json`):
```json
{
  "T0_AUC": 0.94,
  "T0_PR_AUC": 0.87,
  "Recall_at_20pct": 0.828,
  "Precision_at_20pct": 0.838,
  "Ghosting_Rate": 0.645,
  "Instant_Merge_Rate": 0.326
}
```

---

## 📖 Detailed Usage

### Running Individual Scripts

Each script in `scripts/` can be run independently:

```bash
# Feature engineering only
python scripts/02_engineer_features.py

# Model training only
python scripts/03_train_models.py

# Robustness checks
python scripts/12_robustness_checks.py

# Bootstrap confidence intervals
python scripts/bootstrap_results.py
```

### Using the Python Package

```python
from src.features import extract_features
from src.models import train_model, evaluate_model
from src.metrics import compute_metrics

# Load data
import pandas as pd
prs = pd.read_parquet('data/raw/pull_requests.parquet')

# Extract T0 features
features = extract_features(prs, feature_set='T0')

# Train model
model = train_model(features, target='effort_score')

# Evaluate
metrics = evaluate_model(model, test_data)
print(f"AUC: {metrics['auc']:.3f}")
```

### Interactive Analysis

Open `notebooks/paper_analysis.py` in VS Code or Jupyter Lab to explore data interactively.

---

## 🧪 Reproduction Checklist

- [ ] Python 3.9+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] AIDev dataset downloaded to `data/raw/`
- [ ] Run `python run_pipeline.py`
- [ ] Verify `outputs/` contains all figures
- [ ] Check `paper_stats.json` matches paper metrics (±2%)

**Expected Runtime**: ~30 minutes on standard laptop (i5/8GB RAM)

---

## 📊 Paper Metrics Correspondence

| Paper Metric | Script | Output File |
|--------------|--------|-------------|
| **Table 1**: Agent statistics | `08_paper_stats.py` | `paper_stats.json` |
| **Table 3**: Model performance | `03_train_models.py` | `model_results.csv` |
| **Table 4**: Size-controlled AUC | `12_robustness_checks.py` | `size_quartile_auc.csv` |
| **Table 5**: Robustness checks | `12_robustness_checks.py` | `robustness_metrics.json` |
| **Figure 2**: Model performance | `06_make_figures.py` | `model_performance.png` |
| **Figure 3**: Two-regime analysis | `10_two_regime_analysis.py` | `instant_vs_normal_dist.png` |
| **Figure 4**: Ghosting analysis | `11_mechanism_heatmap.py` | `ghosting_analysis.png` |

---

## 🛠️ Troubleshooting

### Common Issues

**Import errors**:
```bash
# Ensure you're in the project root and venv is activated
pip install -r requirements.txt --upgrade
```

**Memory errors**:
```bash
# Use smaller batch size in config.py
# Or increase swap space
```

**Dataset not found**:
```bash
# Verify data structure
ls data/raw/
# Should contain: pull_requests.parquet, pr_commits.parquet, pr_comments.parquet
```

**Figures don't match paper**:
- Ensure you're using AIDev v1.0 (October 2025 snapshot)
- Check random seeds in `src/config.py` (should be 42)
- Rerun full pipeline: `python run_pipeline.py`

For more issues, see [Zenodo Q&A] or [open an issue](REPOSITORY_URL/issues).

---

## 📜 Citation

If you use this replication package, please cite:

```bibtex
@inproceedings{anonymous2026aipr,
  title={Early-Stage Prediction of Review Effort in AI-Generated Pull Requests},
  author={Anonymous for Review},
  booktitle={Proceedings of the 23rd International Conference on Mining Software Repositories},
  year={2026},
  note={MSR Mining Challenge}
}
```

**Dataset Citation**:
```bibtex
@article{li2025aidev,
  title={{The Rise of AI Teammates in Software Engineering (SE) 3.0}}, 
  author={Li, Hao and Zhang, Haoxiang and Hassan, Ahmed E.},
  journal={arXiv preprint arXiv:2507.15003},
  year={2025}
}
```

---

## ⚖️ License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **MSR 2026 Organizers** for hosting the Mining Challenge
- **AIDev Dataset Authors** (Li, Zhang, Hassan) for providing the dataset
- **Open Source Community** for tools used in this research

---

## 🔒 Ethics & Privacy

This research uses only **publicly available GitHub data** from the AIDev dataset. No private repositories, personally identifiable information (PII), or proprietary code is included. All data collection complied with GitHub's Terms of Service.

**Anonymization Notice**: This repository is anonymized for double-blind peer review. Author information will be disclosed upon acceptance.

---

**Questions?** See [DATASET.md](DATASET.md) or check the [Zenodo record](https://zenodo.org/[TO_BE_FILLED])
