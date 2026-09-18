# A Scalable Multi-GPU Framework for Stock Price Forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22831395.svg)](https://doi.org/10.5281/zenodo.22831395)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](code/LICENSE)
[![Data: CC0](https://img.shields.io/badge/Data-CC0--1.0-lightgrey.svg)](data/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](code/requirements.txt)

Code, data, and a reproducibility capsule for the paper **"A Scalable
Multi-GPU Framework for Stock Price Forecasting"** (under review, *IEEE
Access*) — an MSc thesis project on training deep forecasting models across
many stocks at once without wasting GPU time.

> If you're here to reproduce a specific number from the paper, jump to
> [Reproducing the paper](#reproducing-the-paper). If you're here to
> understand *why* this repo exists, keep reading.

## The problem this solves

Deep learning forecasters like BiLSTM and the Temporal Fusion Transformer
(TFT) beat classical statistical baselines on stock price data, but training
one model per stock — repeated across dozens of symbols — is slow and wastes
GPU capacity. The naive fix, "put a second GPU on it," doesn't help much for
workloads like this: each individual stock's model is too small to saturate
a modern GPU on its own, so you end up paying for hardware that mostly sits
idle. This repo is the empirical answer to two questions:

1. How much does GPU acceleration actually buy you for per-symbol financial
   forecasting, and where does it stop scaling?
2. Can you recover the scaling that default multi-GPU training leaves on the
   table, without touching model accuracy?

## What's actually novel here

The **Dynamic Multi-Symbol Multi-Stream Training Scheduler** (`code/TFT/experiment_symbols_simple_tft.py`,
`code/TFT/experiment_symbols_hybrid_tft.py`) is the core contribution. Instead of
training one model at a time (or replicating one model across GPUs via
standard data parallelism), it treats **each stock symbol as an independent
training job** and packs many of these small jobs onto explicit CUDA streams
per GPU — with the number of concurrent streams computed at runtime from
available VRAM, per-model memory footprint, batch size, and kernel compute
intensity. No cross-symbol gradient synchronization is needed, because the
jobs genuinely don't depend on each other. This is task-level concurrency,
not the operation-level stream overlap PyTorch already gives you inside a
single training job.

## Headline results

| Question | Finding |
|---|---|
| Does TFT beat BiLSTM? | TFT wins on 45 of 50 stocks across NYSE, LSE, and PSX (*p* < 0.001) |
| Does GPU training beat CPU? | Up to **32×** faster on one GPU, **36×** on two, for BiLSTM |
| Does a second GPU help TFT out of the box? | No — default-stream multi-GPU replication gives *no* benefit over one GPU at this scale |
| Does the proposed scheduler fix that? | Yes — it recovers the multi-GPU benefit, completing a 32-symbol TFT workload on two GPUs in **464 s vs. 743 s** (1.6× faster) than default-stream execution |
| Do the forecasts carry tradeable signal? | A weekly-rebalanced long/short backtest suggests the deep learning forecasts carry more economic content than momentum/MA-cross technical baselines — see [Table 12 reproduction](#reproducing-table-12-trading-evaluation) below |

Evaluated on 50 stocks: 20 from the NYSE, 15 from the London Stock Exchange,
and 15 from the Pakistan Stock Exchange, using daily OHLCV data.

## Repository structure

```
.
├── code/
│   ├── Data Collecting and Preprocessing/   # yfinance streamer, news fetcher, sequence builder
│   ├── BILSTM/                              # BiLSTM model + single- and multi-symbol training scripts
│   ├── TFT/                                 # SimpleTFT model + the multi-stream scheduler experiments
│   ├── Statistical/                         # ARIMA/SARIMA/GARCH/Auto-ARIMA/Exp. Smoothing baselines
│   └── requirements.txt
├── data/
│   ├── US/        # 20 NYSE symbols, daily OHLCV
│   ├── London/    # 15 LSE symbols
│   └── Pakistan/  # 15 PSX symbols
├── environment/    # Dockerfile + docker-compose (CUDA 11.8, GPU-enabled)
├── metadata/       # Code Ocean capsule metadata (thesis abstract, authorship)
├── table12_run.py           # Standalone script reproducing the paper's trading-evaluation table
├── table12_metrics.json     # ...and its output metrics
├── table12_predictions.csv  # ...and the underlying predictions
└── REPRODUCING.md            # Code Ocean / Docker reproduction instructions
```

Each subfolder under `code/` has its own scope described in
[`code/README.md`](code/README.md).

## Data

Daily OHLCV data for 50 symbols across three exchanges, collected via the
[`yfinance`](https://pypi.org/project/yfinance/) API and released here under
[CC0 1.0](data/LICENSE) (public domain). This is the exact dataset used for
every result in the paper — nothing is subsampled or synthetic.

## Getting started

### Docker (recommended, matches the paper's environment exactly)

```bash
docker compose -f environment/docker-compose.yml up ml-training
```

This builds a CUDA 11.8 + PyTorch + TensorFlow image (see
[`environment/Dockerfile`](environment/Dockerfile)) with GPU passthrough. See
[`REPRODUCING.md`](REPRODUCING.md) for the underlying `docker run` invocation
if you're driving this outside Compose (e.g. Code Ocean).

### Native

```bash
pip install -r code/requirements.txt
export PYTHONPATH="code/Data Collecting and Preprocessing"
```

Training entry points:

```bash
python "code/BILSTM/train_bilstm.py"                       # single-symbol BiLSTM
python "code/TFT/experiment_symbols_simple_tft.py"          # multi-stream TFT scheduler
python "code/Statistical/run_comprehensive_experiment.py"   # statistical baselines
```

A CUDA-capable GPU is required to reproduce the multi-GPU/multi-stream
timing results; the statistical baselines and small-scale runs work fine on
CPU. The paper's own hardware was a 2× NVIDIA RTX A4500 (20 GB VRAM each)
workstation — see the paper's Experimental Setup table for full specs.

## Reproducing the paper

### Table 12 (trading evaluation)

```bash
python table12_run.py
```

This computes the paper's weekly-rebalanced long/short portfolio evaluation
(Momentum and MA-cross baselines vs. BiLSTM and SimpleTFT signals) on the
full 50-stock dataset, on one shared buy-and-hold benchmark grid, and writes
`table12_metrics.json` / `table12_predictions.csv`. The docstring at the top
of the script spells out the exact protocol (lookback, horizon, split,
rebalance grid) so the numbers are independently checkable.

The `*_v1_misaligned.*` files alongside it are an earlier run kept
deliberately for transparency: that version had a subtly misaligned weekly
rebalance grid across strategies. `table12_run.py` and its outputs are the
corrected version and the one the paper reports.

### Everything else

The GPU scaling and multi-stream scheduler benchmarks (throughput, speedup,
wall-clock comparisons) are produced by the scripts under `code/TFT/` and
`code/BILSTM/`; see [`code/README.md`](code/README.md) for what each script
measures and how its outputs map to the paper's tables and figures.

## Citation

The paper is currently under review at *IEEE Access*; a full citation and
DOI will be added here once it's published. In the meantime, please cite
this repository:

```bibtex
@software{ali2026mscthesis,
  author    = {Ali, Abdussamad and Khan, Ayaz H. and Chaudhry, Muhammad Imran and El-Maleh, Aiman Helmi},
  title     = {{AbdussamadAI/MSc-Thesis: Code and Data for "A Scalable Multi-GPU Framework for Stock Price Forecasting"}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22831395},
  url       = {https://doi.org/10.5281/zenodo.22831395}
}
```

(See [`CITATION.cff`](CITATION.cff) — GitHub's "Cite this repository" button
uses this automatically.)

## License

- **Code** (`code/`): [MIT](code/LICENSE)
- **Data** (`data/`): [CC0 1.0](data/LICENSE) (public domain)

## Authors

- **Abdussamad Ali** — King Fahd University of Petroleum and Minerals (KFUPM) — [ORCID](https://orcid.org/0009-0003-0942-3544)
- **Ayaz H. Khan** — KFUPM
- **Muhammad Imran Chaudhry** — KFUPM — [ORCID](https://orcid.org/0000-0002-1855-3557)
- **Aiman Helmi El-Maleh** — KFUPM
