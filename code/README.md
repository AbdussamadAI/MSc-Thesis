# Capsule Forecasting Project

Time-series forecasting experiments for equities using multiple model families (BiLSTM, TFT variants, statistical baselines) plus data collection utilities.

## Project structure
- `Data Collecting and Preprocessing/`
  - `stock_streamer.py` – fetches intraday market data for US/London/Pakistan via yfinance; adds engineered features and saves CSVs under `stock_data/<market>/`.
  - `stock_news_fetcher.py` – pulls and deduplicates news articles per ticker; writes JSON under `stock_news/<market>/`.
  - `data_preprocessing.py` – sequence builder and scalers for supervised models; expects a `Date` column.
  - `evaluation.py` – MAE/MSE/RMSE helpers and pretty-print.
- `BILSTM/`
  - `train_bilstm.py` – trains a per-symbol BiLSTM on every CSV in `BILSTM/stock_data`, saves models/plots/results.
  - `experiment_symbols_bilstm_full.py` – larger BiLSTM for multi-symbol experiments (optionally GPU/DDP), saves results under `bilstm_*` dirs.
- `TFT/`
  - `simple_tft.py` – shared simplified Temporal Fusion Transformer implementation.
  - `experiment_symbols_tft.py` – single-GPU/CPU experiments over varying symbol counts using `SimpleTFT`.
  - `experiment_symbols_simple_tft.py` – SimpleTFT with CUDA stream/multi-GPU scheduling.
  - `experiment_symbols_hybrid_tft.py` – hybrid mode: two stocks per CUDA stream across GPUs.
  - `train_tft_optimized.py` – single-symbol training loop using `SimpleTFT` with higher-capacity defaults.
- `Statistical/`
  - `baseline_models.py` – classic baselines (naive, MA, exp smoothing, ARIMA/SARIMA/auto-ARIMA, linear, garch-ish).
  - `run_comprehensive_experiment.py` – orchestrates dependency checks, baseline runs, and Bayesian-optimization test (uses `SimpleTFT`).
- `requirements.txt` – Python dependencies.

## Environment notes
- Python 3.8+ recommended.
- Many scripts rely on the preprocessing helpers; set `PYTHONPATH` so `Data Collecting and Preprocessing` is importable, e.g.:
  ```bash
  export PYTHONPATH="Data Collecting and Preprocessing"
  ```
- Place stock CSVs under the expected `stock_data/<market>/` tree for each model family (e.g., `BILSTM/stock_data/US/AAPL.csv`). Scripts will create results/model/plot directories as needed.

## Running scripts
- Data collection
  - Stream prices continuously:
    ```bash
    PYTHONPATH="Data Collecting and Preprocessing" python "Data Collecting and Preprocessing/stock_streamer.py"
    ```
  - Fetch news:
    ```bash
    PYTHONPATH="Data Collecting and Preprocessing" python "Data Collecting and Preprocessing/stock_news_fetcher.py"
    ```
- BiLSTM
  - Per-symbol training:
    ```bash
    PYTHONPATH="Data Collecting and Preprocessing" python BILSTM/train_bilstm.py
    ```
  - Multi-symbol experiment:
    ```bash
    PYTHONPATH="Data Collecting and Preprocessing" python BILSTM/experiment_symbols_bilstm_full.py --device auto  # see argparse help in file
    ```
- TFT (SimpleTFT architecture everywhere)
  - Single-GPU/CPU experiment over symbol counts:
    ```bash
    PYTHONPATH="Data Collecting and Preprocessing" python TFT/experiment_symbols_tft.py --device cpu  # or cuda
    ```
  - CUDA streams variant:
    ```bash
    PYTHONPATH="Data Collecting and Preprocessing" python TFT/experiment_symbols_simple_tft.py --num_symbols 4 --multi_gpu
    ```
  - Hybrid (2 stocks per stream):
    ```bash
    PYTHONPATH="Data Collecting and Preprocessing" python TFT/experiment_symbols_hybrid_tft.py --num_symbols 4 --multi_gpu
    ```
  - Single-symbol optimized defaults:
    ```bash
    PYTHONPATH="Data Collecting and Preprocessing" python TFT/train_tft_optimized.py
    ```
- Statistical baselines
  - Run sample baselines/comprehensive pipeline:
    ```bash
    PYTHONPATH="Data Collecting and Preprocessing" python Statistical/run_comprehensive_experiment.py
    ```

## Data expectations
- CSV columns: at minimum `Date, Open, High, Low, Close, Volume`; optional `Ticker`.
- Scripts assume per-market subfolders (e.g., `US`, `London`, `Pakistan`) under each `stock_data` root.

