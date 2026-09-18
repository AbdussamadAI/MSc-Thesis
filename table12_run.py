#!/usr/bin/env python3
"""Compute Table-12 trading metrics on the real capsule 50-stock dataset:
Momentum, MA5/MA20-cross (statistical baselines), BiLSTM, and SimpleTFT
(deep-learning models), all evaluated on the SAME weekly rebalance grid so
that every strategy shares one buy-and-hold benchmark.

Protocol (mirrors the paper and trading_metrics/compute_trading_metrics.py):
  - daily data 2020-05-18 .. 2025-05-16, three markets (US, London, Pakistan)
  - per symbol: sequential 80/20 split on the raw row index (split = int(n*0.8))
  - weekly non-overlapping origins: t = split, split+5, split+10, ... < n-5
    (identical grid used for momentum/MA-cross so all signals pool the same
    weeks and the same realized-return panel -> one shared buy-and-hold)
  - BiLSTM / SimpleTFT: 60-day lookback window ending at t, predicting the
    close price 5 trading days ahead (t+5); trained only on sequences whose
    target falls before the split (no leakage across the test boundary)
  - pooled cross-sectional long-short portfolio: top-10 / bottom-10 by signal,
    equally weighted, rebalanced weekly
  - reported stats: annualized Sharpe (52 weeks), annualized volatility,
    total compounded return, max drawdown, win rate, one-sample t-test
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

CAPSULE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CAPSULE_DIR, "code", "TFT"))
sys.path.insert(0, os.path.join(CAPSULE_DIR, "code", "BILSTM"))

from simple_tft import create_simple_tft_model  # noqa: E402
from model import BiLSTMModel  # noqa: E402

SEED = 42
LOOKBACK = 60
HORIZON = 5
EPOCHS = 50
BATCH_SIZE = 64
TOP_N = 10
PERIODS_PER_YEAR = 52
SPLIT = 0.8

BILSTM_HIDDEN = 256
BILSTM_LAYERS = 2
BILSTM_DROPOUT = 0.2
BILSTM_LR = 1e-3

TFT_HIDDEN = 256
TFT_HEADS = 8
TFT_LAYERS = 4
TFT_DROPOUT = 0.15
TFT_LR = 2e-3
TFT_WEIGHT_DECAY = 1e-5

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
TARGET = "Close"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUT_CSV = os.path.join(CAPSULE_DIR, "table12_predictions.csv")
OUT_JSON = os.path.join(CAPSULE_DIR, "table12_metrics.json")

MARKETS = ["US", "London", "Pakistan"]


def discover_symbols():
    symbols = {}
    for market in MARKETS:
        d = os.path.join(CAPSULE_DIR, "data", market)
        names = sorted(
            f[:-4] for f in os.listdir(d) if f.endswith(".csv") and not f.startswith(".")
        )
        symbols[market] = names
    return symbols


def train_and_predict(model, Xtr, ytr, Xte, epochs, lr, criterion_name="mse"):
    model = model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    if criterion_name == "smooth_l1":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=TFT_WEIGHT_DECAY)
        lossf = nn.SmoothL1Loss()
        use_clip = True
    else:
        opt = torch.optim.Adam(params, lr=lr)
        lossf = nn.MSELoss()
        use_clip = False

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    n = len(Xtr_t)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            opt.zero_grad()
            out = model(Xtr_t[idx]).squeeze(-1)
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            if use_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    model.eval()
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        preds = model(Xte_t).squeeze(-1).cpu().numpy()
    return preds


def metric_block(series: pd.Series) -> dict:
    r = series.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return {"error": "not enough weeks", "n_weeks": int(len(r))}
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    sharpe = mean / std * np.sqrt(PERIODS_PER_YEAR) if std > 0 else float("nan")
    total = float(np.prod(1.0 + r) - 1.0)
    ann_ret = float((1.0 + mean) ** PERIODS_PER_YEAR - 1.0)
    ann_vol = float(std * np.sqrt(PERIODS_PER_YEAR))
    cum = np.cumprod(1.0 + r)
    max_dd = float(np.max(1.0 - cum / np.maximum.accumulate(cum)))
    win = float(np.mean(r > 0))
    t_stat, p_value = stats.ttest_1samp(r, 0.0)
    return {
        "n_weeks": int(len(r)),
        "weekly_mean_return": mean,
        "total_return": total,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "annualized_sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
    }


def long_short_summary(panel: pd.DataFrame, signal: str, top_n: int = TOP_N):
    rows = []
    for wk, g in panel.groupby("week"):
        g = g.dropna(subset=[signal, "realized"])
        if len(g) < 2 * top_n:
            continue
        top = g.nlargest(top_n, signal)
        bot = g.nsmallest(top_n, signal)
        rows.append(
            {
                "week": wk,
                "long_short": top["realized"].mean() - bot["realized"].mean(),
                "long_only": top["realized"].mean(),
                "short_only": -bot["realized"].mean(),
                "buy_hold": g["realized"].mean(),
                "n_stocks": len(g),
            }
        )
    return pd.DataFrame(rows)


def process_symbol(market, symbol, t_start):
    path = os.path.join(CAPSULE_DIR, "data", market, f"{symbol}.csv")
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    n = len(df)
    if n < LOOKBACK + HORIZON + 60:
        print(f"skip {market}/{symbol}: too few rows ({n})", flush=True)
        return []

    split = int(n * SPLIT)
    origins = [t for t in range(split, n - HORIZON, HORIZON) if t - 19 >= 0]
    if not origins:
        print(f"skip {market}/{symbol}: no valid weekly origins", flush=True)
        return []

    data = df[FEATURES].astype(float).ffill().bfill()
    feat_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = feat_scaler.fit_transform(data)
    close = df[TARGET].astype(float).values
    tgt_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_tgt = tgt_scaler.fit_transform(close.reshape(-1, 1)).ravel()

    max_i = n - LOOKBACK - HORIZON
    Xtr_list, ytr_list = [], []
    for i in range(0, max_i + 1):
        target_idx = i + LOOKBACK + HORIZON - 1
        if target_idx >= split:
            continue
        Xtr_list.append(scaled[i : i + LOOKBACK])
        ytr_list.append(scaled_tgt[target_idx])

    if len(Xtr_list) < 100:
        print(f"skip {market}/{symbol}: too little training data ({len(Xtr_list)})", flush=True)
        return []

    Xtr = np.array(Xtr_list, dtype=np.float32)
    ytr = np.array(ytr_list, dtype=np.float32)
    Xte = np.stack([scaled[t - LOOKBACK + 1 : t + 1] for t in origins]).astype(np.float32)

    ymean, ystd = ytr.mean(), ytr.std() + 1e-6
    ytr_s = (ytr - ymean) / ystd

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    bilstm = BiLSTMModel(
        input_size=len(FEATURES),
        hidden_size=BILSTM_HIDDEN,
        num_layers=BILSTM_LAYERS,
        dropout_rate=BILSTM_DROPOUT,
    )
    pred_bilstm_s = train_and_predict(bilstm, Xtr, ytr_s, Xte, EPOCHS, BILSTM_LR, "mse")
    pred_bilstm = tgt_scaler.inverse_transform(
        (pred_bilstm_s * ystd + ymean).reshape(-1, 1)
    ).ravel()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tft = create_simple_tft_model(
        input_size=len(FEATURES),
        hidden_size=TFT_HIDDEN,
        num_heads=TFT_HEADS,
        num_layers=TFT_LAYERS,
        dropout=TFT_DROPOUT,
    )
    pred_tft_s = train_and_predict(tft, Xtr, ytr_s, Xte, EPOCHS, TFT_LR, "smooth_l1")
    pred_tft = tgt_scaler.inverse_transform((pred_tft_s * ystd + ymean).reshape(-1, 1)).ravel()

    dates = df["Date"].values
    rows = []
    for j, t in enumerate(origins):
        ma5 = close[t - 4 : t + 1].mean()
        ma20 = close[t - 19 : t + 1].mean()
        rows.append(
            {
                "market": market,
                "symbol": symbol,
                "date": dates[t],
                "close_t": float(close[t]),
                "close_t5": float(close[t + HORIZON]),
                "momentum": float(close[t] / close[t - 5] - 1.0),
                "ma_cross": float(ma5 / ma20 - 1.0),
                "pred_bilstm": float(pred_bilstm[j]),
                "pred_tft": float(pred_tft[j]),
            }
        )

    mae_b = float(np.mean(np.abs(pred_bilstm - close[[t + HORIZON for t in origins]])))
    mae_t = float(np.mean(np.abs(pred_tft - close[[t + HORIZON for t in origins]])))
    print(
        f"{market}/{symbol:10s} n={n:5d} origins={len(origins):3d} MAE_b={mae_b:9.4f} "
        f"MAE_t={mae_t:9.4f} [{time.time()-t_start:.0f}s]",
        flush=True,
    )
    return rows


def build_weekly_panel(preds: pd.DataFrame) -> pd.DataFrame:
    panel = []
    for _, row in preds.iterrows():
        iso = pd.Timestamp(row["date"]).isocalendar()
        panel.append(
            {
                "market": row["market"],
                "symbol": row["symbol"],
                "date": row["date"],
                "week": (int(iso.year), int(iso.week)),
                "realized": row["close_t5"] / row["close_t"] - 1.0,
                "momentum": row["momentum"],
                "ma_cross": row["ma_cross"],
                "pred_bilstm": row["pred_bilstm"] / row["close_t"] - 1.0,
                "pred_tft": row["pred_tft"] / row["close_t"] - 1.0,
            }
        )
    return pd.DataFrame(panel)


def main():
    t_start = time.time()
    symbols = discover_symbols()
    print("symbols:", {m: len(v) for m, v in symbols.items()}, flush=True)

    rows = []
    for market in MARKETS:
        for symbol in symbols[market]:
            rows.extend(process_symbol(market, symbol, t_start))

    preds = pd.DataFrame(rows)
    preds.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV} rows={len(preds)}", flush=True)

    panel = build_weekly_panel(preds)
    print(
        f"panel: {len(panel)} weekly rows, {panel['symbol'].nunique()} stocks, "
        f"{panel['week'].nunique()} weeks",
        flush=True,
    )

    results = {
        "protocol": {
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "split": SPLIT,
            "top_n": TOP_N,
            "periods_per_year": PERIODS_PER_YEAR,
            "bilstm": {"hidden": BILSTM_HIDDEN, "epochs": EPOCHS},
            "tft": {
                "hidden": TFT_HIDDEN,
                "layers": TFT_LAYERS,
                "heads": TFT_HEADS,
                "epochs": EPOCHS,
            },
            "note": "All four strategies (Momentum, MA-cross, BiLSTM, TFT) evaluated on "
            "the identical weekly-origin grid per symbol (t=split,split+5,... on the raw "
            "row index) so they share one pooled buy-and-hold benchmark.",
        },
        "n_stocks": int(preds["symbol"].nunique()),
        "n_weeks": int(panel["week"].nunique()),
    }

    buy_hold_summ = None
    for name, signal in [
        ("Momentum", "momentum"),
        ("MA_cross", "ma_cross"),
        ("BiLSTM", "pred_bilstm"),
        ("TFT", "pred_tft"),
    ]:
        summ = long_short_summary(panel, signal, TOP_N)
        if summ.empty:
            results[name] = {"error": "no weeks with >=2*TOP_N pooled stocks", "n_weeks": 0}
            continue
        results[name] = {
            leg: metric_block(summ[leg])
            for leg in ["long_short", "long_only", "short_only", "buy_hold"]
        }
        results[name]["n_weeks"] = int(len(summ))
        buy_hold_summ = summ

    if buy_hold_summ is not None:
        results["buy_hold_pooled"] = metric_block(buy_hold_summ["buy_hold"])
    else:
        results["buy_hold_pooled"] = {"error": "no weeks with >=2*TOP_N pooled stocks"}

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\n" + json.dumps(results, indent=2, default=float), flush=True)
    print(f"\ntotal wall time: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
