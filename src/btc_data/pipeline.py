import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class PipelineConfig:
    symbol: str = "BTC-USD"
    interval: str = "1d"
    start: Optional[str] = None  # e.g., "2017-01-01"
    end: Optional[str] = None    # e.g., "2026-01-01"
    out_dir: Path = Path("data")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    # yfinance returns DatetimeIndex
    df = df.rename(columns=rename_map).reset_index().rename(columns={"Date": "date"})
    # Ensure datetime and sorting
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    return df


def fetch_btcusd(cfg: PipelineConfig) -> pd.DataFrame:
    """
    Download BTC-USD OHLCV data from Yahoo Finance via yfinance.
    """
    df = yf.download(
        tickers=cfg.symbol,
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if not isinstance(df, pd.DataFrame) or df is None or df.empty:
        raise RuntimeError("No data returned from yfinance. Adjust date range or interval.")
    return _normalize_columns(df)


def clean_btcusd(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean BTC-USD OHLCV data and engineer basic return features.
    Returns (raw_like_df, cleaned_df) where cleaned_df includes return columns and outlier flags.
    """
    # Work on a copy to avoid mutating the input
    raw_df = df.copy()

    price_cols = ["open", "high", "low", "close", "adj_close"]

    # Drop rows where all price + volume are missing
    mask_all_nan = raw_df[price_cols + ["volume"]].isna().all(axis=1)
    raw_df = raw_df.loc[~mask_all_nan].reset_index(drop=True)

    # Forward/backward fill price columns to handle small gaps (e.g., exchange outages)
    raw_df[price_cols] = raw_df[price_cols].ffill().bfill()

    # Volume handling: keep NaNs as 0 to avoid fabricating volume
    raw_df["volume"] = raw_df["volume"].fillna(0)

    # Compute log returns from close
    raw_df["log_return"] = np.log(raw_df["close"]).diff()

    # Outlier detection on log returns using z-score threshold
    ret = raw_df["log_return"]
    mean = ret.mean(skipna=True)
    std = ret.std(skipna=True)
    # Avoid division by zero for very small std
    if pd.isna(std) or std == 0:
        z = pd.Series(np.nan, index=ret.index)
    else:
        z = (ret - mean) / std
    raw_df["is_outlier"] = (z.abs() > 5)

    # Winsorize/clip extreme returns at ±5σ
    lower = mean - 5 * (0 if pd.isna(std) else std)
    upper = mean + 5 * (0 if pd.isna(std) else std)
    raw_df["return_clean"] = ret.clip(lower=lower, upper=upper)

    # Prepare cleaned view (same core columns + engineered)
    cleaned_cols = ["date"] + price_cols + ["volume", "log_return", "return_clean", "is_outlier"]
    cleaned_df = raw_df[cleaned_cols].copy()

    return df.copy(), cleaned_df


def run_pipeline(cfg: PipelineConfig) -> Path:
    """
    Executes the pipeline: fetch -> clean -> write CSVs.
    Returns the path to the processed CSV.
    """
    raw = fetch_btcusd(cfg)
    raw_like, cleaned = clean_btcusd(raw)

    raw_dir = cfg.out_dir / "raw"
    proc_dir = cfg.out_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"btcusd_{cfg.interval}.csv"
    proc_path = proc_dir / f"btcusd_clean_{cfg.interval}.csv"

    raw_like.to_csv(raw_path, index=False)
    cleaned.to_csv(proc_path, index=False)

    return proc_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BTC-USD data pipeline (fetch + clean)")
    p.add_argument("--start", type=str, default=None, help="Start date, e.g., 2017-01-01")
    p.add_argument("--end", type=str, default=None, help="End date, e.g., 2026-01-01")
    p.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        help="Sampling interval",
    )
    p.add_argument("--out-dir", type=str, default="data", help="Output directory root")
    return p


def main():
    args = build_arg_parser().parse_args()
    cfg = PipelineConfig(
        start=args.start,
        end=args.end,
        interval=args.interval,
        out_dir=Path(args.out_dir),
    )
    proc_path = run_pipeline(cfg)
    print(f"Wrote cleaned data to: {proc_path}")


if __name__ == "__main__":
    main()
