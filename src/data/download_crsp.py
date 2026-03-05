"""
src/data/download_crsp.py
Download CRSP stock return data from WRDS.

Usage:
    python -m src.data.download_crsp

Prerequisites:
    1. WRDS account (through university)
    2. Set up credentials:
       - First run will prompt for username/password
       - Or create ~/.pgpass file: wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD
    3. pip install wrds
"""

import os
import yaml
import wrds
import pandas as pd
from pathlib import Path


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def download_crsp_daily(config: dict) -> pd.DataFrame:
    """
    Download CRSP daily stock returns.
    Returns: DataFrame with permno, date, ret, shrcd, exchcd, ticker, prc, vol
    """
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]

    print(f"Connecting to WRDS...")
    db = wrds.Connection()  # will prompt for credentials first time

    # --- Daily stock returns ---
    print(f"Downloading CRSP daily returns: {start} to {end} ...")

    query = f"""
        SELECT a.permno, a.date, a.ret, a.prc, a.vol, a.shrout,
               b.shrcd, b.exchcd, b.ticker, b.comnam
        FROM crsp.dsf AS a
        LEFT JOIN crsp.dsenames AS b
            ON a.permno = b.permno
            AND a.date >= b.namedt
            AND a.date <= b.nameendt
        WHERE a.date BETWEEN '{start}' AND '{end}'
            AND b.shrcd IN (10, 11)       -- common shares only
            AND b.exchcd IN (1, 2, 3)     -- NYSE, AMEX, NASDAQ
    """

    df_ret = db.raw_sql(query)
    print(f"  Downloaded {len(df_ret):,} daily return observations")
    print(f"  Unique stocks (permnos): {df_ret['permno'].nunique():,}")
    print(f"  Date range: {df_ret['date'].min()} to {df_ret['date'].max()}")

    # --- Market return (value-weighted) ---
    print("Downloading market returns (value-weighted index)...")

    query_mkt = f"""
        SELECT date, vwretd AS market_ret
        FROM crsp.dsi
        WHERE date BETWEEN '{start}' AND '{end}'
    """

    df_mkt = db.raw_sql(query_mkt)
    print(f"  Downloaded {len(df_mkt):,} market return observations")

    db.close()

    return df_ret, df_mkt


def compute_abnormal_returns(
    df_ret: pd.DataFrame, df_mkt: pd.DataFrame, horizon: int = 3
) -> pd.DataFrame:
    """
    Compute cumulative abnormal returns over a forward-looking horizon.

    For each stock-date:
      CAR(t, t+h) = cumulative_stock_return(t+1, t+h) - cumulative_market_return(t+1, t+h)

    This is used to create labels: CAR > 0 → outperform, CAR <= 0 → underperform
    """
    print(f"Computing {horizon}-day abnormal returns...")

    df_ret = df_ret.copy()
    df_mkt = df_mkt.copy()

    df_ret["date"] = pd.to_datetime(df_ret["date"])
    df_mkt["date"] = pd.to_datetime(df_mkt["date"])

    # Sort
    df_ret = df_ret.sort_values(["permno", "date"]).reset_index(drop=True)
    df_mkt = df_mkt.sort_values("date").reset_index(drop=True)

    # Compute forward cumulative returns per stock
    # cum_ret(t, t+h) = (1+r_{t+1}) * (1+r_{t+2}) * ... * (1+r_{t+h}) - 1
    df_ret["ret_clean"] = df_ret["ret"].fillna(0)

    cum_rets = []
    for _, group in df_ret.groupby("permno"):
        g = group.copy()
        g["fwd_cum_ret"] = (
            (1 + g["ret_clean"])
            .rolling(window=horizon, min_periods=horizon)
            .apply(lambda x: x.prod() - 1, raw=True)
            .shift(-horizon)
        )
        cum_rets.append(g)
    df_ret = pd.concat(cum_rets, ignore_index=True)

    # Compute forward cumulative market return
    df_mkt["mkt_clean"] = df_mkt["market_ret"].fillna(0)
    df_mkt["fwd_cum_mkt"] = (
        (1 + df_mkt["mkt_clean"])
        .rolling(window=horizon, min_periods=horizon)
        .apply(lambda x: x.prod() - 1, raw=True)
        .shift(-horizon)
    )

    # Merge and compute abnormal return
    df = df_ret.merge(df_mkt[["date", "fwd_cum_mkt"]], on="date", how="left")
    df["abnormal_ret"] = df["fwd_cum_ret"] - df["fwd_cum_mkt"]

    # Create binary label
    df["label"] = (df["abnormal_ret"] > 0).astype(int)

    n_valid = df["label"].notna().sum()
    label_dist = df["label"].value_counts(normalize=True)
    print(f"  Valid labels: {n_valid:,}")
    print(f"  Label distribution: {label_dist.to_dict()}")

    return df


def main():
    config = load_config()
    output_dir = Path(config["paths"]["raw_returns"])
    output_dir.mkdir(parents=True, exist_ok=True)

    label_dir = Path(config["paths"]["labels"])
    label_dir.mkdir(parents=True, exist_ok=True)

    # Download
    df_ret, df_mkt = download_crsp_daily(config)

    # Save raw data
    df_ret.to_parquet(output_dir / "crsp_daily_returns.parquet", index=False)
    df_mkt.to_parquet(output_dir / "crsp_market_returns.parquet", index=False)
    print(f"\nSaved raw returns to {output_dir}/")

    # Compute labels for baseline horizon
    horizon = config["data"]["label_horizon_days"]
    df_labelled = compute_abnormal_returns(df_ret, df_mkt, horizon=horizon)
    df_labelled.to_parquet(
        label_dir / f"crsp_labels_{horizon}d.parquet", index=False
    )
    print(f"Saved {horizon}-day labels to {label_dir}/")

    # Also pre-compute all sensitivity horizons (for RQ3 later)
    for h in config["sensitivity"]["label_horizons"]:
        if h != horizon:
            print(f"\nPre-computing {h}-day labels for sensitivity analysis...")
            df_h = compute_abnormal_returns(df_ret, df_mkt, horizon=h)
            df_h.to_parquet(label_dir / f"crsp_labels_{h}d.parquet", index=False)

    print("\n✓ CRSP data pipeline complete!")
    print(f"  Raw returns: {output_dir}/crsp_daily_returns.parquet")
    print(f"  Market returns: {output_dir}/crsp_market_returns.parquet")
    print(f"  Labels: {label_dir}/crsp_labels_*d.parquet")


if __name__ == "__main__":
    main()
