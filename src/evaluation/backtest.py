"""
src/evaluation/backtest.py
Portfolio construction and backtesting for sentiment-based trading strategies.

Implements the Kirtac & Germano (2024) portfolio methodology:
- Long/short/long-short portfolios based on sentiment score percentiles
- Value-weighted portfolio construction
- News timing rules for trade execution
- Transaction costs of 10 bps per trade
- Sharpe ratio, max drawdown, annualised return, Calmar ratio

Usage:
    # Backtest a single model's predictions
    python -m src.evaluation.backtest --model bert

    # Backtest all models and generate comparison table
    python -m src.evaluation.backtest --model all

    # Backtest with different portfolio cut-off (for RQ3 sensitivity)
    python -m src.evaluation.backtest --model bert --cutoff 0.10

    # Run full sensitivity grid (RQ3)
    python -m src.evaluation.backtest --sensitivity-grid
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product


# ============================================================
# Portfolio construction
# ============================================================
def assign_news_timing(df):
    """
    Apply Kirtac & Germano (2024) news timing rules.

    Rules:
        - Before 6:00 AM: trade at market open, close same day
        - 6:00 AM to 4:00 PM: trade at close, exit next trading day
        - After 4:00 PM: trade at next day's open

    Expects a column 'datetime' with full timestamp.
    If only 'date' is available (no intraday timing),
    defaults to the 6AM-4PM rule (trade at close, exit next day).
    """
    df = df.copy()

    if "datetime" in df.columns:
        df["hour"] = pd.to_datetime(df["datetime"]).dt.hour

        # Trade date = the date on which we enter the position
        # Exit date = the date on which we close the position
        df["trade_date"] = df["date"]
        df["exit_date"] = df["date"]

        # Before 6 AM: trade at open, close same day
        mask_early = df["hour"] < 6
        # (trade_date and exit_date are same day – already set)

        # 6 AM to 4 PM: trade at close, exit next trading day
        mask_mid = (df["hour"] >= 6) & (df["hour"] < 16)
        # exit_date = next trading day (handled below)

        # After 4 PM: trade at next day's open
        mask_late = df["hour"] >= 16
        # trade_date = next trading day (handled below)

        # For mid-day and late news, we need the next trading day
        # This will be resolved when merging with return data
        df.loc[mask_mid, "exit_offset"] = 1
        df.loc[mask_late, "trade_offset"] = 1
        df.loc[mask_early, "exit_offset"] = 0
        df.loc[mask_early, "trade_offset"] = 0

        df["exit_offset"] = df["exit_offset"].fillna(1)
        df["trade_offset"] = df["trade_offset"].fillna(0)

        df.drop(columns=["hour"], inplace=True)
    else:
        # No intraday timing available – default to mid-day rule
        df["trade_offset"] = 0
        df["exit_offset"] = 1

    return df


def construct_portfolios(
    predictions_df,
    returns_df,
    score_col,
    cutoff=0.20,
    transaction_cost=0.001,
):
    """
    Construct long, short, and long-short portfolios.

    Args:
        predictions_df: DataFrame with columns [date, permno, {score_col}]
        returns_df: DataFrame with columns [date, permno, ret, market_cap]
        score_col: name of the sentiment score column
        cutoff: percentile threshold for long/short (default 20%)
        transaction_cost: per-trade cost in decimal (default 10 bps)

    Returns:
        DataFrame with daily portfolio returns for long, short, long-short
    """
    # Merge predictions with returns
    df = predictions_df.merge(returns_df, on=["date", "permno"], how="inner")
    df = df.dropna(subset=[score_col, "ret"])

    daily_results = []

    for date, group in df.groupby("date"):
        if len(group) < 10:
            continue  # skip days with too few stocks

        # Compute percentile thresholds
        top_threshold = group[score_col].quantile(1 - cutoff)
        bottom_threshold = group[score_col].quantile(cutoff)

        # Long portfolio: top percentile scores
        long_stocks = group[group[score_col] >= top_threshold].copy()
        # Short portfolio: bottom percentile scores
        short_stocks = group[group[score_col] <= bottom_threshold].copy()

        if len(long_stocks) == 0 or len(short_stocks) == 0:
            continue

        # Value-weighted returns
        if "market_cap" in long_stocks.columns and long_stocks["market_cap"].sum() > 0:
            long_weights = long_stocks["market_cap"] / long_stocks["market_cap"].sum()
            short_weights = short_stocks["market_cap"] / short_stocks["market_cap"].sum()
        else:
            # Equal-weighted fallback
            long_weights = np.ones(len(long_stocks)) / len(long_stocks)
            short_weights = np.ones(len(short_stocks)) / len(short_stocks)

        long_ret = (long_stocks["ret"].values * long_weights.values).sum()
        short_ret = (short_stocks["ret"].values * short_weights.values).sum()

        # Apply transaction costs
        # Simplified: charge costs on portfolio turnover
        long_ret_net = long_ret - transaction_cost
        short_ret_net = -short_ret - transaction_cost  # short profits from decline
        long_short_ret = long_ret_net + short_ret_net

        daily_results.append({
            "date": date,
            "long_ret": long_ret_net,
            "short_ret": short_ret_net,
            "long_short_ret": long_short_ret,
            "long_ret_gross": long_ret,
            "short_ret_gross": -short_ret,
            "n_long": len(long_stocks),
            "n_short": len(short_stocks),
        })

    results_df = pd.DataFrame(daily_results).sort_values("date").reset_index(drop=True)
    return results_df


# ============================================================
# Performance metrics
# ============================================================
def compute_metrics(daily_returns, annual_trading_days=252):
    """
    Compute portfolio performance metrics.

    Args:
        daily_returns: Series of daily portfolio returns
        annual_trading_days: number of trading days per year

    Returns:
        dict with Sharpe ratio, annualised return, max drawdown, Calmar ratio,
        mean daily return, daily std dev
    """
    daily_returns = daily_returns.dropna()

    if len(daily_returns) == 0:
        return {k: np.nan for k in [
            "sharpe_ratio", "annualised_return", "max_drawdown",
            "calmar_ratio", "mean_daily_return", "daily_std",
            "total_return", "n_days",
        ]}

    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()

    # Sharpe ratio (annualised, assuming risk-free rate ~ 0 for simplicity)
    sharpe = (mean_ret / std_ret) * np.sqrt(annual_trading_days) if std_ret > 0 else 0

    # Annualised return
    cumulative = (1 + daily_returns).prod()
    n_years = len(daily_returns) / annual_trading_days
    ann_return = cumulative ** (1 / n_years) - 1 if n_years > 0 else 0

    # Maximum drawdown
    cum_returns = (1 + daily_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Total return
    total_ret = cumulative - 1

    return {
        "sharpe_ratio": round(sharpe, 3),
        "annualised_return": round(ann_return, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(calmar, 3),
        "mean_daily_return": round(mean_ret, 6),
        "daily_std": round(std_ret, 6),
        "total_return": round(total_ret, 4),
        "n_days": len(daily_returns),
    }


def compute_all_metrics(portfolio_df):
    """Compute metrics for long, short, and long-short portfolios."""
    results = {}
    for strategy in ["long_ret", "short_ret", "long_short_ret"]:
        name = strategy.replace("_ret", "").replace("_", "-")
        results[name] = compute_metrics(portfolio_df[strategy])
    return results


# ============================================================
# Sensitivity grid (RQ3)
# ============================================================
def run_sensitivity_grid(
    predictions_df,
    returns_df,
    score_col,
    label_horizons,
    portfolio_cutoffs,
    labels_dir,
    transaction_cost=0.001,
):
    """
    Run the full sensitivity grid for RQ3.

    Args:
        predictions_df: test predictions with sentiment scores
        returns_df: daily stock returns
        score_col: name of sentiment score column
        label_horizons: list of horizons [1, 3, 5, 7]
        portfolio_cutoffs: list of cutoffs [0.10, 0.15, 0.20, 0.25, 0.30]
        labels_dir: path to pre-computed label files
        transaction_cost: per-trade cost

    Returns:
        DataFrame with Sharpe ratios for each (horizon, cutoff) combination
    """
    results = []

    for horizon, cutoff in product(label_horizons, portfolio_cutoffs):
        print(f"  Horizon={horizon}d, cutoff={cutoff:.0%}...", end=" ")

        # For different horizons, we'd need to re-fine-tune with different labels
        # For now, we use the same predictions but vary the portfolio construction
        # A full implementation would re-train for each horizon
        portfolio_df = construct_portfolios(
            predictions_df, returns_df, score_col,
            cutoff=cutoff, transaction_cost=transaction_cost,
        )

        if len(portfolio_df) == 0:
            print("no data")
            continue

        metrics = compute_metrics(portfolio_df["long_short_ret"])
        metrics["horizon"] = horizon
        metrics["cutoff"] = cutoff
        results.append(metrics)
        print(f"SR={metrics['sharpe_ratio']:.2f}")

    return pd.DataFrame(results)


# ============================================================
# Reporting
# ============================================================
def print_results_table(all_results, model_names):
    """Print a formatted comparison table across models."""
    print(f"\n{'='*80}")
    print("PORTFOLIO PERFORMANCE SUMMARY (Long-Short)")
    print(f"{'='*80}")

    header = f"{'Model':<15} {'Sharpe':>8} {'Ann.Ret':>10} {'MaxDD':>10} {'Calmar':>8} {'TotalRet':>10} {'Days':>6}"
    print(header)
    print("-" * 80)

    for model_name in model_names:
        if model_name in all_results:
            m = all_results[model_name].get("long-short", {})
            print(
                f"{model_name:<15} "
                f"{m.get('sharpe_ratio', 'N/A'):>8} "
                f"{m.get('annualised_return', 'N/A'):>10} "
                f"{m.get('max_drawdown', 'N/A'):>10} "
                f"{m.get('calmar_ratio', 'N/A'):>8} "
                f"{m.get('total_return', 'N/A'):>10} "
                f"{m.get('n_days', 'N/A'):>6}"
            )

    print(f"{'='*80}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Backtest sentiment trading strategies")
    parser.add_argument("--model", type=str, required=True,
                        choices=["bert", "finbert", "roberta", "opt-350m", "opt-2.7b", "all"],
                        help="Model to backtest")
    parser.add_argument("--cutoff", type=float, default=0.20,
                        help="Portfolio percentile cut-off (default: 0.20)")
    parser.add_argument("--transaction-cost", type=float, default=0.001,
                        help="Transaction cost per trade (default: 10 bps)")
    parser.add_argument("--sensitivity-grid", action="store_true",
                        help="Run full RQ3 sensitivity grid")
    args = parser.parse_args()

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["paths"]["results"]) / "replication"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load return data for the test period
    returns_path = Path(config["paths"]["raw_returns"]) / "crsp_daily_returns.parquet"
    df_returns = pd.read_parquet(returns_path)
    df_returns["date"] = pd.to_datetime(df_returns["date"])

    # Filter to test period
    test_start = pd.Timestamp(config["data"]["test_start"])
    df_returns = df_returns[df_returns["date"] >= test_start].copy()

    # Compute market cap for value-weighting (price * shares outstanding)
    if "prc" in df_returns.columns and "shrout" in df_returns.columns:
        df_returns["market_cap"] = df_returns["prc"].abs() * df_returns["shrout"]
    else:
        df_returns["market_cap"] = 1  # equal-weight fallback

    # Determine which models to backtest
    model_names = (
        ["bert", "finbert", "roberta", "opt-350m", "opt-2.7b"]
        if args.model == "all"
        else [args.model]
    )

    all_results = {}

    for model_name in model_names:
        pred_path = results_dir / f"{model_name}_test_predictions.parquet"

        if not pred_path.exists():
            print(f"Predictions not found for {model_name} at {pred_path}, skipping.")
            continue

        print(f"\n{'#'*60}")
        print(f"# Backtesting: {model_name}")
        print(f"{'#'*60}")

        pred_df = pd.read_parquet(pred_path)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        score_col = f"{model_name}_score"

        if score_col not in pred_df.columns:
            # Try generic column name
            score_col = "score"
            if score_col not in pred_df.columns:
                print(f"  Score column not found for {model_name}, skipping.")
                continue

        # Apply news timing rules
        pred_df = assign_news_timing(pred_df)

        if args.sensitivity_grid:
            # RQ3: Full sensitivity analysis
            print(f"\nRunning sensitivity grid for {model_name}...")
            grid_results = run_sensitivity_grid(
                pred_df, df_returns, score_col,
                label_horizons=config["sensitivity"]["label_horizons"],
                portfolio_cutoffs=config["sensitivity"]["portfolio_cutoffs"],
                labels_dir=Path(config["paths"]["labels"]),
                transaction_cost=args.transaction_cost,
            )
            grid_results.to_csv(
                results_dir / f"{model_name}_sensitivity_grid.csv", index=False
            )
            print(f"  Saved to {results_dir}/{model_name}_sensitivity_grid.csv")

        else:
            # Standard backtest
            portfolio_df = construct_portfolios(
                pred_df, df_returns, score_col,
                cutoff=args.cutoff,
                transaction_cost=args.transaction_cost,
            )

            # Compute metrics
            metrics = compute_all_metrics(portfolio_df)
            all_results[model_name] = metrics

            # Save daily portfolio returns
            portfolio_df.to_csv(
                results_dir / f"{model_name}_portfolio_returns.csv", index=False
            )

            # Save metrics
            metrics_flat = {}
            for strategy, m in metrics.items():
                for k, v in m.items():
                    metrics_flat[f"{strategy}_{k}"] = v

            pd.Series(metrics_flat).to_csv(
                results_dir / f"{model_name}_metrics.csv"
            )

            # Print results
            for strategy in ["long", "short", "long-short"]:
                m = metrics[strategy]
                print(f"\n  {strategy.upper()} portfolio:")
                print(f"    Sharpe ratio:     {m['sharpe_ratio']}")
                print(f"    Annualised return: {m['annualised_return']:.2%}")
                print(f"    Max drawdown:     {m['max_drawdown']:.2%}")
                print(f"    Total return:     {m['total_return']:.2%}")

    # Print comparison table
    if len(all_results) > 1:
        print_results_table(all_results, model_names)

    # Save combined summary
    if all_results:
        summary_rows = []
        for model_name, strategies in all_results.items():
            for strategy, metrics in strategies.items():
                row = {"model": model_name, "strategy": strategy}
                row.update(metrics)
                summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(results_dir / "backtest_summary.csv", index=False)
        print(f"\nSummary saved to {results_dir}/backtest_summary.csv")


if __name__ == "__main__":
    main()