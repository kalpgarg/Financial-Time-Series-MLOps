"""
Sentiment feature engineering pipeline.

Takes FinBERT-enriched headlines and produces per-stock-date sentiment features:
  - PCA-reduced embeddings (768 → 50 dims)
  - 7-day rolling aggregation with exponential recency decay
  - Weighted sentiment stats, article counts, sentiment trend

Usage:
    python -m role2_ml_modeling.features.sentiment_features \
        --headlines data/headlines_enriched.csv \
        --embeddings data/headlines_enriched.embeddings.npy \
        --ohlcv data/merged_features.csv \
        --output data/news_features.csv \
        --pca-output models/finbert_pca.pkl
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def fit_pca(embeddings, n_components=50, random_state=42):
    """Fit PCA on FinBERT CLS embeddings.

    Args:
        embeddings: ndarray of shape (N, 768).
        n_components: Number of PCA components.
        random_state: Random seed.

    Returns:
        Tuple of (fitted PCA object, reduced embeddings ndarray [N, n_components]).
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    print(
        f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f} "
        f"({n_components} components)"
    )
    return pca, reduced


def aggregate_sentiment_features(
    news_df,
    stock_dates_df,
    pca_components=50,
    window_days=7,
    decay_lambda=0.5,
):
    """Aggregate headline-level sentiment into per-stock-date features.

    Uses a rolling window of `window_days` calendar days with exponential
    recency weighting (λ = decay_lambda).

    Args:
        news_df: DataFrame with columns: symbol, published_at, news_date,
            sentiment_score, sentiment_label, positive_prob, negative_prob,
            neutral_prob, embedding (list/array of PCA-reduced values).
        stock_dates_df: DataFrame with columns: symbol, date (unique stock-dates).
        pca_components: Number of embedding dimensions.
        window_days: Rolling window size in calendar days.
        decay_lambda: Exponential decay rate.

    Returns:
        DataFrame with one row per (symbol, date) containing aggregated
        sentiment features and PCA embedding columns (emb_0..emb_N).
    """
    final_rows = []

    stock_dates = (
        stock_dates_df[["symbol", "date"]]
        .drop_duplicates()
        .sort_values(["symbol", "date"])
    )

    for symbol, stock_group in stock_dates.groupby("symbol"):
        news_group = news_df[news_df["symbol"] == symbol].copy()
        news_group = news_group.sort_values("published_at")

        for current_date in stock_group["date"]:
            start_date = current_date - pd.Timedelta(days=window_days - 1)

            window = news_group[
                (news_group["news_date"] >= start_date)
                & (news_group["news_date"] <= current_date)
            ].copy()

            # No news in window → zero features
            if len(window) == 0:
                row = _zero_row(symbol, current_date, pca_components)
                final_rows.append(row)
                continue

            # Recency weights
            latest_time = window["published_at"].max()
            window["age_days"] = (
                latest_time - window["published_at"]
            ).dt.total_seconds() / 86400
            window["weight"] = np.exp(-decay_lambda * window["age_days"])
            window["weight"] /= window["weight"].sum()
            weights = window["weight"].values

            # Weighted sentiment aggregates
            weighted_sentiment = np.average(
                window["sentiment_score"], weights=weights
            )
            avg_positive = np.average(
                window["positive_prob"], weights=weights
            )
            avg_negative = np.average(
                window["negative_prob"], weights=weights
            )
            avg_neutral = np.average(
                window["neutral_prob"], weights=weights
            )

            # Weighted embeddings
            emb = np.vstack(window["embedding"])
            daily_embedding = np.average(emb, axis=0, weights=weights)

            # Sentiment trend (last 3 vs first 3 articles)
            sorted_scores = (
                window.sort_values("published_at")["sentiment_score"]
            )
            sentiment_trend = (
                sorted_scores.tail(3).mean() - sorted_scores.head(3).mean()
            )

            row = {
                "symbol": symbol,
                "date": current_date,
                "article_count": len(window),
                "weighted_sentiment": weighted_sentiment,
                "sentiment_std": window["sentiment_score"].std(),
                "sentiment_max": window["sentiment_score"].max(),
                "sentiment_min": window["sentiment_score"].min(),
                "positive_articles": (
                    window["sentiment_label"] == "positive"
                ).sum(),
                "negative_articles": (
                    window["sentiment_label"] == "negative"
                ).sum(),
                "neutral_articles": (
                    window["sentiment_label"] == "neutral"
                ).sum(),
                "avg_positive_prob": avg_positive,
                "avg_negative_prob": avg_negative,
                "avg_neutral_prob": avg_neutral,
                "sentiment_trend": sentiment_trend,
            }
            for i in range(pca_components):
                row[f"emb_{i}"] = daily_embedding[i]

            final_rows.append(row)

    return pd.DataFrame(final_rows)


def _zero_row(symbol, date, pca_components):
    """Create a zero-filled row for stock-dates with no news."""
    row = {
        "symbol": symbol,
        "date": date,
        "article_count": 0,
        "weighted_sentiment": 0,
        "sentiment_std": 0,
        "sentiment_max": 0,
        "sentiment_min": 0,
        "positive_articles": 0,
        "negative_articles": 0,
        "neutral_articles": 0,
        "avg_positive_prob": 0,
        "avg_negative_prob": 0,
        "avg_neutral_prob": 0,
        "sentiment_trend": 0,
    }
    for i in range(pca_components):
        row[f"emb_{i}"] = 0
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment feature aggregation"
    )
    parser.add_argument(
        "--headlines",
        default="data/headlines_enriched.csv",
        help="Enriched headlines CSV",
    )
    parser.add_argument(
        "--embeddings",
        default="data/headlines_enriched.embeddings.npy",
        help="FinBERT embeddings .npy file",
    )
    parser.add_argument(
        "--ohlcv",
        default="data/merged_ohlc_15min.csv",
        help="15-min OHLCV CSV (for stock dates)",
    )
    parser.add_argument(
        "--output",
        default="data/news_features.csv",
        help="Output sentiment features CSV",
    )
    parser.add_argument(
        "--pca-output",
        default="models/finbert_pca.pkl",
        help="Output PCA model path",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    # Load enriched headlines
    headlines_path = project_root / args.headlines
    print(f"Reading enriched headlines from {headlines_path}")
    news_df = pd.read_csv(headlines_path)
    news_df["published_at"] = (
        pd.to_datetime(news_df["published_at"], utc=True)
        .dt.tz_localize(None)
    )
    news_df["news_date"] = news_df["published_at"].dt.normalize()

    # Load raw embeddings and fit PCA
    emb_path = project_root / args.embeddings
    print(f"Loading embeddings from {emb_path}")
    all_embs = np.load(emb_path)

    pca, reduced_embs = fit_pca(all_embs, n_components=args.pca_components)

    pca_path = project_root / args.pca_output
    os.makedirs(pca_path.parent, exist_ok=True)
    joblib.dump(pca, pca_path)
    print(f"PCA model saved to {pca_path}")

    # Attach reduced embeddings to news_df
    news_df["embedding"] = list(reduced_embs)

    # Load OHLCV for stock dates
    ohlcv_path = project_root / args.ohlcv
    print(f"Reading OHLCV from {ohlcv_path}")
    ohlcv = pd.read_csv(ohlcv_path)
    ohlcv["datetime"] = pd.to_datetime(ohlcv["datetime"])
    ohlcv["date"] = ohlcv["datetime"].dt.normalize()

    stock_dates = (
        ohlcv[["symbol", "date"]].drop_duplicates().sort_values(["symbol", "date"])
    )

    # Aggregate
    print("Aggregating sentiment features...")
    features_df = aggregate_sentiment_features(
        news_df, stock_dates, pca_components=args.pca_components
    )

    output_path = project_root / args.output
    os.makedirs(output_path.parent, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"Sentiment features saved to {output_path} ({features_df.shape})")


if __name__ == "__main__":
    main()
