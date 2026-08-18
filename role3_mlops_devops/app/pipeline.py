"""The real scoring pipeline: FinBERT -> PCA -> feature engineering -> XGBoost.

This is a faithful port of the team's ``stock_prediction.ipynb`` prediction
code. It is the single scoring path used by both callers:

* ``app/routes.py``   -- one symbol on demand via ``POST /predict``
* ``batch_score.py``  -- all constituents once a day (the Airflow task)

Both hand this module a news DataFrame and an OHLCV DataFrame and get back one
row of prediction per symbol. Keeping it in one place means the API and the
pipeline cannot compute a prediction differently.

Faithfulness over cleanliness
-----------------------------
The feature construction mirrors the training-time notebook exactly, including
its quirks -- most notably the two-day backward shift applied to the 09:15
features (see ``_build_ohlcv_features``). "Improving" any of this would create
train/serve skew against the saved XGBoost model, so it is reproduced verbatim.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.config import (
    ENCODER_PATH,
    FEATURE_COLUMNS_PATH,
    FINBERT_MODEL,
    PCA_PATH,
    XGB_PATH,
)

logger = logging.getLogger(__name__)

# -- constants mirrored from the notebook ----------------------------------
PCA_COMPONENTS = 50
NEWS_WINDOW_DAYS = 6            # last 7 calendar days, inclusive
LAMBDA_DECAY = 0.5             # recency weighting for sentiment
SENTIMENT_POS_THRESHOLD = 0.1
SENTIMENT_NEG_THRESHOLD = -0.1
FINBERT_BATCH_SIZE = 32
FINBERT_MAX_LENGTH = 128

# XGBoost class index -> label. Column order [Negative, Neutral, Positive] is
# fixed by the saved model (classes_ == [0, 1, 2]).
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

_artifacts: Optional[Dict] = None
_finbert: Optional[Dict] = None


# ==========================================================================
# Artifact + model loading (cached)
# ==========================================================================

def load_artifacts() -> Dict:
    """Load and cache the four pickled artifacts."""
    global _artifacts
    if _artifacts is None:
        for path in (PCA_PATH, ENCODER_PATH, FEATURE_COLUMNS_PATH, XGB_PATH):
            if not path.exists():
                raise FileNotFoundError(f"Missing model artifact: {path}")
        _artifacts = {
            "pca": joblib.load(PCA_PATH),
            "encoder": joblib.load(ENCODER_PATH),
            "feature_columns": joblib.load(FEATURE_COLUMNS_PATH),
            "xgb": joblib.load(XGB_PATH),
        }
        logger.info(
            "artifacts loaded (%d features, %d symbols)",
            len(_artifacts["feature_columns"]),
            len(_artifacts["encoder"].classes_),
        )
    return _artifacts


def _load_finbert() -> Dict:
    """Lazily load FinBERT. Torch/transformers are imported here so that tests
    which monkeypatch :func:`embed_headlines` never need them installed."""
    global _finbert
    if _finbert is None:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        model.to(device).eval()
        _finbert = {"tokenizer": tokenizer, "model": model, "device": device}
        logger.info("FinBERT loaded on %s", device)
    return _finbert


def warmup() -> None:
    """Eagerly load everything at boot so the first request pays nothing and a
    missing artifact fails fast."""
    load_artifacts()
    _load_finbert()


def is_ready() -> bool:
    return _artifacts is not None


# ==========================================================================
# FinBERT embedding (monkeypatchable seam for tests)
# ==========================================================================

def embed_headlines(texts: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Run FinBERT over ``texts``.

    Returns ``(probs, embeddings, id2label)`` where ``probs`` is ``(n, 3)`` in
    the model's own class order, ``embeddings`` is the ``(n, 768)`` [CLS]
    hidden state, and ``id2label`` maps column index to sentiment name.

    Tests replace this function wholesale to avoid loading the weights.
    """
    import torch
    from scipy.special import softmax

    fb = _load_finbert()
    tokenizer, model, device = fb["tokenizer"], fb["model"], fb["device"]

    all_probs, all_embs = [], []
    for i in range(0, len(texts), FINBERT_BATCH_SIZE):
        batch = [str(t) for t in texts[i : i + FINBERT_BATCH_SIZE]]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=FINBERT_MAX_LENGTH,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
        all_probs.append(softmax(outputs.logits.cpu().numpy(), axis=1))
        all_embs.append(outputs.hidden_states[-1][:, 0, :].cpu().numpy())

    probs = np.vstack(all_probs)
    embs = np.vstack(all_embs)
    return probs, embs, dict(model.config.id2label)


# ==========================================================================
# News features
# ==========================================================================

def _build_news_features(
    news_df: pd.DataFrame,
    latest_date: pd.Timestamp,
    latest_symbols: np.ndarray,
    embed_fn: Callable,
) -> pd.DataFrame:
    """Recency-weighted sentiment + PCA-reduced embedding features per symbol.

    Symbols with no news in the window get an all-zero feature row, exactly as
    the notebook does, so every symbol survives the later inner merge.
    """
    pca = load_artifacts()["pca"]
    emb_cols = [f"emb_{i}" for i in range(PCA_COMPONENTS)]

    latest_news = pd.DataFrame()
    if not news_df.empty:
        news = news_df.copy()
        news["published_at"] = pd.to_datetime(news["published_at"], utc=True)
        news["news_date"] = news["published_at"].dt.tz_localize(None).dt.normalize()
        window_start = latest_date - pd.Timedelta(days=NEWS_WINDOW_DAYS)
        latest_news = news[
            (news["news_date"] >= window_start)
            & (news["news_date"] <= latest_date)
            & (news["symbol"].isin(latest_symbols))
        ].copy()

    if not latest_news.empty:
        probs, embs, id2label = embed_fn(latest_news["headline"].astype(str).tolist())
        for idx, label in id2label.items():
            latest_news[f"{label}_prob"] = probs[:, idx]

        latest_news["sentiment_score"] = (
            latest_news["positive_prob"] - latest_news["negative_prob"]
        ) * (1 - latest_news["neutral_prob"])
        latest_news["sentiment_label"] = np.where(
            latest_news["sentiment_score"] > SENTIMENT_POS_THRESHOLD,
            "positive",
            np.where(
                latest_news["sentiment_score"] < SENTIMENT_NEG_THRESHOLD,
                "negative",
                "neutral",
            ),
        )
        # transform() at inference -- never fit_transform().
        latest_news["embedding"] = list(pca.transform(embs))

    rows: List[Dict] = []
    symbols_with_news = set()

    if not latest_news.empty:
        for symbol, window in latest_news.groupby("symbol"):
            symbols_with_news.add(symbol)
            window = window.copy().sort_values("published_at")

            latest_time = window["published_at"].max()
            age_days = (latest_time - window["published_at"]).dt.total_seconds() / 86400
            weight = np.exp(-LAMBDA_DECAY * age_days)
            weight = weight / weight.sum()
            weights = weight.values

            emb = np.vstack(window["embedding"])
            daily_embedding = np.average(emb, axis=0, weights=weights)

            sorted_scores = window.sort_values("published_at")["sentiment_score"]
            sentiment_trend = sorted_scores.tail(3).mean() - sorted_scores.head(3).mean()

            row = {
                "symbol": symbol,
                "date": latest_date,
                "article_count": len(window),
                "weighted_sentiment": np.average(
                    window["sentiment_score"], weights=weights
                ),
                "sentiment_std": window["sentiment_score"].std(),
                "sentiment_max": window["sentiment_score"].max(),
                "sentiment_min": window["sentiment_score"].min(),
                "positive_articles": (window["sentiment_label"] == "positive").sum(),
                "negative_articles": (window["sentiment_label"] == "negative").sum(),
                "neutral_articles": (window["sentiment_label"] == "neutral").sum(),
                "avg_positive_prob": np.average(
                    window["positive_prob"], weights=weights
                ),
                "avg_negative_prob": np.average(
                    window["negative_prob"], weights=weights
                ),
                "avg_neutral_prob": np.average(window["neutral_prob"], weights=weights),
                "sentiment_trend": sentiment_trend,
            }
            for i in range(PCA_COMPONENTS):
                row[f"emb_{i}"] = daily_embedding[i]
            rows.append(row)

    # Zero-feature rows for symbols with no news in the window.
    zero_news = {
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
        **{col: 0 for col in emb_cols},
    }
    for symbol in latest_symbols:
        if symbol not in symbols_with_news:
            rows.append({"symbol": symbol, "date": latest_date, **zero_news})

    return pd.DataFrame(rows)


# ==========================================================================
# OHLCV features
# ==========================================================================

def _build_ohlcv_features(
    ohlcv_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Timestamp, np.ndarray]:
    """Daily OHLCV features for the latest available date.

    Rolling windows (7/14/30d) need history, so this consumes the full supplied
    OHLCV frame and only filters to the latest date at the very end.
    """
    ohlcv = ohlcv_df.copy()
    ohlcv["datetime"] = pd.to_datetime(ohlcv["datetime"])
    ohlcv["date"] = ohlcv["datetime"].dt.normalize()
    ohlcv = ohlcv.sort_values(["symbol", "date", "datetime"])

    latest_date = ohlcv["date"].max()
    latest_symbols = (
        ohlcv.loc[ohlcv["date"] == latest_date, "symbol"].dropna().unique()
    )

    daily = (
        ohlcv.groupby(["symbol", "date"])
        .agg(
            day_open=("open", "first"),
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
            day_volume=("volume", "sum"),
        )
        .reset_index()
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    g = daily.groupby("symbol")
    daily["daily_return"] = g["day_close"].pct_change()
    daily["open_close_pct"] = (daily["day_close"] - daily["day_open"]) / daily["day_open"]
    daily["high_low_pct"] = (daily["day_high"] - daily["day_low"]) / daily["day_low"]

    for lag in (1, 2, 3):
        daily[f"return_lag_{lag}"] = g["daily_return"].shift(lag)
    for w in (7, 14, 30):
        daily[f"return_{w}d"] = g["daily_return"].transform(lambda x: x.rolling(w).mean())
    for w in (7, 14, 30):
        daily[f"volatility_{w}d"] = g["daily_return"].transform(lambda x: x.rolling(w).std())
    for w in (7, 14, 30):
        daily[f"price_ma_{w}"] = g["day_close"].transform(lambda x: x.rolling(w).mean())
    for w in (7, 14, 30):
        daily[f"price_vs_ma{w}"] = (
            daily["day_close"] - daily[f"price_ma_{w}"]
        ) / daily[f"price_ma_{w}"]
    for w in (7, 14, 30):
        daily[f"volume_{w}d"] = g["day_volume"].transform(lambda x: x.rolling(w).mean())
    daily["volume_ratio"] = daily["day_volume"] / daily["volume_7d"]

    # 09:15 (first bar of the day) open/close.
    open_915 = (
        ohlcv.sort_values(["symbol", "datetime"])
        .groupby(["symbol", "date"])
        .first()
        .reset_index()[["symbol", "date", "open", "close"]]
        .rename(columns={"open": "open_915", "close": "close_915"})
    )
    # NOTE (faithful to training): the notebook shifts this date back two days
    # before merging. This is reproduced deliberately to match the saved model.
    # A side effect is that the 09:15 features are NaN for the latest date at
    # inference time (the "future" bars they would come from do not exist yet);
    # XGBoost tolerates the NaNs. Flagged for the report -- do not "fix" here.
    open_915["date"] = open_915["date"] - pd.Timedelta(days=1)
    open_915["date"] = open_915["date"] - pd.Timedelta(days=1)

    daily = daily.merge(open_915, on=["symbol", "date"], how="left")
    daily["gap_from_prev_close"] = (daily["open_915"] - daily["day_close"]) / daily["day_close"]
    daily["first15_return"] = (daily["close_915"] - daily["open_915"]) / daily["open_915"]
    daily["first15_direction"] = np.sign(daily["first15_return"])

    daily = daily.replace([np.inf, -np.inf], np.nan)
    latest = daily[daily["date"] == latest_date].copy()
    return latest, latest_date, latest_symbols


# ==========================================================================
# Top-level scoring
# ==========================================================================

def score_frames(
    news_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    embed_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Score every symbol present on the latest OHLCV date.

    Returns one row per symbol with: symbol, date, prediction label, integer
    class, the three class probabilities, confidence, and the two headline news
    features used by the frontend/monitoring.
    """
    art = load_artifacts()
    encoder, feature_columns, xgb = art["encoder"], art["feature_columns"], art["xgb"]
    embed_fn = embed_fn or embed_headlines

    ohlcv_features, latest_date, latest_symbols = _build_ohlcv_features(ohlcv_df)
    if len(latest_symbols) == 0:
        raise ValueError("no OHLCV rows supplied; cannot determine a prediction date")

    news_features = _build_news_features(
        news_df, latest_date, latest_symbols, embed_fn
    )

    prediction_df = news_features.merge(
        ohlcv_features, on=["symbol", "date"], how="inner"
    )
    if prediction_df.empty:
        raise ValueError(
            "no symbols left after merging news and OHLCV features; check inputs"
        )

    unknown = [s for s in prediction_df["symbol"].unique() if s not in encoder.classes_]
    if unknown:
        raise ValueError(f"symbols not seen during training: {unknown}")
    prediction_df["symbol_encoded"] = encoder.transform(prediction_df["symbol"])
    prediction_df = prediction_df.replace([np.inf, -np.inf], np.nan)

    missing = [c for c in feature_columns if c not in prediction_df.columns]
    if missing:
        raise ValueError(f"engineered features missing required columns: {missing}")

    x_new = prediction_df[feature_columns].copy()
    y_pred = np.asarray(xgb.predict(x_new))
    y_prob = np.asarray(xgb.predict_proba(x_new))

    prediction_df["predicted"] = y_pred
    prediction_df["prob_negative"] = y_prob[:, 0]
    prediction_df["prob_neutral"] = y_prob[:, 1]
    prediction_df["prob_positive"] = y_prob[:, 2]
    prediction_df["confidence"] = y_prob.max(axis=1)
    prediction_df["direction"] = prediction_df["predicted"].map(LABEL_MAP)

    return prediction_df[
        [
            "symbol",
            "date",
            "direction",
            "confidence",
            "prob_negative",
            "prob_neutral",
            "prob_positive",
            "article_count",
            "weighted_sentiment",
        ]
    ].reset_index(drop=True)
