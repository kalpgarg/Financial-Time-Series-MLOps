# ============================================================
# FINAL PREDICTION CODE
# FINBERT + PCA + NEWS FEATURES + OHLCV FEATURES + XGBOOST
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import torch
import joblib

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from scipy.special import softmax


# ============================================================
# 2. PATHS
# ============================================================

NEWS_PATH = "/content/drive/MyDrive/headlines.csv"

OHLCV_PATH = "/content/drive/MyDrive/merged_ohlc_15min.csv"

PCA_PATH = (
    "/content/drive/MyDrive/finbert_pca.pkl"

)

ENCODER_PATH = (
    "/content/drive/MyDrive/symbol_encoder.pkl"

)

FEATURE_COLUMNS_PATH = (
    "/content/drive/MyDrive/feature_columns.pkl"

)

MODEL_PATH = (
    "/content/drive/MyDrive/xgboost_model.pkl"

)

OUTPUT_PATH = (
    "/content/drive/MyDrive/latest_predictions.csv"

)


# ============================================================
# 3. LOAD SAVED TRAINING OBJECTS
# ============================================================

pca = joblib.load(
    PCA_PATH
)

le = joblib.load(
    ENCODER_PATH
)

feature_columns = joblib.load(
    FEATURE_COLUMNS_PATH
)

xgb_model = joblib.load(
    MODEL_PATH
)

print("Saved objects loaded.")


# ============================================================
# 4. LOAD NEW DATA
# ============================================================

news_df = pd.read_csv(
    NEWS_PATH
)

ohlcv = pd.read_csv(
    OHLCV_PATH
)

print(
    "News shape:",
    news_df.shape
)

print(
    "OHLCV shape:",
    ohlcv.shape
)


# ============================================================
# 5. PREPARE NEWS DATE
# ============================================================

news_df["published_at"] = pd.to_datetime(
    news_df["published_at"],
    utc=True
)

news_df["news_date"] = (
    news_df["published_at"]
    .dt.tz_localize(None)
    .dt.normalize()
)


# ============================================================
# 6. PREPARE OHLCV DATE
# ============================================================

ohlcv["datetime"] = pd.to_datetime(
    ohlcv["datetime"]
)

ohlcv["date"] = (
    ohlcv["datetime"]
    .dt.normalize()
)

ohlcv = ohlcv.sort_values(
    [
        "symbol",
        "date",
        "datetime"
    ]
)


# ============================================================
# 7. GET MAXIMUM OHLCV DATE
# ============================================================

latest_date = (
    ohlcv["date"]
    .max()
)

print(
    "Prediction feature date:",
    latest_date
)


# ============================================================
# 8. SYMBOLS PRESENT ON MAXIMUM DATE
# ============================================================

latest_symbols = (

    ohlcv.loc[
        ohlcv["date"] == latest_date,
        "symbol"
    ]

    .dropna()

    .unique()
)

print(
    "Symbols:",
    len(latest_symbols)
)


# ============================================================
# 9. TAKE ONLY LAST 7 CALENDAR DAYS NEWS
# ============================================================

news_start_date = (
    latest_date
    -
    pd.Timedelta(
        days=6
    )
)

latest_news = news_df[

    (
        news_df["news_date"]
        >= news_start_date
    )

    &

    (
        news_df["news_date"]
        <= latest_date
    )

    &

    (
        news_df["symbol"]
        .isin(latest_symbols)
    )

].copy()


print(
    "News window:",
    news_start_date,
    "to",
    latest_date
)

print(
    "Recent news rows:",
    len(latest_news)
)


# ============================================================
# 10. LOAD FINBERT
# ============================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

print(
    "Using:",
    device
)

MODEL_NAME = "ProsusAI/finbert"

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME
    )
)

model.to(
    device
)

model.eval()


# ============================================================
# 11. FINBERT FUNCTION
# ============================================================

def run_finbert(texts):

    encoded = tokenizer(

        texts,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=128
    )

    encoded = {

        k: v.to(device)

        for k, v
        in encoded.items()
    }

    with torch.no_grad():

        outputs = model(

            **encoded,

            output_hidden_states=True
        )

    probs = softmax(

        outputs.logits
        .cpu()
        .numpy(),

        axis=1
    )

    embeddings = (

        outputs
        .hidden_states[-1]

        [:, 0, :]

        .cpu()

        .numpy()
    )

    return (
        probs,
        embeddings
    )


# ============================================================
# 12. RUN FINBERT ONLY WHEN NEWS EXISTS
# ============================================================

PCA_COMPONENTS = 50


if len(latest_news) > 0:

    batch_size = 32

    all_probs = []

    all_embs = []


    for i in range(

        0,

        len(latest_news),

        batch_size
    ):

        batch_text = (

            latest_news[
                "headline"
            ]

            .iloc[
                i:i+batch_size
            ]

            .astype(str)

            .tolist()
        )

        probs, embs = run_finbert(
            batch_text
        )

        all_probs.append(
            probs
        )

        all_embs.append(
            embs
        )


    # ---------------------------------------------
    # IMPORTANT
    # list -> NumPy arrays
    # ---------------------------------------------

    all_probs = np.vstack(
        all_probs
    )

    all_embs = np.vstack(
        all_embs
    )

    print(
        "FinBERT probabilities:",
        all_probs.shape
    )

    print(
        "FinBERT embeddings:",
        all_embs.shape
    )


    # ---------------------------------------------
    # ATTACH FINBERT PROBABILITIES
    # ---------------------------------------------

    label_map = (
        model.config.id2label
    )


    for idx, label in (
        label_map.items()
    ):

        latest_news[
            f"{label}_prob"
        ] = (

            all_probs[
                :,
                idx
            ]
        )


    # ---------------------------------------------
    # SENTIMENT SCORE
    # ---------------------------------------------

    latest_news[
        "sentiment_score"
    ] = (

        latest_news[
            "positive_prob"
        ]

        -

        latest_news[
            "negative_prob"
        ]

    ) * (

        1

        -

        latest_news[
            "neutral_prob"
        ]

    )


    # ---------------------------------------------
    # SENTIMENT LABEL
    # ---------------------------------------------

    latest_news[
        "sentiment_label"
    ] = np.where(

        latest_news[
            "sentiment_score"
        ] > 0.1,

        "positive",

        np.where(

            latest_news[
                "sentiment_score"
            ] < -0.1,

            "negative",

            "neutral"
        )
    )


    # ---------------------------------------------
    # SAVED PCA
    #
    # Prediction = transform()
    # NOT fit_transform()
    # ---------------------------------------------

    reduced_embs = (
        pca.transform(
            all_embs
        )
    )

    latest_news[
        "embedding"
    ] = list(
        reduced_embs
    )


else:

    print(
        "No news found."
    )

    print(
        "FinBERT skipped."
    )


# ============================================================
# 13. CREATE NEWS FEATURES
# ============================================================

news_rows = []


if len(latest_news) > 0:

    for symbol, window in (
        latest_news.groupby(
            "symbol"
        )
    ):

        window = (

            window

            .copy()

            .sort_values(
                "published_at"
            )
        )


        # -----------------------------------------
        # RECENCY WEIGHTS
        # -----------------------------------------

        latest_time = (

            window[
                "published_at"
            ]

            .max()
        )


        window[
            "age_days"
        ] = (

            latest_time

            -

            window[
                "published_at"
            ]

        ).dt.total_seconds() / 86400


        lambda_decay = 0.5


        window[
            "weight"
        ] = np.exp(

            -lambda_decay

            *

            window[
                "age_days"
            ]
        )


        window[
            "weight"
        ] = (

            window[
                "weight"
            ]

            /

            window[
                "weight"
            ].sum()
        )


        weights = (

            window[
                "weight"
            ]

            .values
        )


        # -----------------------------------------
        # WEIGHTED SENTIMENT
        # -----------------------------------------

        weighted_sentiment = np.average(

            window[
                "sentiment_score"
            ],

            weights=weights
        )


        # -----------------------------------------
        # WEIGHTED PROBABILITIES
        # -----------------------------------------

        avg_positive = np.average(

            window[
                "positive_prob"
            ],

            weights=weights
        )


        avg_negative = np.average(

            window[
                "negative_prob"
            ],

            weights=weights
        )


        avg_neutral = np.average(

            window[
                "neutral_prob"
            ],

            weights=weights
        )


        # -----------------------------------------
        # WEIGHTED EMBEDDING
        # -----------------------------------------

        emb = np.vstack(

            window[
                "embedding"
            ]
        )


        daily_embedding = np.average(

            emb,

            axis=0,

            weights=weights
        )


        # -----------------------------------------
        # SENTIMENT TREND
        # -----------------------------------------

        sentiment_trend = (

            window

            .sort_values(
                "published_at"
            )

            ["sentiment_score"]

            .tail(3)

            .mean()

            -

            window

            .sort_values(
                "published_at"
            )

            ["sentiment_score"]

            .head(3)

            .mean()
        )


        # -----------------------------------------
        # CREATE ROW
        # -----------------------------------------

        row = {

            "symbol":
                symbol,

            "date":
                latest_date,

            "article_count":
                len(window),

            "weighted_sentiment":
                weighted_sentiment,

            "sentiment_std":
                window[
                    "sentiment_score"
                ].std(),

            "sentiment_max":
                window[
                    "sentiment_score"
                ].max(),

            "sentiment_min":
                window[
                    "sentiment_score"
                ].min(),

            "positive_articles":
                (
                    window[
                        "sentiment_label"
                    ]
                    == "positive"
                ).sum(),

            "negative_articles":
                (
                    window[
                        "sentiment_label"
                    ]
                    == "negative"
                ).sum(),

            "neutral_articles":
                (
                    window[
                        "sentiment_label"
                    ]
                    == "neutral"
                ).sum(),

            "avg_positive_prob":
                avg_positive,

            "avg_negative_prob":
                avg_negative,

            "avg_neutral_prob":
                avg_neutral,

            "sentiment_trend":
                sentiment_trend
        }


        # -----------------------------------------
        # ADD PCA FEATURES
        # -----------------------------------------

        for i in range(
            PCA_COMPONENTS
        ):

            row[
                f"emb_{i}"
            ] = (
                daily_embedding[i]
            )


        news_rows.append(
            row
        )


# ============================================================
# 14. NO-NEWS SYMBOLS
# ZERO FEATURES
# ============================================================

if len(latest_news) > 0:

    symbols_with_news = set(

        latest_news[
            "symbol"
        ]

        .unique()
    )

else:

    symbols_with_news = set()


for symbol in latest_symbols:

    if symbol not in symbols_with_news:

        print(
            symbol,
            ": no news -> zero news features"
        )


        row = {

            "symbol":
                symbol,

            "date":
                latest_date,

            "article_count":
                0,

            "weighted_sentiment":
                0,

            "sentiment_std":
                0,

            "sentiment_max":
                0,

            "sentiment_min":
                0,

            "positive_articles":
                0,

            "negative_articles":
                0,

            "neutral_articles":
                0,

            "avg_positive_prob":
                0,

            "avg_negative_prob":
                0,

            "avg_neutral_prob":
                0,

            "sentiment_trend":
                0
        }


        for i in range(
            PCA_COMPONENTS
        ):

            row[
                f"emb_{i}"
            ] = 0


        news_rows.append(
            row
        )


# ============================================================
# 15. FINAL NEWS FEATURES
# ============================================================

latest_news_features = pd.DataFrame(
    news_rows
)

print(
    "Latest news feature shape:",
    latest_news_features.shape
)


# ============================================================
# 16. DAILY OHLCV
#
# IMPORTANT:
# USE HISTORICAL DATA FIRST.
# DO NOT FILTER LATEST DATE YET.
# ============================================================

daily_ohlcv = (

    ohlcv

    .groupby(
        [
            "symbol",
            "date"
        ]
    )

    .agg(

        day_open=(
            "open",
            "first"
        ),

        day_high=(
            "high",
            "max"
        ),

        day_low=(
            "low",
            "min"
        ),

        day_close=(
            "close",
            "last"
        ),

        day_volume=(
            "volume",
            "sum"
        )
    )

    .reset_index()
)


daily_ohlcv = (

    daily_ohlcv

    .sort_values(
        [
            "symbol",
            "date"
        ]
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# 17. GROUP BY STOCK
# ============================================================

g = daily_ohlcv.groupby(
    "symbol"
)


# ============================================================
# 18. DAILY RETURN
# ============================================================

daily_ohlcv[
    "daily_return"
] = (

    g[
        "day_close"
    ]

    .pct_change()
)


# ============================================================
# 19. OPEN-CLOSE %
# ============================================================

daily_ohlcv[
    "open_close_pct"
] = (

    daily_ohlcv[
        "day_close"
    ]

    -

    daily_ohlcv[
        "day_open"
    ]

) / (

    daily_ohlcv[
        "day_open"
    ]
)


# ============================================================
# 20. HIGH-LOW %
# ============================================================

daily_ohlcv[
    "high_low_pct"
] = (

    daily_ohlcv[
        "day_high"
    ]

    -

    daily_ohlcv[
        "day_low"
    ]

) / (

    daily_ohlcv[
        "day_low"
    ]
)


# ============================================================
# 21. RETURN LAGS
# ============================================================

for lag in [
    1,
    2,
    3
]:

    daily_ohlcv[
        f"return_lag_{lag}"
    ] = (

        g[
            "daily_return"
        ]

        .shift(
            lag
        )
    )


# ============================================================
# 22. ROLLING RETURNS
# ============================================================

for w in [
    7,
    14,
    30
]:

    daily_ohlcv[
        f"return_{w}d"
    ] = (

        g[
            "daily_return"
        ]

        .transform(

            lambda x:

            x.rolling(
                w
            ).mean()
        )
    )


# ============================================================
# 23. VOLATILITY
# ============================================================

for w in [
    7,
    14,
    30
]:

    daily_ohlcv[
        f"volatility_{w}d"
    ] = (

        g[
            "daily_return"
        ]

        .transform(

            lambda x:

            x.rolling(
                w
            ).std()
        )
    )


# ============================================================
# 24. MOVING AVERAGE
# ============================================================

for w in [
    7,
    14,
    30
]:

    daily_ohlcv[
        f"price_ma_{w}"
    ] = (

        g[
            "day_close"
        ]

        .transform(

            lambda x:

            x.rolling(
                w
            ).mean()
        )
    )


# ============================================================
# 25. PRICE VS MA
# ============================================================

for w in [
    7,
    14,
    30
]:

    daily_ohlcv[
        f"price_vs_ma{w}"
    ] = (

        daily_ohlcv[
            "day_close"
        ]

        -

        daily_ohlcv[
            f"price_ma_{w}"
        ]

    ) / (

        daily_ohlcv[
            f"price_ma_{w}"
        ]
    )


# ============================================================
# 26. VOLUME FEATURES
# ============================================================

for w in [
    7,
    14,
    30
]:

    daily_ohlcv[
        f"volume_{w}d"
    ] = (

        g[
            "day_volume"
        ]

        .transform(

            lambda x:

            x.rolling(
                w
            ).mean()
        )
    )


# ============================================================
# 27. VOLUME RATIO
# ============================================================

daily_ohlcv[
    "volume_ratio"
] = (

    daily_ohlcv[
        "day_volume"
    ]

    /

    daily_ohlcv[
        "volume_7d"
    ]
)


# ============================================================
# 28. 9:15 FEATURES
#
# MATCHING YOUR CURRENT TRAINING CODE
# ============================================================

open_915 = (

    ohlcv

    .sort_values(
        [
            "symbol",
            "datetime"
        ]
    )

    .groupby(
        [
            "symbol",
            "date"
        ]
    )

    .first()

    .reset_index()
)


open_915 = open_915[
    [
        "symbol",
        "date",
        "open",
        "close"
    ]
]


open_915 = open_915.rename(

    columns={

        "open":
            "open_915",

        "close":
            "close_915"
    }
)


# ============================================================
# IMPORTANT
#
# YOUR CURRENT TRAINING CODE SHIFTS THIS TWICE.
# THIS REPRODUCES IT TO MATCH THE SAVED MODEL.
# ============================================================

open_915[
    "date"
] = (

    open_915[
        "date"
    ]

    -

    pd.Timedelta(
        days=1
    )
)


open_915[
    "date"
] = (

    open_915[
        "date"
    ]

    -

    pd.Timedelta(
        days=1
    )
)


# ============================================================
# 29. MERGE 9:15 FEATURES
# ============================================================

daily_ohlcv = daily_ohlcv.merge(

    open_915,

    on=[
        "symbol",
        "date"
    ],

    how="left"
)


# ============================================================
# 30. OVERNIGHT GAP
# ============================================================

daily_ohlcv[
    "gap_from_prev_close"
] = (

    daily_ohlcv[
        "open_915"
    ]

    -

    daily_ohlcv[
        "day_close"
    ]

) / (

    daily_ohlcv[
        "day_close"
    ]
)


# ============================================================
# 31. FIRST 15-MIN RETURN
# ============================================================

daily_ohlcv[
    "first15_return"
] = (

    daily_ohlcv[
        "close_915"
    ]

    -

    daily_ohlcv[
        "open_915"
    ]

) / (

    daily_ohlcv[
        "open_915"
    ]
)


# ============================================================
# 32. FIRST 15-MIN DIRECTION
# ============================================================

daily_ohlcv[
    "first15_direction"
] = np.sign(

    daily_ohlcv[
        "first15_return"
    ]
)


# ============================================================
# 33. HANDLE INFINITY
# ============================================================

daily_ohlcv = daily_ohlcv.replace(

    [
        np.inf,
        -np.inf
    ],

    np.nan
)


# ============================================================
# 34. NOW TAKE ONLY MAXIMUM DATE
# ============================================================

latest_ohlcv_features = (

    daily_ohlcv[

        daily_ohlcv[
            "date"
        ]

        == latest_date

    ]

    .copy()
)


print(
    "Latest OHLCV feature shape:",
    latest_ohlcv_features.shape
)


# ============================================================
# 35. CHECK OHLCV HISTORY
# ============================================================

history_count = (

    daily_ohlcv

    .groupby(
        "symbol"
    )

    .size()
)


short_history = (

    history_count[
        history_count < 30
    ]
)


if len(short_history) > 0:

    print(
        "\nWARNING: Stocks with fewer than 30 OHLCV days:"
    )

    print(
        short_history
    )


# ============================================================
# 36. MERGE NEWS + OHLCV
# ============================================================

prediction_df = (

    latest_news_features

    .merge(

        latest_ohlcv_features,

        on=[
            "symbol",
            "date"
        ],

        how="inner"
    )
)


print(
    "Merged prediction dataframe:",
    prediction_df.shape
)


# ============================================================
# 37. CHECK UNKNOWN SYMBOLS
# ============================================================

unknown_symbols = [

    symbol

    for symbol in (

        prediction_df[
            "symbol"
        ]

        .unique()
    )

    if symbol not in le.classes_
]


if len(
    unknown_symbols
) > 0:

    raise ValueError(

        "These stocks were not present during training: "

        +

        str(
            unknown_symbols
        )
    )


# ============================================================
# 38. SYMBOL ENCODING
# ============================================================

prediction_df[
    "symbol_encoded"
] = (

    le.transform(

        prediction_df[
            "symbol"
        ]
    )
)


# ============================================================
# 39. HANDLE INF
# ============================================================

prediction_df = prediction_df.replace(

    [
        np.inf,
        -np.inf
    ],

    np.nan
)


# ============================================================
# 40. CHECK REQUIRED FEATURES
# ============================================================

missing_columns = [

    col

    for col in feature_columns

    if col not in prediction_df.columns
]


print(
    "Missing features:",
    missing_columns
)


if len(
    missing_columns
) > 0:

    raise ValueError(

        "Prediction data is missing these model features: "

        +

        str(
            missing_columns
        )
    )


# ============================================================
# 41. CREATE X_NEW
#
# SAME FEATURES
# SAME ORDER
# AS MODEL TRAINING
# ============================================================

X_new = (

    prediction_df[
        feature_columns
    ]

    .copy()
)


print(
    "X_new shape:",
    X_new.shape
)


print(
    "Expected feature count:",
    len(feature_columns)
)


# ============================================================
# 42. CHECK NULLS
# ============================================================

null_counts = (

    X_new

    .isna()

    .sum()
)


null_counts = (

    null_counts[

        null_counts > 0

    ]
)


if len(
    null_counts
) > 0:

    print(
        "\nNaN columns:"
    )

    print(
        null_counts
    )


# ============================================================
# 43. PREDICT
# ============================================================

y_pred = xgb_model.predict(
    X_new
)


y_prob = xgb_model.predict_proba(
    X_new
)


y_pred = np.asarray(
    y_pred
)

y_prob = np.asarray(
    y_prob
)


# ============================================================
# 44. ADD PREDICTION
# ============================================================

prediction_df[
    "Predicted"
] = y_pred


prediction_df[
    "Prob_Negative"
] = (

    y_prob[
        :,
        0
    ]
)


prediction_df[
    "Prob_Neutral"
] = (

    y_prob[
        :,
        1
    ]
)


prediction_df[
    "Prob_Positive"
] = (

    y_prob[
        :,
        2
    ]
)


prediction_df[
    "Confidence"
] = (

    y_prob.max(
        axis=1
    )
)


# ============================================================
# 45. PREDICTION LABEL
# ============================================================

label_mapping = {

    0:
        "Negative",

    1:
        "Neutral",

    2:
        "Positive"
}


prediction_df[
    "Prediction"
] = (

    prediction_df[
        "Predicted"
    ]

    .map(
        label_mapping
    )
)


# ============================================================
# 46. FINAL OUTPUT
# ============================================================

result = (

    prediction_df[
        [
            "symbol",
            "date",
            "article_count",
            "weighted_sentiment",
            "Prediction",
            "Prob_Negative",
            "Prob_Neutral",
            "Prob_Positive",
            "Confidence"
        ]
    ]

    .copy()
)


print(
    "\nFINAL PREDICTIONS"
)

print(
    result
)


# ============================================================
# 47. SAVE OUTPUT TO GOOGLE DRIVE
# ============================================================

result.to_csv(

    OUTPUT_PATH,

    index=False
)


print(
    "\nPrediction saved:"
)

print(
    OUTPUT_PATH
)