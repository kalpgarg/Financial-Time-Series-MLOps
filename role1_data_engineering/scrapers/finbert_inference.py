"""
FinBERT sentiment inference for financial headlines.

Loads the ProsusAI/finbert transformer model and runs batched inference
to produce sentiment probabilities (positive, negative, neutral),
a composite sentiment score, and CLS token embeddings per headline.

Usage:
    python -m role1_data_engineering.scrapers.finbert_inference \
        --input data/headlines.csv \
        --output data/headlines_enriched.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"


def load_finbert(device=None):
    """Load FinBERT tokenizer and model.

    Args:
        device: torch device string or None (auto-detect GPU/CPU).

    Returns:
        Tuple of (tokenizer, model, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    print(f"FinBERT loaded on {device}")
    return tokenizer, model, device


def run_finbert_batch(texts, tokenizer, model, device, max_length=128):
    """Run FinBERT inference on a batch of texts.

    Args:
        texts: List of headline strings.
        tokenizer: FinBERT tokenizer.
        model: FinBERT model.
        device: torch device.
        max_length: Maximum token length.

    Returns:
        Tuple of (probs: ndarray [N, 3], embeddings: ndarray [N, 768]).
    """
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    probs = softmax(outputs.logits.cpu().numpy(), axis=1)
    embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()

    return probs, embeddings


def process_headlines(news_df, batch_size=32):
    """Run FinBERT inference over all headlines in a DataFrame.

    Adds columns: positive_prob, negative_prob, neutral_prob,
    sentiment_score, sentiment_label, embedding.

    Args:
        news_df: DataFrame with 'headline' column.
        batch_size: Inference batch size.

    Returns:
        Enriched DataFrame (same rows, extra columns).
    """
    tokenizer, model, device = load_finbert()

    label_map = model.config.id2label

    all_probs = []
    all_embs = []

    for i in range(0, len(news_df), batch_size):
        batch_text = (
            news_df["headline"].iloc[i : i + batch_size].astype(str).tolist()
        )
        probs, embs = run_finbert_batch(batch_text, tokenizer, model, device)
        all_probs.append(probs)
        all_embs.append(embs)
        print(f"Processed {i + len(batch_text)}/{len(news_df)}")

    all_probs = np.vstack(all_probs)
    all_embs = np.vstack(all_embs)

    # Attach sentiment probabilities using model's label ordering
    for idx, label in label_map.items():
        news_df[f"{label}_prob"] = all_probs[:, idx]

    # Composite sentiment score
    news_df["sentiment_score"] = (
        news_df["positive_prob"] - news_df["negative_prob"]
    ) * (1 - news_df["neutral_prob"])

    # Discrete label
    news_df["sentiment_label"] = np.where(
        news_df["sentiment_score"] > 0.1,
        "positive",
        np.where(news_df["sentiment_score"] < -0.1, "negative", "neutral"),
    )

    # Raw embeddings (stored as list of arrays)
    news_df["embedding"] = list(all_embs)

    return news_df


def main():
    parser = argparse.ArgumentParser(description="FinBERT headline inference")
    parser.add_argument(
        "--input",
        default="data/headlines.csv",
        help="Path to raw headlines CSV",
    )
    parser.add_argument(
        "--output",
        default="data/headlines_enriched.csv",
        help="Path to save enriched CSV",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / args.input
    output_path = project_root / args.output

    print(f"Reading headlines from {input_path}")
    news_df = pd.read_csv(input_path)
    news_df["published_at"] = pd.to_datetime(news_df["published_at"], utc=True)
    news_df = news_df.sort_values(["symbol", "published_at"])

    # Create date column (market day: subtract 9h15m for IST market open)
    news_df["news_date"] = (
        news_df["published_at"] - pd.Timedelta(hours=9, minutes=15)
    ).dt.normalize()

    news_df = process_headlines(news_df, batch_size=args.batch_size)

    # Save without embedding column (too large for CSV; PCA step reads raw)
    save_df = news_df.drop(columns=["embedding"])
    os.makedirs(output_path.parent, exist_ok=True)
    save_df.to_csv(output_path, index=False)
    print(f"Enriched headlines saved to {output_path}")

    # Save embeddings separately as numpy array
    emb_path = output_path.with_suffix(".embeddings.npy")
    np.save(emb_path, np.vstack(news_df["embedding"].values))
    print(f"Embeddings saved to {emb_path}")


if __name__ == "__main__":
    main()
