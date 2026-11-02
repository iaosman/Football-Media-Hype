"""
Phase 3: NLP sentiment scoring on Guardian article headlines.

Uses HuggingFace cardiffnlp/twitter-roberta-base-sentiment-latest
Three-class output: positive / neutral / negative

Input:  data/raw/guardian_articles/{player_slug}_articles.csv
Output: data/processed/sentiment_{player_slug}.csv
        data/processed/guardian_monthly_sentiment.csv  (all players)

Run AFTER 02_hype_index.py has pulled Guardian articles.
"""

import os
import glob
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ── LOAD MODEL (downloads once, ~500MB, cached after) ─────────────────────────
print("Loading sentiment model...")
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipe = pipeline(
    "text-classification",
    model=MODEL,
    tokenizer=MODEL,
    max_length=128,
    truncation=True,
    top_k=1,
)
print("Model loaded.")

LABEL_MAP = {
    "positive": 1.0,
    "neutral":  0.0,
    "negative": -1.0,
}

def score_texts(texts, batch_size=32):
    """Score a list of texts. Returns list of floats in [-1, 1]."""
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results = sentiment_pipe(batch)
        for r in results:
            label = r[0]["label"].lower()
            scores.append(LABEL_MAP.get(label, 0.0))
    return scores

# ── PROCESS EACH PLAYER ───────────────────────────────────────────────────────
article_files = glob.glob("data/raw/guardian_articles/*_articles.csv")
if not article_files:
    print("No article files found. Run 02_hype_index.py first.")
    exit(0)

all_monthly_sentiment = []

for fpath in sorted(article_files):
    player_slug = os.path.basename(fpath).replace("_articles.csv", "")
    print(f"\nScoring: {player_slug}")

    df = pd.read_csv(fpath)
    if df.empty or "headline" not in df.columns:
        print("  Empty or missing headline column — skipping")
        continue

    # Score headlines (faster than full body, still meaningful)
    headlines = df["headline"].fillna("").tolist()
    df["sentiment_score"] = score_texts(headlines)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # Save article-level with sentiment
    df.to_csv(f"data/processed/sentiment_{player_slug}.csv", index=False)

    # Aggregate to monthly
    monthly = df.groupby("month").agg(
        article_count=("headline", "count"),
        mean_sentiment=("sentiment_score", "mean"),
        negative_share=("sentiment_score", lambda x: (x == -1.0).mean()),
        positive_share=("sentiment_score", lambda x: (x ==  1.0).mean()),
    ).reset_index()
    monthly["player_slug"] = player_slug
    all_monthly_sentiment.append(monthly)
    print(f"  Articles: {len(df)} | Monthly rows: {len(monthly)}")
    print(f"  Mean sentiment: {df['sentiment_score'].mean():.3f}")

if all_monthly_sentiment:
    combined = pd.concat(all_monthly_sentiment, ignore_index=True)
    combined.to_csv("data/processed/guardian_monthly_sentiment.csv", index=False)
    print(f"\nSaved: data/processed/guardian_monthly_sentiment.csv")
    print(f"Total monthly rows: {len(combined):,}")
