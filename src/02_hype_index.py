"""
Phase 2: Build the hype index for each player.

Three components (all normalised 0-1 before combining):
  1. Google Trends — UK search interest, weekly → aggregated monthly
  2. Guardian API  — article count per month (sentiment added in Phase 3)
  3. Wikipedia     — page views per month

Output: data/raw/hype_raw_{player_slug}.csv per player
        data/processed/hype_index.csv  (all players, player x month)

BEFORE RUNNING:
  - Set GUARDIAN_API_KEY below (register free at open-platform.theguardian.com)
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from datetime import datetime, timedelta

# ── CONFIG ────────────────────────────────────────────────────────────────────
GUARDIAN_API_KEY = os.environ.get("GUARDIAN_API_KEY", "YOUR_KEY_HERE")
START_DATE = "2010-01-01"
END_DATE   = "2024-06-30"

# Player list — will be replaced by output of Phase 1
# Format: (search_name, guardian_query, wikipedia_title, nationality_group)
# nationality_group: "british" or "foreign"
PLAYERS = [
    # Tier 1 — High hype, delivered
    ("Bukayo Saka",         "bukayo saka",         "Bukayo_Saka",         "british"),
    ("Jude Bellingham",     "jude bellingham",     "Jude_Bellingham",     "british"),
    ("Cole Palmer",         "cole palmer",         "Cole_Palmer",         "british"),
    ("Phil Foden",          "phil foden",          "Phil_Foden",          "british"),
    ("Michael Olise",       "michael olise",       "Michael_Olise",       "foreign"),
    ("Florian Wirtz",       "florian wirtz",       "Florian_Wirtz",       "foreign"),
    ("Pedri",               "pedri barcelona",     "Pedri",               "foreign"),
    ("Dani Olmo",           "dani olmo",           "Dani_Olmo",           "foreign"),
    # Tier 2 — High hype, inconsistent
    ("Kobbie Mainoo",       "kobbie mainoo",       "Kobbie_Mainoo",       "british"),
    ("Marcus Rashford",     "marcus rashford",     "Marcus_Rashford",     "british"),
    ("Jadon Sancho",        "jadon sancho",        "Jadon_Sancho",        "british"),
    ("Gavi",                "gavi barcelona",      "Gavi_(footballer)",   "foreign"),
    ("Leroy Sané",          "leroy sane",          "Leroy_Sané",          "foreign"),
    ("Ferran Torres",       "ferran torres",       "Ferran_Torres",       "foreign"),
    # Tier 3 — High hype, underdelivered
    ("Dele Alli",           "dele alli",           "Dele_Alli",           "british"),
    ("Jack Grealish",       "jack grealish",       "Jack_Grealish",       "british"),
    ("Callum Hudson-Odoi",  "callum hudson odoi",  "Callum_Hudson-Odoi",  "british"),
    ("Ross Barkley",        "ross barkley",        "Ross_Barkley",        "british"),
    ("Renato Sanches",      "renato sanches",      "Renato_Sanches",      "foreign"),
    ("Thomas Lemar",        "thomas lemar",        "Thomas_Lemar",        "foreign"),
    ("Hakim Ziyech",        "hakim ziyech",        "Hakim_Ziyech",        "foreign"),
    ("João Félix",          "joao felix",          "João_Félix",          "foreign"),
]

# ── HELPER: slugify name for filenames ────────────────────────────────────────
def slug(name):
    return name.lower().replace(" ", "_").replace("-", "_").replace("é", "e").replace("ã", "a")

# ── 1. GOOGLE TRENDS ─────────────────────────────────────────────────────────
def get_google_trends(player_name, start="2010-01-01", end="2024-06-30"):
    """Pull weekly UK search interest for a player. Returns monthly mean."""
    pytrends = TrendReq(hl="en-GB", tz=0)
    # Google Trends max timeframe per call: ~5 years; chunk it
    chunks = []
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    chunk_start = start_dt

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=365*4), end_dt)
        tf = f"{chunk_start.strftime('%Y-%m-%d')} {chunk_end.strftime('%Y-%m-%d')}"
        try:
            pytrends.build_payload([player_name], cat=20, timeframe=tf, geo="GB")
            df = pytrends.interest_over_time()
            if not df.empty:
                df = df[[player_name]].rename(columns={player_name: "trends_score"})
                chunks.append(df)
            time.sleep(1.5)  # be polite to Google
        except Exception as e:
            print(f"  Trends chunk error for {player_name}: {e}")
        chunk_start = chunk_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame()

    trends = pd.concat(chunks)
    trends.index = pd.to_datetime(trends.index)
    # Aggregate to monthly mean
    monthly = trends.resample("MS").mean().reset_index()
    monthly.columns = ["month", "trends_score"]
    monthly["player"] = player_name
    return monthly

# ── 2. GUARDIAN API ───────────────────────────────────────────────────────────
def get_guardian_articles(query, start="2010-01-01", end="2024-06-30", api_key=GUARDIAN_API_KEY):
    """Pull article count per month from The Guardian. Returns monthly counts."""
    if api_key == "YOUR_KEY_HERE":
        print("  WARNING: No Guardian API key set. Skipping Guardian pull.")
        return pd.DataFrame()

    base_url = "https://content.guardianapis.com/search"
    records = []
    page = 1

    while True:
        params = {
            "q": query,
            "section": "football",
            "from-date": start,
            "to-date": end,
            "page-size": 200,
            "page": page,
            "api-key": api_key,
            "show-fields": "headline,bodyText",
            "order-by": "oldest",
        }
        resp = requests.get(base_url, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"  Guardian error {resp.status_code} for '{query}'")
            break

        data = resp.json()["response"]
        results = data.get("results", [])
        if not results:
            break

        for r in results:
            records.append({
                "date": r["webPublicationDate"][:10],
                "headline": r.get("fields", {}).get("headline", ""),
                "body": r.get("fields", {}).get("bodyText", "")[:500],
            })

        if page >= data.get("pages", 1):
            break
        page += 1
        time.sleep(0.2)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    # Monthly article count
    monthly = df.groupby("month").agg(
        guardian_count=("headline", "count"),
        headlines=("headline", lambda x: " | ".join(x.dropna().tolist()[:5]))
    ).reset_index()
    return monthly, df  # return both: monthly agg + raw (for NLP later)

# ── 3. WIKIPEDIA PAGE VIEWS ───────────────────────────────────────────────────
def get_wikipedia_views(wiki_title, start="2015-07-01", end="2024-06-30"):
    """
    Pull monthly Wikipedia page views.
    Note: Wikimedia API only goes back to July 2015.
    """
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/all-agents/{wiki_title}/monthly/"
        f"{start.replace('-', '')[:8]}/{end.replace('-', '')[:8]}"
    )
    headers = {"User-Agent": "FootballHypeResearch/1.0 (academic research)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"  Wikipedia error {resp.status_code} for {wiki_title}")
            return pd.DataFrame()
        items = resp.json().get("items", [])
        records = [
            {"month": pd.to_datetime(i["timestamp"][:6], format="%Y%m"),
             "wiki_views": i["views"]}
            for i in items
        ]
        return pd.DataFrame(records)
    except Exception as e:
        print(f"  Wikipedia exception for {wiki_title}: {e}")
        return pd.DataFrame()

# ── MAIN: BUILD HYPE INDEX ────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data/raw/guardian_articles", exist_ok=True)
    all_monthly = []

    for (search_name, guardian_q, wiki_title, nat_group) in PLAYERS:
        print(f"\n{'='*60}")
        print(f"Processing: {search_name} ({nat_group})")
        s = slug(search_name)

        # 1. Google Trends
        print("  Pulling Google Trends...")
        trends_df = get_google_trends(search_name)
        if not trends_df.empty:
            trends_df.to_csv(f"data/raw/trends_{s}.csv", index=False)
            print(f"  Trends: {len(trends_df)} monthly rows")

        # 2. Guardian
        print("  Pulling Guardian articles...")
        result = get_guardian_articles(guardian_q)
        if result and not isinstance(result, pd.DataFrame):
            guardian_monthly, guardian_raw = result
            guardian_raw.to_csv(f"data/raw/guardian_articles/{s}_articles.csv", index=False)
            print(f"  Guardian: {len(guardian_raw)} articles → {len(guardian_monthly)} monthly rows")
        else:
            guardian_monthly = pd.DataFrame()

        # 3. Wikipedia
        print("  Pulling Wikipedia views...")
        wiki_df = get_wikipedia_views(wiki_title)
        if not wiki_df.empty:
            print(f"  Wikipedia: {len(wiki_df)} monthly rows")

        # Merge three sources on month
        # Build a common monthly spine 2010-01 to 2024-06
        months = pd.date_range(start="2010-01-01", end="2024-06-01", freq="MS")
        monthly = pd.DataFrame({"month": months})
        monthly["player"] = search_name
        monthly["nationality_group"] = nat_group

        if not trends_df.empty:
            monthly = monthly.merge(trends_df[["month", "trends_score"]], on="month", how="left")
        else:
            monthly["trends_score"] = np.nan

        if not guardian_monthly.empty:
            monthly = monthly.merge(guardian_monthly[["month", "guardian_count"]], on="month", how="left")
        else:
            monthly["guardian_count"] = np.nan

        if not wiki_df.empty:
            monthly = monthly.merge(wiki_df, on="month", how="left")
        else:
            monthly["wiki_views"] = np.nan

        all_monthly.append(monthly)
        print(f"  Monthly panel rows: {len(monthly)}")

    # Combine all players
    panel = pd.concat(all_monthly, ignore_index=True)

    # Normalise each component 0-1 within player (player-relative hype)
    for col in ["trends_score", "guardian_count", "wiki_views"]:
        panel[col] = panel.groupby("player")[col].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )

    # Composite hype index: mean of available components
    panel["hype_index"] = panel[["trends_score", "guardian_count", "wiki_views"]].mean(axis=1, skipna=True)

    panel.to_csv("data/processed/hype_index.csv", index=False)
    print(f"\n{'='*60}")
    print(f"DONE. Saved: data/processed/hype_index.csv")
    print(f"Total rows: {len(panel):,} | Players: {panel['player'].nunique()}")
