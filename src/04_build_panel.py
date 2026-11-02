"""
Phase 4: Merge performance data + hype index into final panel dataset.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("Loading hype index...")
hype = pd.read_csv("data/processed/hype_index.csv", parse_dates=["month"])

print("Loading FBRef performance data...")
perf = pd.read_csv("data/raw/fbref_pl_all_seasons_raw.csv")

print(f"Hype rows: {len(hype):,} | Perf rows: {len(perf):,}")

# Rename FBRef columns to standard names
col_map = {
    "Unnamed: 1_level_0_Player": "player",
    "Unnamed: 2_level_0_Nation": "nation",
    "Unnamed: 3_level_0_Pos": "position",
    "Unnamed: 4_level_0_Squad": "squad",
    "Unnamed: 5_level_0_Age": "age",
    "Playing Time_Min": "minutes",
    "Playing Time_90s": "nineties_played",
    "Performance_Gls": "goals",
    "Performance_Ast": "assists",
}
perf = perf.rename(columns=col_map)

print(f"Key columns: {[c for c in ['player','nation','position','minutes','nineties_played','goals','assists','season'] if c in perf.columns]}")

# Drop aggregate and header rows
if "player" in perf.columns:
    perf = perf[~perf["player"].isin(["Squad Total", "Opponent Total", "Player"])]
    perf = perf[perf["player"].notna()]

# Remove GKs
if "position" in perf.columns:
    perf = perf[~perf["position"].str.contains("GK", na=False)]

# Numeric conversions
if "minutes" in perf.columns:
    perf["minutes"] = pd.to_numeric(
        perf["minutes"].astype(str).str.replace(",", ""), errors="coerce"
    )

if "nineties_played" in perf.columns:
    perf["nineties_played"] = pd.to_numeric(perf["nineties_played"], errors="coerce")
    for col in ["goals", "assists"]:
        if col in perf.columns:
            perf[col] = pd.to_numeric(perf[col], errors="coerce")
            perf[f"{col}_per90"] = perf[col] / perf["nineties_played"].replace(0, np.nan)

# British flag (FBRef format: "eng ENG" — take last word)
BRITISH_CODES = {"eng", "sco", "wal", "nir"}
if "nation" in perf.columns:
    perf["nation_clean"] = perf["nation"].astype(str).str.extract(r'\b([a-zA-Z]{2,3})\b$')[0].str.lower()
    perf["british"] = perf["nation_clean"].isin(BRITISH_CODES).astype(int)
    print(f"British rows: {perf['british'].sum():.0f} | Non-British: {(perf['british']==0).sum()}")
else:
    perf["british"] = np.nan
    print("WARNING: No nation column found")

print(f"Performance data after cleaning: {len(perf):,} player-seasons")

# Aggregate hype to season level
def month_to_season(dt):
    return f"{dt.year}-{dt.year+1}" if dt.month >= 8 else f"{dt.year-1}-{dt.year}"

hype["season"] = hype["month"].apply(month_to_season)
hype_season = hype.groupby(["player", "season", "nationality_group"]).agg(
    hype_index_mean=("hype_index", "mean"),
    hype_index_peak=("hype_index", "max"),
    trends_mean=("trends_score", "mean"),
    guardian_count_season=("guardian_count", "sum"),
    wiki_views_mean=("wiki_views", "mean"),
).reset_index()

# Merge
panel = pd.merge(perf, hype_season, on=["player", "season"], how="inner")
print(f"\nMerged panel (pre-filter): {len(panel):,} player-seasons")

# Eligibility filter: >= 500 minutes
if "minutes" in panel.columns:
    panel = panel[panel["minutes"] >= 500]
    print(f"After >=500 min filter: {len(panel):,} player-seasons")

panel.to_csv("data/processed/panel_final.csv", index=False)
print(f"\nSaved: data/processed/panel_final.csv | Shape: {panel.shape}")
cols = [c for c in ["player","season","minutes","goals_per90","assists_per90","hype_index_mean","british"] if c in panel.columns]
print(panel[cols].head(10).to_string())
