"""
04_regression.py
Two-stage excess-coverage analysis.

Stage 1: excess_coverage = residual from
    log(article_count+1) ~ ga_per90 + log(minutes+1) + club_quality + season_idx
    (british excluded so residual captures any British coverage premium)

Stage 2: panel FE regression
    ga_per90_{t+1} ~ excess_coverage_t + ga_per90_t + minutes_t
                   + british × excess_coverage_t + debut_age
                   + player_FE + season_FE

Outputs:
  data/processed/panel_with_excess.csv   (panel + excess_coverage column)
  results/stage1_ols.txt
  results/stage2_fe.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Optional: linearmodels for panel FE; fall back to statsmodels with dummies
try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("linearmodels not found — using statsmodels with entity dummies")

import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

Path("results").mkdir(exist_ok=True)

# ── 0. Load panel ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/full_panel.csv")
print(f"Raw panel: {df.shape}")

# ── 1. Restrict to player-seasons with FPL data (minutes recorded) ─────────────
df = df[df["minutes"] > 0].copy()
print(f"After minutes>0 filter: {df.shape}")

# ── 2. Log transforms ──────────────────────────────────────────────────────────
df["log_coverage"]  = np.log1p(df["article_count"])
df["log_minutes"]   = np.log1p(df["minutes"])
df["british_int"]   = df["british"].astype(int)

# ── 3. STAGE 1: Excess coverage OLS ───────────────────────────────────────────
print("\n── STAGE 1: Coverage model ──")
stage1_formula = "log_coverage ~ ga_per90 + log_minutes + club_quality + season_idx"
m1 = smf.ols(stage1_formula, data=df).fit()
print(m1.summary().tables[1])

df["excess_coverage"] = m1.resid
df["log_coverage_hat"] = m1.fittedvalues

print(f"\nExcess coverage stats:")
print(df["excess_coverage"].describe().round(3))
print(f"\nMean excess coverage — British:     {df[df['british_int']==1]['excess_coverage'].mean():.3f}")
print(f"Mean excess coverage — Non-British: {df[df['british_int']==0]['excess_coverage'].mean():.3f}")
t, p = stats.ttest_ind(
    df[df['british_int']==1]['excess_coverage'].dropna(),
    df[df['british_int']==0]['excess_coverage'].dropna()
)
print(f"t-test t={t:.3f}, p={p:.4f}")

with open("results/stage1_ols.txt", "w") as f:
    f.write(m1.summary().as_text())
print("Saved: results/stage1_ols.txt")

# ── 4. Build next-season outcome ───────────────────────────────────────────────
# Sort and shift: for each player, what is ga_per90 in the following season?
SEASON_ORDER = ["2016-17","2017-18","2018-19","2019-20","2020-21","2021-22","2022-23","2023-24"]
df["season_rank"] = df["season"].map({s:i for i,s in enumerate(SEASON_ORDER)})
df = df.sort_values(["player","season_rank"])

df["ga_per90_next"] = df.groupby("player")["ga_per90"].shift(-1)
df["minutes_next"]  = df.groupby("player")["minutes"].shift(-1)

# Keep rows where next season exists and player had meaningful minutes both seasons
outcome_df = df.dropna(subset=["ga_per90_next"]).copy()
outcome_df = outcome_df[outcome_df["minutes"] >= 450].copy()  # ~5 full games minimum
print(f"\nOutcome sample (minutes>=450, next season exists): {len(outcome_df)} rows, "
      f"{outcome_df['player'].nunique()} players")

# ── 5. STAGE 2: Panel FE regression ───────────────────────────────────────────
print("\n── STAGE 2: Development outcome model ──")

if HAS_LINEARMODELS:
    # Two-way FE: player + season
    outcome_df = outcome_df.set_index(["player","season_idx"])
    mod = PanelOLS.from_formula(
        "ga_per90_next ~ excess_coverage + ga_per90 + log_minutes "
        "+ british_int + british_int:excess_coverage + debut_age "
        "+ EntityEffects + TimeEffects",
        data=outcome_df,
        drop_absorbed=True,
    )
    res2 = mod.fit(cov_type="clustered", cluster_entity=True)
    print(res2.summary)
    summary_text = str(res2.summary)
    outcome_df = outcome_df.reset_index()
else:
    # Statsmodels with player dummies (time FE via C(season))
    stage2_formula = (
        "ga_per90_next ~ excess_coverage + ga_per90 + log_minutes "
        "+ british_int + british_int:excess_coverage + debut_age "
        "+ C(season) + C(player)"
    )
    res2 = smf.ols(stage2_formula, data=outcome_df).fit(
        cov_type="cluster", cov_kwds={"groups": outcome_df["player"]}
    )
    # Print only key coefficients (not the 1000+ player dummies)
    key_vars = [v for v in res2.params.index
                if not v.startswith("C(player)") and v != "Intercept"]
    print(res2.summary2().tables[1].loc[key_vars])
    summary_text = res2.summary2().as_text()

with open("results/stage2_fe.txt", "w") as f:
    f.write(summary_text)
print("Saved: results/stage2_fe.txt")

# ── 6. Key results summary ─────────────────────────────────────────────────────
print("\n── KEY RESULTS ──")
if HAS_LINEARMODELS:
    params = res2.params
    pvals  = res2.pvalues
else:
    params = res2.params
    pvals  = res2.pvalues

for name in ["excess_coverage", "british_int", "british_int:excess_coverage",
             "ga_per90", "log_minutes", "debut_age"]:
    for k in params.index:
        if name in k and "C(player)" not in k and "C(season)" not in k:
            print(f"  {k:45s}  coef={params[k]:+.4f}  p={pvals[k]:.4f}")
            break

# ── 7. Save panel with excess coverage ────────────────────────────────────────
df.to_csv("data/processed/panel_with_excess.csv", index=False)
print("\nSaved: data/processed/panel_with_excess.csv")
