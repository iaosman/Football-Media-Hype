"""
Phase 5: Panel fixed effects regression + event study.

Models:
  1. OLS with player + season fixed effects (via dummies or within-transform)
  2. Key coefficient: hype_index_mean × british (interaction)
  3. Outcome: goals_assists_per90 (primary), xg_per90 (secondary)

Event study:
  - Define hype spike: player-season where hype_index_peak > player's 75th percentile
  - Plot performance in t-2 to t+2 seasons around spike
  - Compare british vs non-british

Output:
  outputs/tables/regression_results.csv
  outputs/figures/event_study.png
  outputs/figures/hype_trajectories.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

# Use Agg backend (no display needed)
mpl.use("Agg")

# Try to import statsmodels for proper FE regression
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not found — using manual within-transformation")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading panel...")
panel = pd.read_csv("data/processed/panel_final.csv")
print(f"Panel shape: {panel.shape}")
print(f"Players: {panel['player'].nunique()} | Seasons: {panel['season'].nunique()}")

# ── OUTCOME VARIABLE ──────────────────────────────────────────────────────────
# Primary: goal contributions per 90
if "goals_per90" in panel.columns and "assists_per90" in panel.columns:
    panel["ga_per90"] = panel["goals_per90"].fillna(0) + panel["assists_per90"].fillna(0)
elif "goals" in panel.columns and "assists" in panel.columns and "nineties_played" in panel.columns:
    panel["ga_per90"] = (panel["goals"].fillna(0) + panel["assists"].fillna(0)) / panel["nineties_played"].replace(0, np.nan)

# ── 1. DESCRIPTIVE SUMMARY ────────────────────────────────────────────────────
print("\n=== Descriptive Summary ===")
for nat in [1, 0]:
    label = "British" if nat == 1 else "Non-British"
    sub = panel[panel["british"] == nat]
    if "hype_index_mean" in sub.columns:
        print(f"{label}: n={len(sub)}, hype_mean={sub['hype_index_mean'].mean():.3f}, "
              f"hype_sd={sub['hype_index_mean'].std():.3f}")
    if "ga_per90" in sub.columns:
        print(f"  GA/90: mean={sub['ga_per90'].mean():.3f}, sd={sub['ga_per90'].std():.3f}")

# ── 2. FIXED EFFECTS REGRESSION ──────────────────────────────────────────────
print("\n=== Panel Fixed Effects Regression ===")

# Create interaction term
panel["hype_x_british"] = panel["hype_index_mean"] * panel["british"].fillna(0)

if HAS_STATSMODELS and "ga_per90" in panel.columns:
    # Use C() for categorical fixed effects
    formula = "ga_per90 ~ hype_index_mean + british + hype_x_british + C(player) + C(season)"
    try:
        model = smf.ols(formula, data=panel.dropna(subset=["ga_per90", "hype_index_mean", "british"])).fit()
        print(model.summary2().tables[1][["Coef.", "Std.Err.", "t", "P>|t|"]].to_string())
        # Save key coefficients
        coefs = model.summary2().tables[1]
        key = coefs[coefs.index.isin(["hype_index_mean", "british", "hype_x_british"])]
        key.to_csv("outputs/tables/regression_results.csv")
        print("\nKey coefficients saved: outputs/tables/regression_results.csv")
    except Exception as e:
        print(f"Regression error: {e}")
else:
    print("Manual within-transformation (demean by player and season):")
    reg_df = panel.dropna(subset=["hype_index_mean", "british"]).copy()
    if "ga_per90" in reg_df.columns:
        reg_df = reg_df.dropna(subset=["ga_per90"])
        for col in ["ga_per90", "hype_index_mean", "hype_x_british"]:
            reg_df[f"{col}_dm"] = reg_df[col] - reg_df.groupby("player")[col].transform("mean")
            reg_df[f"{col}_dm"] = reg_df[f"{col}_dm"] - reg_df.groupby("season")[f"{col}_dm"].transform("mean")
        # Simple OLS on demeaned
        from numpy.linalg import lstsq
        X = reg_df[["hype_index_mean_dm", "hype_x_british_dm"]].values
        X = np.hstack([np.ones((len(X), 1)), X])
        y = reg_df["ga_per90_dm"].values
        coef, _, _, _ = lstsq(X, y, rcond=None)
        print(f"  Intercept:          {coef[0]:.4f}")
        print(f"  hype_index_mean:    {coef[1]:.4f}")
        print(f"  hype × british:     {coef[2]:.4f}")
        print("  (Note: SEs not computed in manual mode — install statsmodels for full output)")

# ── 3. HYPE TRAJECTORY PLOT ───────────────────────────────────────────────────
print("\n=== Plotting hype trajectories ===")

if "hype_index_mean" in panel.columns and "british" in panel.columns:
    fig, ax = plt.subplots(figsize=(12, 5))

    for nat, label, color in [(1, "British", "#E63946"), (0, "Non-British", "#457B9D")]:
        sub = panel[panel["british"] == nat].copy()
        if sub.empty:
            continue
        season_mean = sub.groupby("season")["hype_index_mean"].mean()
        ax.plot(season_mean.index, season_mean.values, marker="o", label=label,
                color=color, linewidth=2, markersize=5)

    ax.set_xlabel("Season", fontsize=11)
    ax.set_ylabel("Mean Hype Index (0–1)", fontsize=11)
    ax.set_title("Mean Monthly Hype Index by Season\nBritish vs Non-British PL Players", fontsize=13)
    ax.legend(fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/figures/hype_trajectories.png", dpi=150)
    plt.close()
    print("Saved: outputs/figures/hype_trajectories.png")

# ── 4. EVENT STUDY ────────────────────────────────────────────────────────────
print("\n=== Event Study ===")

if "hype_index_peak" in panel.columns and "ga_per90" in panel.columns:
    # Define spike: season where hype_index_peak exceeds player's own 75th pctile
    panel["hype_p75"] = panel.groupby("player")["hype_index_peak"].transform(lambda x: x.quantile(0.75))
    panel["spike"] = (panel["hype_index_peak"] > panel["hype_p75"]).astype(int)

    # For each spike event, extract performance t-2 to t+2 relative to spike season
    panel_sorted = panel.sort_values(["player", "season"]).copy()
    panel_sorted["season_idx"] = panel_sorted.groupby("player").cumcount()
    spike_seasons = panel_sorted[panel_sorted["spike"] == 1][["player", "season_idx", "british"]].copy()

    records = []
    for _, row in spike_seasons.iterrows():
        p, t0, nat = row["player"], row["season_idx"], row["british"]
        for offset in range(-2, 3):
            t = t0 + offset
            match = panel_sorted[(panel_sorted["player"] == p) & (panel_sorted["season_idx"] == t)]
            if not match.empty and "ga_per90" in match.columns:
                records.append({
                    "offset": offset,
                    "british": nat,
                    "ga_per90": match["ga_per90"].values[0],
                })

    if records:
        es_df = pd.DataFrame(records)
        fig, ax = plt.subplots(figsize=(9, 5))
        for nat, label, color in [(1, "British", "#E63946"), (0, "Non-British", "#457B9D")]:
            sub = es_df[es_df["british"] == nat].groupby("offset")["ga_per90"].agg(["mean", "sem"])
            ax.errorbar(sub.index, sub["mean"], yerr=sub["sem"] * 1.96,
                        label=label, color=color, marker="o", linewidth=2, capsize=4)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1, label="Hype spike")
        ax.set_xlabel("Seasons relative to hype spike", fontsize=11)
        ax.set_ylabel("Goal + Assist contributions per 90", fontsize=11)
        ax.set_title("Event Study: Performance Around Hype Spikes\nBritish vs Non-British Players", fontsize=13)
        ax.legend(fontsize=11)
        ax.set_xticks(range(-2, 3))
        plt.tight_layout()
        plt.savefig("outputs/figures/event_study.png", dpi=150)
        plt.close()
        print("Saved: outputs/figures/event_study.png")
    else:
        print("No spike events found — check panel data")

print("\nAnalysis complete.")
