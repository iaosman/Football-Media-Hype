"""
Phase 5: Panel fixed effects regression + event study.

Models:
  V1 — Contemporaneous: hype_index_mean (baseline, likely reverse-causal)
  V2 — Lagged:          hype_lag1 (prior-season hype → this-season performance)
                        Fixes temporal ordering; preferred causal interpretation.

  Both models: OLS with player + season fixed effects, british interaction term.
  Outcome: ga_per90 (goals + assists per 90 minutes).

Event study:
  - Define hype spike: player-season where hype_index_peak > player's 75th percentile
  - Plot performance in t-2 to t+2 seasons around spike
  - Compare british vs non-british

Output:
  outputs/tables/regression_results_v1.csv   (contemporaneous)
  outputs/tables/regression_results_v2.csv   (lagged — primary)
  outputs/tables/model_comparison.csv        (side-by-side key coefs)
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

def run_fe_model(panel_df, hype_col, label, out_csv):
    """Run OLS with player + season FE, return key coefficients as DataFrame."""
    interact_col = f"{hype_col}_x_british"
    df = panel_df.copy()

    if hype_col not in df.columns:
        print(f"  [{label}] Column '{hype_col}' not found — skipping")
        return None

    df[interact_col] = df[hype_col] * df["british"].fillna(0)
    required = ["ga_per90", hype_col, "british"]
    df = df.dropna(subset=required)

    if len(df) < 10:
        print(f"  [{label}] Insufficient data after dropna ({len(df)} rows) — skipping")
        return None

    print(f"\n=== {label} | n={len(df)} player-seasons ===")

    if HAS_STATSMODELS:
        formula = f"ga_per90 ~ {hype_col} + british + {interact_col} + C(player) + C(season)"
        try:
            model = smf.ols(formula, data=df).fit()
            coefs = model.summary2().tables[1]
            print(coefs[["Coef.", "Std.Err.", "t", "P>|t|"]].to_string())
            key = coefs[coefs.index.isin([hype_col, "british", interact_col])].copy()
            key["model"] = label
            key.to_csv(out_csv)
            print(f"  Saved: {out_csv}")
            return key
        except Exception as e:
            print(f"  Regression error: {e}")
            return None
    else:
        print("  Manual within-transformation (demean by player and season):")
        for col in ["ga_per90", hype_col, interact_col]:
            df[f"{col}_dm"] = df[col] - df.groupby("player")[col].transform("mean")
            df[f"{col}_dm"] -= df.groupby("season")[f"{col}_dm"].transform("mean")
        from numpy.linalg import lstsq
        X = df[[f"{hype_col}_dm", f"{interact_col}_dm"]].values
        X = np.hstack([np.ones((len(X), 1)), X])
        y = df["ga_per90_dm"].values
        coef, _, _, _ = lstsq(X, y, rcond=None)
        print(f"  Intercept:        {coef[0]:.4f}")
        print(f"  {hype_col}:  {coef[1]:.4f}")
        print(f"  hype × british:   {coef[2]:.4f}")
        print("  (SEs unavailable in manual mode — install statsmodels for full output)")
        return None

# V1 — Contemporaneous (baseline, likely reverse-causal)
print("\n" + "="*60)
print("V1: CONTEMPORANEOUS HYPE (baseline)")
print("="*60)
key_v1 = run_fe_model(panel, "hype_index_mean", "V1_Contemporaneous",
                       "outputs/tables/regression_results_v1.csv")

# V2 — Lagged (primary causal model)
print("\n" + "="*60)
print("V2: LAGGED HYPE — primary causal model")
print("    (prior-season hype → this-season performance)")
print("="*60)
key_v2 = run_fe_model(panel, "hype_lag1", "V2_Lagged",
                       "outputs/tables/regression_results_v2.csv")

# Side-by-side comparison
if key_v1 is not None and key_v2 is not None:
    comparison = pd.concat([key_v1, key_v2])
    comparison.to_csv("outputs/tables/model_comparison.csv")
    print("\n=== Model Comparison (key coefficients) ===")
    print(comparison[["Coef.", "P>|t|", "model"]].to_string())
    print("Saved: outputs/tables/model_comparison.csv")
elif key_v2 is not None:
    print("\n=== V2 Lagged Results (primary model) ===")
    print(key_v2[["Coef.", "Std.Err.", "P>|t|"]].to_string())

print("\nInterpretation guide:")
print("  V1 hype_index_mean coef: contemporaneous association (likely inflated by reverse causality)")
print("  V2 hype_lag1 coef:       lagged effect — causal interpretation defensible")
print("  If V2 < V1 or becomes insignificant: confirms reverse causality was driving V1")

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
