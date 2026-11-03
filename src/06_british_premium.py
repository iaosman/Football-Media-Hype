"""
Phase 6: British Media Premium Analysis

Research question (Level 2):
  Do British players receive disproportionate Guardian media coverage
  per unit of on-pitch output compared to equally-performing non-British peers?

This is a DESCRIPTIVE claim — no causal inference required.
We measure coverage disproportion, not coverage effects.

Approach:
  1. Compute articles-per-goal-contribution (coverage per output unit)
  2. Descriptive comparison: British vs non-British at matched performance tiers
  3. Regression: guardian_count ~ ga_per90 + british + interaction + season FE
     (no player FE — we want BETWEEN-player variation, that's the premium)
  4. Scatter plot with fitted lines + annotated player names
  5. Bar chart: mean articles per output tier, split by nationality

Output:
  outputs/tables/british_premium_regression.csv
  outputs/tables/british_premium_descriptive.csv
  outputs/figures/coverage_vs_performance.png
  outputs/figures/coverage_premium_bars.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

mpl.use("Agg")

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading panel...")
panel = pd.read_csv("data/processed/panel_final.csv")
print(f"Shape: {panel.shape} | Players: {panel['player'].nunique()}")

# ── OUTCOME + CONTROLS ────────────────────────────────────────────────────────
# Primary performance metric
if "goals_per90" in panel.columns and "assists_per90" in panel.columns:
    panel["ga_per90"] = panel["goals_per90"].fillna(0) + panel["assists_per90"].fillna(0)

# Total goal contributions (absolute, not rate)
if "goals" in panel.columns and "assists" in panel.columns:
    panel["ga_total"] = panel["goals"].fillna(0) + panel["assists"].fillna(0)

# Coverage per output unit — core metric
# guardian_count_season = total Guardian articles mentioning player that season
# articles_per_ga = how much coverage each goal contribution "earns"
panel["articles_per_ga"] = np.where(
    panel["ga_total"] > 0,
    panel["guardian_count_season"] / panel["ga_total"],
    np.nan  # exclude zero-output seasons from ratio (would be infinite)
)

# Log transform for regression (guardian counts are right-skewed)
panel["log_guardian"] = np.log1p(panel["guardian_count_season"])

# Drop rows missing key variables
reg_df = panel.dropna(subset=["guardian_count_season", "ga_per90", "british"]).copy()
print(f"Regression sample: {len(reg_df)} player-seasons")

# ── 1. DESCRIPTIVE SUMMARY ────────────────────────────────────────────────────
print("\n=== Descriptive: Coverage vs Performance ===")

desc_rows = []
for nat in [1, 0]:
    label = "British" if nat == 1 else "Non-British"
    sub = reg_df[reg_df["british"] == nat]
    row = {
        "group": label,
        "n": len(sub),
        "ga_per90_mean": sub["ga_per90"].mean(),
        "guardian_count_mean": sub["guardian_count_season"].mean(),
        "articles_per_ga_mean": sub["articles_per_ga"].mean(),
        "articles_per_ga_median": sub["articles_per_ga"].median(),
    }
    desc_rows.append(row)
    print(f"{label} (n={row['n']}):")
    print(f"  GA/90:              {row['ga_per90_mean']:.3f}")
    print(f"  Guardian articles:  {row['guardian_count_mean']:.1f}/season")
    print(f"  Articles per G+A:  {row['articles_per_ga_mean']:.1f} (median: {row['articles_per_ga_median']:.1f})")

desc_df = pd.DataFrame(desc_rows)

# Performance-tier breakdown
print("\n=== Coverage by Performance Tier ===")
reg_df["perf_tier"] = pd.cut(
    reg_df["ga_per90"],
    bins=[0, 0.3, 0.5, 0.7, 5.0],
    labels=["Low (<0.30)", "Mid (0.30–0.50)", "Good (0.50–0.70)", "Elite (>0.70)"]
)

tier_rows = []
for tier in reg_df["perf_tier"].cat.categories:
    tier_sub = reg_df[reg_df["perf_tier"] == tier]
    for nat in [1, 0]:
        label = "British" if nat == 1 else "Non-British"
        nat_sub = tier_sub[tier_sub["british"] == nat]
        if len(nat_sub) >= 2:
            row = {
                "tier": tier,
                "group": label,
                "n": len(nat_sub),
                "guardian_mean": nat_sub["guardian_count_season"].mean(),
                "ga_per90_mean": nat_sub["ga_per90"].mean(),
                "articles_per_ga_mean": nat_sub["articles_per_ga"].mean(),
            }
            tier_rows.append(row)
            print(f"  [{tier}] {label} (n={row['n']}): "
                  f"{row['guardian_mean']:.1f} articles/season | "
                  f"{row['articles_per_ga_mean']:.1f} articles/G+A")

tier_df = pd.DataFrame(tier_rows)

# ── 2. REGRESSION: Coverage ~ Performance + British ───────────────────────────
print("\n=== Regression: Guardian Coverage ~ GA/90 + British ===")
print("(No player FE — between-player variation is the point)")
print("Outcome: log(1 + guardian_count_season)")

if HAS_STATSMODELS:
    # Main model: does british predict MORE coverage controlling for performance?
    formula_main = "log_guardian ~ ga_per90 + british + C(season)"
    # Interaction model: does british amplify the coverage return on performance?
    formula_interact = "log_guardian ~ ga_per90 + british + ga_per90:british + C(season)"

    results = {}
    for name, formula in [("Main", formula_main), ("Interaction", formula_interact)]:
        try:
            model = smf.ols(formula, data=reg_df).fit()
            coefs = model.summary2().tables[1]
            key_vars = ["ga_per90", "british", "ga_per90:british"]
            key = coefs[coefs.index.isin(key_vars)]
            results[name] = key
            print(f"\n--- {name} model ---")
            print(key[["Coef.", "Std.Err.", "t", "P>|t|"]].to_string())
            print(f"  R²={model.rsquared:.3f} | n={int(model.nobs)}")
        except Exception as e:
            print(f"  Error in {name} model: {e}")

    # Save interaction model (more informative)
    if "Interaction" in results:
        results["Interaction"].to_csv("outputs/tables/british_premium_regression.csv")
        print("\nSaved: outputs/tables/british_premium_regression.csv")

    # Compute the premium in interpretable terms
    if "Main" in results:
        coefs_main = results["Main"]
        if "british" in coefs_main.index:
            brit_coef = coefs_main.loc["british", "Coef."]
            brit_pval = coefs_main.loc["british", "P>|t|"]
            # log(1+x) coefficient → approximate % uplift
            pct_uplift = (np.exp(brit_coef) - 1) * 100
            print(f"\n★ British premium: {brit_coef:.3f} log-units "
                  f"≈ {pct_uplift:+.0f}% more articles per season "
                  f"(p={brit_pval:.3f})")

desc_df.to_csv("outputs/tables/british_premium_descriptive.csv", index=False)
print("Saved: outputs/tables/british_premium_descriptive.csv")

# ── 3. SCATTER: Coverage vs Performance, British vs Non-British ───────────────
print("\n=== Plotting: Coverage vs Performance ===")

fig, ax = plt.subplots(figsize=(11, 7))

brit_colors = {1: "#E63946", 0: "#457B9D"}
brit_labels = {1: "British", 0: "Non-British"}

for nat in [1, 0]:
    sub = reg_df[reg_df["british"] == nat]
    ax.scatter(sub["ga_per90"], sub["guardian_count_season"],
               color=brit_colors[nat], label=brit_labels[nat],
               s=60, alpha=0.75, zorder=3)

    # Fitted line (linear)
    if len(sub) >= 3:
        z = np.polyfit(sub["ga_per90"], sub["guardian_count_season"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sub["ga_per90"].min(), sub["ga_per90"].max(), 100)
        ax.plot(x_line, p(x_line), color=brit_colors[nat],
                linewidth=2, linestyle="--", alpha=0.8)

# Annotate notable players (top coverage or outliers)
notable = ["Jack Grealish", "Marcus Rashford", "Jadon Sancho",
           "Bukayo Saka", "Phil Foden", "Cole Palmer", "Pedri", "Gavi"]
for _, row in reg_df.iterrows():
    if row["player"] in notable:
        ax.annotate(
            row["player"].split()[-1],  # surname only
            xy=(row["ga_per90"], row["guardian_count_season"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7.5, color="#333333", alpha=0.85
        )

ax.set_xlabel("Goal + Assist Contributions per 90 minutes", fontsize=12)
ax.set_ylabel("Guardian Articles Mentioning Player (season total)", fontsize=12)
ax.set_title(
    "British Media Premium: Coverage vs On-Pitch Output\n"
    "Guardian articles per season — British vs Non-British PL players aged 17–23",
    fontsize=13
)
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Annotation explaining the premium
ax.text(0.03, 0.97,
        "Dashed lines = coverage trend per nationality group\n"
        "Gap between lines at equal GA/90 = media premium",
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

plt.tight_layout()
plt.savefig("outputs/figures/coverage_vs_performance.png", dpi=150)
plt.close()
print("Saved: outputs/figures/coverage_vs_performance.png")

# ── 4. BAR CHART: Articles per performance tier ───────────────────────────────
if len(tier_df) >= 2:
    print("\n=== Plotting: Coverage premium by performance tier ===")

    pivot = tier_df.pivot(index="tier", columns="group", values="guardian_mean")
    if "British" in pivot.columns and "Non-British" in pivot.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(pivot))
        w = 0.35
        bars_b = ax.bar(x - w/2, pivot["British"], w,
                        label="British", color="#E63946", alpha=0.85)
        bars_nb = ax.bar(x + w/2, pivot["Non-British"], w,
                         label="Non-British", color="#457B9D", alpha=0.85)

        # Premium labels
        for i, tier in enumerate(pivot.index):
            b = pivot.loc[tier, "British"]
            nb = pivot.loc[tier, "Non-British"]
            if pd.notna(b) and pd.notna(nb) and nb > 0:
                premium = ((b - nb) / nb) * 100
                ax.text(i, max(b, nb) + 0.2, f"{premium:+.0f}%",
                        ha="center", fontsize=9, fontweight="bold",
                        color="#E63946" if premium > 0 else "#457B9D")

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, fontsize=11)
        ax.set_xlabel("Performance Tier (GA per 90 mins)", fontsize=12)
        ax.set_ylabel("Mean Guardian Articles per Season", fontsize=12)
        ax.set_title(
            "British Media Premium by Performance Tier\n"
            "Mean Guardian articles per season — matched performance brackets",
            fontsize=13
        )
        ax.legend(fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("outputs/figures/coverage_premium_bars.png", dpi=150)
        plt.close()
        print("Saved: outputs/figures/coverage_premium_bars.png")

print("\n=== British Premium Analysis complete ===")
