"""
Phase 7: Counterfactual Simulation — British Media Premium

Question: What would Grealish / Rashford / Foden's media coverage have looked
like if they were non-British and their coverage tracked performance the way
non-British PL players' does?

Method:
  1. Fit a coverage model on non-British PL players:
       guardian_count ~ ga_per90  (simple linear, season FE)
  2. Predict "non-British counterfactual" coverage for each target player
     using their actual on-pitch output each season
  3. Bootstrap 10,000 samples of the non-British model to generate
     95% confidence bands on the counterfactual
  4. Gap = actual coverage minus counterfactual = British premium per season

Outputs:
  outputs/figures/counterfactual_grealish.png
  outputs/figures/counterfactual_rashford.png
  outputs/figures/counterfactual_foden.png
  outputs/figures/counterfactual_all3.png
  outputs/tables/counterfactual_summary.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

mpl.use("Agg")

np.random.seed(42)
N_BOOTSTRAP = 10_000

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading panel...")
panel = pd.read_csv("data/processed/panel_final.csv")

# Build outcome variable
if "goals_per90" in panel.columns and "assists_per90" in panel.columns:
    panel["ga_per90"] = panel["goals_per90"].fillna(0) + panel["assists_per90"].fillna(0)

# Clean
panel = panel.dropna(subset=["guardian_count_season", "ga_per90", "british"])

print(f"Panel: {len(panel)} player-seasons | "
      f"British: {panel['british'].sum():.0f} | "
      f"Non-British: {(panel['british']==0).sum():.0f}")

# ── FIT BASELINE MODEL ON NON-BRITISH PLAYERS ─────────────────────────────────
# This is "what coverage looks like when it tracks performance, not nationality"
nb = panel[panel["british"] == 0].copy()
print(f"\nNon-British baseline sample: {len(nb)} player-seasons")
print(f"Players: {nb['player'].unique().tolist()}")

# Simple OLS: guardian_count ~ ga_per90
# Using raw counts (not log) for interpretability in plots
X_nb = np.column_stack([np.ones(len(nb)), nb["ga_per90"].values])
y_nb = nb["guardian_count_season"].values

# OLS coefficients
coef_nb, _, _, _ = np.linalg.lstsq(X_nb, y_nb, rcond=None)
intercept_nb, slope_nb = coef_nb
print(f"\nNon-British model: coverage = {intercept_nb:.2f} + {slope_nb:.2f} × GA/90")
print(f"Interpretation: each additional GA/90 unit → {slope_nb:.1f} more articles/season")

# Residuals for bootstrap
y_pred_nb = X_nb @ coef_nb
residuals_nb = y_nb - y_pred_nb

# ── BOOTSTRAP: 10,000 SAMPLES OF THE NON-BRITISH MODEL ───────────────────────
print(f"\nRunning {N_BOOTSTRAP:,} bootstrap samples...")
boot_intercepts = np.zeros(N_BOOTSTRAP)
boot_slopes = np.zeros(N_BOOTSTRAP)

for i in range(N_BOOTSTRAP):
    idx = np.random.choice(len(nb), size=len(nb), replace=True)
    X_b = X_nb[idx]
    y_b = y_nb[idx]
    try:
        c, _, _, _ = np.linalg.lstsq(X_b, y_b, rcond=None)
        boot_intercepts[i] = c[0]
        boot_slopes[i] = c[1]
    except Exception:
        boot_intercepts[i] = intercept_nb
        boot_slopes[i] = slope_nb

print(f"Bootstrap complete. Slope 95% CI: "
      f"[{np.percentile(boot_slopes, 2.5):.2f}, {np.percentile(boot_slopes, 97.5):.2f}]")

# ── COUNTERFACTUAL FUNCTION ───────────────────────────────────────────────────
def counterfactual_for_player(player_name):
    """Return season-by-season actual vs counterfactual coverage for one player."""
    p = panel[panel["player"] == player_name].copy().sort_values("season")
    if p.empty:
        print(f"  Player '{player_name}' not found in panel")
        return None

    ga = p["ga_per90"].values
    actual = p["guardian_count_season"].values

    # Point estimate counterfactual
    cf_point = intercept_nb + slope_nb * ga

    # Bootstrap CI on counterfactual
    cf_boot = boot_intercepts[:, None] + boot_slopes[:, None] * ga[None, :]
    cf_lo = np.percentile(cf_boot, 2.5, axis=0)
    cf_hi = np.percentile(cf_boot, 97.5, axis=0)

    # Premium per season
    premium = actual - cf_point
    premium_pct = np.where(cf_point > 0, (premium / cf_point) * 100, np.nan)

    result = p[["season", "ga_per90", "guardian_count_season"]].copy()
    result["cf_point"] = cf_point
    result["cf_lo"] = cf_lo
    result["cf_hi"] = cf_hi
    result["premium_articles"] = premium
    result["premium_pct"] = premium_pct
    return result

# ── RUN ALL THREE ─────────────────────────────────────────────────────────────
targets = ["Jack Grealish", "Marcus Rashford", "Phil Foden"]
results = {}
for name in targets:
    r = counterfactual_for_player(name)
    if r is not None:
        results[name] = r
        avg_premium = r["premium_articles"].mean()
        avg_premium_pct = r["premium_pct"].mean()
        print(f"\n{name}:")
        print(f"  Seasons: {len(r)}")
        print(f"  Avg actual coverage:        {r['guardian_count_season'].mean():.1f} articles/season")
        print(f"  Avg counterfactual:         {r['cf_point'].mean():.1f} articles/season")
        print(f"  Avg British premium:        {avg_premium:+.1f} articles/season ({avg_premium_pct:+.0f}%)")
        print(f"  Season breakdown:")
        for _, row in r.iterrows():
            marker = "▲" if row["premium_articles"] > 0 else "▼"
            print(f"    {row['season']}: actual={row['guardian_count_season']:.1f} | "
                  f"cf={row['cf_point']:.1f} | "
                  f"premium={row['premium_articles']:+.1f} ({row['premium_pct']:+.0f}%) {marker}")

# ── SAVE SUMMARY TABLE ────────────────────────────────────────────────────────
all_rows = []
for name, r in results.items():
    r = r.copy()
    r["player"] = name
    all_rows.append(r)
if all_rows:
    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv("outputs/tables/counterfactual_summary.csv", index=False)
    print("\nSaved: outputs/tables/counterfactual_summary.csv")

# ── PLOT FUNCTION ─────────────────────────────────────────────────────────────
BRIT_COLOR   = "#E63946"
CF_COLOR     = "#457B9D"
SHADING      = "#AED6F1"

def plot_single_player(name, r, ax=None, standalone=True):
    if standalone:
        fig, ax = plt.subplots(figsize=(11, 6))

    seasons = r["season"].tolist()
    x = np.arange(len(seasons))

    # Actual coverage
    ax.plot(x, r["guardian_count_season"], color=BRIT_COLOR,
            marker="o", linewidth=2.5, markersize=7, label="Actual coverage", zorder=4)

    # Counterfactual + CI band
    ax.plot(x, r["cf_point"], color=CF_COLOR, marker="s",
            linewidth=2, markersize=6, linestyle="--",
            label="Counterfactual (non-British model)", zorder=3)
    ax.fill_between(x, r["cf_lo"], r["cf_hi"],
                    color=SHADING, alpha=0.4, label="95% bootstrap CI", zorder=2)

    # Annotate premium gaps
    for i, row in r.reset_index(drop=True).iterrows():
        gap = row["premium_articles"]
        if abs(gap) > 0.3:
            mid_y = (row["guardian_count_season"] + row["cf_point"]) / 2
            color = BRIT_COLOR if gap > 0 else CF_COLOR
            ax.annotate(f"{gap:+.1f}", xy=(i, mid_y),
                        ha="center", va="center", fontsize=8.5,
                        color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor=color, alpha=0.8))

    # GA/90 as secondary axis
    ax2 = ax.twinx()
    ax2.bar(x, r["ga_per90"], color="#F4A261", alpha=0.3, width=0.6, label="GA/90 (output)")
    ax2.set_ylabel("GA per 90 mins (orange bars)", fontsize=10, color="#F4A261")
    ax2.tick_params(axis="y", labelcolor="#F4A261")
    ax2.set_ylim(0, max(r["ga_per90"].max() * 3, 1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=45, fontsize=10)
    ax.set_xlabel("Season", fontsize=11)
    ax.set_ylabel("Guardian Articles per Season", fontsize=11)
    ax.set_title(
        f"{name}: Actual vs Counterfactual Media Coverage\n"
        f"Gap = British media premium (articles above what performance would predict)",
        fontsize=12
    )

    # Merge legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
              loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)

    avg_prem = r["premium_articles"].mean()
    avg_pct  = r["premium_pct"].mean()
    ax.text(0.98, 0.04,
            f"Avg premium: {avg_prem:+.1f} articles/season ({avg_pct:+.0f}%)\n"
            f"Based on {N_BOOTSTRAP:,} bootstrap simulations",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#444444",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    if standalone:
        plt.tight_layout()
        fname = f"outputs/figures/counterfactual_{name.split()[-1].lower()}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")

# ── INDIVIDUAL PLOTS ──────────────────────────────────────────────────────────
print("\n=== Generating plots ===")
for name, r in results.items():
    plot_single_player(name, r, standalone=True)

# ── COMBINED 3-PANEL PLOT ─────────────────────────────────────────────────────
if len(results) == 3:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        "British Media Premium: Actual vs Counterfactual Coverage\n"
        "Counterfactual = predicted coverage if player were non-British "
        "(fitted on Ziyech, Sané, Torres, João Félix, Olise, Renato Sanches)",
        fontsize=13, y=1.01
    )
    for ax, (name, r) in zip(axes, results.items()):
        plot_single_player(name, r, ax=ax, standalone=False)
    plt.tight_layout()
    plt.savefig("outputs/figures/counterfactual_all3.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/figures/counterfactual_all3.png")

# ── PRINT HEADLINE FINDINGS ───────────────────────────────────────────────────
print("\n" + "="*60)
print("HEADLINE FINDINGS")
print("="*60)
for name, r in results.items():
    total_excess = r["premium_articles"].sum()
    avg_pct = r["premium_pct"].mean()
    peak_row = r.loc[r["premium_articles"].idxmax()]
    trough_row = r.loc[r["premium_articles"].idxmin()]
    print(f"\n{name}:")
    print(f"  Total excess articles over career: {total_excess:+.0f}")
    print(f"  Average premium: {avg_pct:+.0f}% per season")
    print(f"  Peak premium season: {peak_row['season']} "
          f"(+{peak_row['premium_articles']:.1f} articles, "
          f"GA/90={peak_row['ga_per90']:.2f})")
    print(f"  Lowest premium season: {trough_row['season']} "
          f"({trough_row['premium_articles']:+.1f} articles, "
          f"GA/90={trough_row['ga_per90']:.2f})")

print("\nDone.")
