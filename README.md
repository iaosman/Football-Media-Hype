# Football Media Hype & Player Development

**Research questions:**
1. Do Premier League players receive media coverage beyond what their on-pitch output and club context would predict?
2. Is British nationality a systematic driver of that excess coverage?
3. Does excess coverage affect career development trajectories?

## Overview

This project builds a novel linked dataset spanning eight Premier League seasons (2016/17 to 2023/24), applies NLP sentiment analysis to football journalism, and uses a two-stage causal inference framework to test whether media overexposure leaves a measurable footprint on player development.

It is an independent portfolio piece combining data engineering, applied NLP, and panel econometrics.

## Pipeline

```
Player population (vaastav/FPL)
        │
        ▼
Nationality classification          ← regex + Wikidata + hardcoded lookup
Debut age via FPL API + Wikidata
        │
        ▼
Guardian API coverage collection    ← 1,180 players × 8 seasons → ~6,300 player-seasons
        │
        ▼
Sentiment scoring                   ← cardiffnlp/twitter-roberta-base-sentiment-latest
        │
        ▼
Full panel merge                    ← coverage + sentiment + FPL performance stats
        │
        ▼
Two-stage regression
  Stage 1: excess coverage residual
  Stage 2: panel FE development outcome
```

**Scripts:**

| Script | What it does |
|--------|-------------|
| `src/00_build_population.py` | Build the 1,180-player outfield population from vaastav FPL data |
| `src/00c_nationality_hardcoded.py` | Classify nationality (British / non-British) via regex, Wikidata, and hardcoded overrides |
| `src/01_collect_coverage.py` | Collect Guardian API article counts and headlines per player-season |
| `src/02_sentiment.py` | Score headline sentiment using a RoBERTa model fine-tuned on Twitter data |
| `src/03_build_panel.py` | Merge population, coverage, sentiment, and FPL performance into one panel |
| `src/04_regression.py` | Two-stage excess-coverage regression with two-way panel fixed effects |

## Data

| File | Description |
|------|-------------|
| `data/raw/player_population.csv` | 1,180 outfield players with debut season, position, debut age, nationality, and British flag |
| `data/processed/coverage_panel.csv` | 6,362 player-season rows with article count and headline count |
| `data/processed/sentiment_panel.csv` | Sentiment mean and positive percentage per player-season |
| `data/processed/full_panel.csv` | Merged panel with FPL performance statistics (ga_per90, minutes, club quality) |
| `data/processed/panel_with_excess.csv` | Full panel with the excess coverage residual added |
| `results/stage1_ols.txt` | Stage 1 OLS output |
| `results/stage2_fe.txt` | Stage 2 PanelOLS output |

Raw coverage JSONs (`data/raw/coverage/`) are excluded from the repo (1,180 files). Re-run `src/01_collect_coverage.py` with a Guardian API key to reproduce them.

## Key Findings

### Stage 1: British media premium

After controlling for goals and assists per 90 minutes, playing time, club quality, and season, British players receive significantly more Guardian football coverage than equivalent non-British players:

| Group | Mean excess log-coverage |
|-------|------------------------|
| British (n=384) | +0.187 |
| Non-British (n=796) | -0.121 |

**t = 6.32, p < 0.0001**

Notably, on-pitch output (`ga_per90`) is not a significant predictor of coverage (p = 0.38), while nationality carries the largest and most significant coefficient in the coverage model (coef = 0.31). This suggests Guardian football coverage is driven more by national identity than by on-pitch merit.

### Stage 2: Development outcome

A two-way fixed-effects panel model (player and season fixed effects, clustered standard errors) tests whether excess coverage in season *t* predicts next-season `ga_per90`:

| Variable | Coefficient | p-value |
|----------|------------|---------|
| excess_coverage | -0.017 | 0.597 |
| british × excess_coverage | +0.013 | 0.737 |
| ga_per90 (lagged) | -0.137 | 0.013 |

Excess coverage shows no significant effect on next-season performance. The negative and significant lagged performance term is consistent with mean reversion in attacking output, which serves as a methodological check that the model is working as expected.

**Interpretation:** The null Stage 2 result reflects the constraints of the available data rather than a definitive absence of effect. The outcome panel is sparse (average 2.6 seasons per player after filtering), `ga_per90` captures only one narrow dimension of player development, and the ~54% name-match rate between the player population and vaastav introduces attenuation bias. The Stage 1 finding is the substantive contribution: a robust, well-identified British media premium that is not explained by performance or club context.

## Limitations

- **Coverage proxy:** Guardian article counts capture broadsheet football journalism only. The British premium may be larger in tabloid or social media data.
- **Outcome measure:** `ga_per90` in the following season is a narrow proxy for development. Transfer value, career earnings, peak performance age, or minutes at a higher level would each be more informative.
- **Name matching:** Approximately 54% of player-seasons have matched FPL performance data. Players whose names differ between the population CSV and vaastav (due to accented characters, name changes, or loan registrations) are dropped, attenuating Stage 2 estimates toward zero.
- **Sentiment model:** The RoBERTa model was fine-tuned on Twitter text rather than sports journalism. Sentiment means are approximately neutral across the sample (mean 0.024), which may reflect domain mismatch as much as genuine neutrality.

## Next Steps

1. **Transfer market value as the development outcome.** Linking player-seasons to Transfermarkt valuation data would replace `ga_per90` with a richer, forward-looking measure that reflects how clubs and scouts actually price development. It would also connect the analysis to the economics literature on labour market premia and media salience, giving Stage 2 a theoretically cleaner test.

2. **Multi-source media coverage.** The Guardian captures broadsheet football journalism only. Replicating the coverage model with tabloid sources (Mirror, Sun, Daily Mail) and Twitter/X mention counts would test whether the British premium is larger in higher-volume, lower-quality media — and whether sentiment effects, absent here, emerge in coverage that is more emotional in register.

3. **Domain-adapted sentiment model.** The RoBERTa model was fine-tuned on Twitter text, and the near-neutral sentiment means in this sample likely reflect domain mismatch as much as genuine neutrality in football journalism. Fine-tuning on annotated Guardian match reports and transfer news would sharpen the sentiment signal and make the interaction between excess coverage and sentiment tone a viable second-stage predictor.

## Reproducing the Analysis

```bash
# 1. Install dependencies
pip install pandas requests transformers torch linearmodels statsmodels

# 2. Set Guardian API key (free at https://open-platform.theguardian.com/access/)
export GUARDIAN_KEY=your-key-here

# 3. Run pipeline in order
python3 src/00_build_population.py
python3 src/00c_nationality_hardcoded.py
python3 src/01_collect_coverage.py        # ~2 hours, resumable
python3 src/02_sentiment.py               # ~20 min on CPU
python3 src/03_build_panel.py
python3 src/04_regression.py
```

Results are written to `data/processed/` and `results/`.

## Dependencies

- Python 3.10+
- `pandas`, `numpy`, `requests`, `scipy`
- `transformers`, `torch` (sentiment scoring)
- `linearmodels`, `statsmodels` (panel regression)
- Guardian Content API key (free tier sufficient)
- vaastav/Fantasy-Premier-League (fetched at runtime, no local copy needed)
