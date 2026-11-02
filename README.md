# Football Media Hype & Player Development

## Research Question
Does media hype create measurable performance or development effects for young British players, and are these effects stronger than for comparable non-British players?

## Motivation
Erik ten Hag and others have argued the British media overhypes domestic players, then becomes excessively critical during natural dips in form — potentially harming confidence, consistency, and long-term development. This project quantifies that claim.

## Design
- **Population**: All outfield players who debuted in the Premier League aged 16–23, 2010–2024, with ≥500 minutes in any single season
- **Exposure**: Continuous hype index per player per month (Google Trends + Guardian article volume/sentiment + Wikipedia page views)
- **Outcome**: Performance trajectory slope (goal contributions per 90, progressive carries per 90, xG involvement)
- **Causal model**: Panel fixed effects regression (player FE + season FE); key coefficient = hype × British nationality interaction
- **Secondary**: Event study around hype spikes (±3 months)

## Data Sources
| Source | What it gives | Tool |
|--------|--------------|------|
| FBRef via `soccerdata` | Player performance metrics by season | Python |
| Google Trends via `pytrends` | Search interest volume, UK, weekly | Python |
| Guardian Open Platform API | Article count + sentiment, 2010–2024 | Python + HuggingFace |
| Wikipedia Pageviews API | Organic interest volume, daily | Python |

## Project Structure
```
football-media-hype/
├── data/
│   ├── raw/          # Unprocessed pulls from APIs and FBRef
│   └── processed/    # Cleaned, merged panel datasets
├── notebooks/        # Analysis notebooks (one per phase)
├── src/              # Reusable scripts
├── outputs/
│   ├── figures/      # Publication-quality plots
│   └── tables/       # Regression outputs
└── docs/             # Methods write-up, notes
```

## Phases
1. **Player population** — pull eligibility list from FBRef
2. **Hype index** — Google Trends + Guardian + Wikipedia pipeline
3. **NLP sentiment** — Guardian headlines through HuggingFace model
4. **Panel dataset** — merge performance + hype index
5. **Analysis** — fixed effects regression + event study
6. **Write-up** — methods section + GitHub documentation
