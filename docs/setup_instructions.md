# Setup Instructions

## Prerequisites
- Python 3.9+
- pip
- A free Guardian API key (takes 2 minutes)

## 1. Get your Guardian API key
1. Go to: https://open-platform.theguardian.com/access/
2. Click "Register for a developer key"
3. Fill in name + email → key arrives instantly by email
4. Set it: `export GUARDIAN_API_KEY="your_key_here"`

## 2. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/football-media-hype.git
cd football-media-hype
pip install soccerdata pytrends pandas numpy requests transformers torch wikipedia-api statsmodels matplotlib seaborn
```

## 3. Run the pipeline
```bash
export GUARDIAN_API_KEY="your_key_here"
bash src/run_all.sh
```

Or step by step:
```bash
python3 src/01_player_population.py   # ~15 min (FBRef, 14 seasons)
python3 src/02_hype_index.py          # ~30 min (Trends + Guardian + Wikipedia)
python3 src/03_nlp_sentiment.py       # ~10 min (NLP model download + inference)
python3 src/04_build_panel.py         # ~1 min
python3 src/05_analysis.py            # ~1 min
```

## 4. Outputs
| File | Description |
|------|-------------|
| `data/processed/panel_final.csv` | Final player × season panel dataset |
| `outputs/figures/hype_trajectories.png` | Mean hype over time by nationality |
| `outputs/figures/event_study.png` | Performance around hype spikes |
| `outputs/tables/regression_results.csv` | Key regression coefficients |

## Notes
- FBRef scraping: the script is polite (4s delay per season). Do not reduce this — FBRef will block you.
- Google Trends: 1.5s delay per chunk. Again, do not reduce.
- The NLP model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) downloads ~500MB on first run and then caches locally.
- Wikipedia API only goes back to July 2015 — pre-2015 wiki data will be NaN.
