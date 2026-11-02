#!/bin/bash
# Run the full pipeline in sequence.
# Set your Guardian API key first: export GUARDIAN_API_KEY="your_key_here"

set -e

echo "========================================"
echo "Football Media Hype — Full Pipeline"
echo "========================================"

echo ""
echo "Step 1: Pull player population from FBRef"
python3 src/01_player_population.py

echo ""
echo "Step 2: Build hype index (Trends + Guardian + Wikipedia)"
python3 src/02_hype_index.py

echo ""
echo "Step 3: NLP sentiment on Guardian headlines"
python3 src/03_nlp_sentiment.py

echo ""
echo "Step 4: Build merged panel dataset"
python3 src/04_build_panel.py

echo ""
echo "Step 5: Analysis — regression + event study"
pip install statsmodels -q
python3 src/05_analysis.py

echo ""
echo "========================================"
echo "DONE. Outputs in:"
echo "  data/processed/panel_final.csv"
echo "  outputs/figures/hype_trajectories.png"
echo "  outputs/figures/event_study.png"
echo "  outputs/tables/regression_results.csv"
echo "========================================"
