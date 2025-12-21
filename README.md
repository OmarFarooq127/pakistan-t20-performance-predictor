# Pakistan T20 Performance Intelligence

This project builds and visualizes performance intelligence (PI) for Pakistan men’s T20 matches.

## What it does
- **Phase 1 – Engine** (`pipeline.py`)
  - Filters male matches, parses ball-by-ball JSONs, constructs player PIs (Bat/Bowl/Field), rolling 10-match forms, team aggregates, and country baselines.
  - Outputs to `outputs/`:
    - `player_match_pi.csv` – player-level PIs + forms
    - `player_raw_audit.csv` – raw stats (runs, wickets, overs) for audit
    - `team_match_aggregates.csv` – match-level team sums, forms, Top3/Top2
    - `country_baseline.csv` – country W/L ratio and average PIs
    - `layer1_metrics.json` – PI → Win verification metrics
    - `pre_match_predictions.csv` – Pakistan pre-match predicted PIs + win flag
- **Phase 2 – Brain** (`layer1_verification`, `layer2_pre_match` inside pipeline)
  - Layer 1: Logistic regression checks if team PIs + chase predict victory (82.7% acc in sample run).
  - Layer 2: RandomForest regressors predict Pakistan Bat/Bowl/Field PIs pre-match, then compare vs opponent baseline + environmental weights to flag expected win.
- **Phase 3 – Frontend** (`app.py`)
  - Streamlit dashboard to present metrics, country baselines, team aggregates, and Pakistan pre-match predictions with simple charts.

## Quick start
1) **Create/activate venv (if not already)**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```
2) **Install deps**
```bash
pip install -r requirements.txt
```
3) **Run the pipeline** (creates `outputs/` files)
```bash
python pipeline.py
```
4) **Launch the dashboard**
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## Using the dashboard
- **Layer 1 Verification**: Shows Accuracy/Precision/Recall proving PI is predictive of wins.
- **Country Baseline Table**: W/L ratio and average PIs per country.
 - **Country Baseline Table**: W/L ratio and average PIs per country. Toggle "Popular countries only" to focus on major teams (India, Australia, England, Pakistan, South Africa, New Zealand, Sri Lanka, Bangladesh, West Indies, Afghanistan). When showing all, popular teams are sorted to the top.
- **Team Aggregate PIs**: Latest team-level PI and form snapshots.
- **Pakistan Pre-Match Predictions**: Select opponent to see predicted Bat/Bowl/Field PIs, opponent baseline, environmental weight, predicted win flag, and trend chart of Pred_Total_PI vs Opp_Total_PI.
  - Use the "Popular opponents only" toggle to focus on major teams in the selector.

## Key formulas
- **Bat_PI** = (Runs + Fours + Sixes) × (StrikeRate/100)
- **Bowl_PI** = (Wickets × 25 + Maidens × 10) × min(10, 6 / Economy)
- **Field_PI** = 8 × Catches + 12 × Stumpings + 10 × RunOuts
- **Form features**: 10-match rolling mean per player, shifted by 1 (no leakage).
- **Environmental weight**: 10 × Chase + 5 × (not Toss_Win) + 3 × Home_Advantage.

## Troubleshooting
- If files are locked (e.g., open in Excel), close them and re-run `python pipeline.py`.
- If `pre_match_predictions.csv` is missing, ensure the pipeline completed without errors.
- For speed, keep `t20s_json/` local; network drives slow parsing.

## Project layout
- `pipeline.py` – data prep + modeling
- `app.py` – Streamlit UI
- `outputs/` – generated artifacts
- `t20s_json/` – raw match JSONs (input)
- `Match List.txt` – match metadata list (input)

## Notes
- All processing is male matches only (as filtered from the match list).
- PIs are heuristic but validated via Layer 1; tune weights in `process_match` if needed.
