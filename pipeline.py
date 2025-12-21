"""
Pipeline to build Pakistan T20 performance intelligence features.
Phase 1: data prep (male matches, PI construction, forms, team aggregates, baselines).
Phase 2: lightweight models (verification + pre-match PI predictors).
Outputs are written to the outputs/ directory.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
MATCH_LIST_PATH = ROOT / "Match List.txt"
SOURCE_JSON_DIR = ROOT / "t20s_json"
MALE_JSON_DIR = ROOT / "male_matches_json"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)
MALE_JSON_DIR.mkdir(exist_ok=True)

# Tunable environmental weights (reduced heuristics)
ENV_WEIGHTS = {
    "CHASE": 7,           # previously 10
    "NOT_TOSS_WIN": 3,   # previously 5
    "HOME": 2,           # previously 3
}


def parse_match_list(match_list_path: Path) -> pd.DataFrame:
    rows = []
    with match_list_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(" - ")]
            if len(parts) < 6:
                continue
            date_str, level, fmt, gender, match_id, fixture = parts[:6]
            if gender.lower() != "male":
                continue
            rows.append(
                {
                    "Date": date_str,
                    "Level": level,
                    "Format": fmt,
                    "Gender": gender,
                    "Match_ID": match_id,
                    "Fixture": fixture,
                }
            )
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def copy_male_jsons(match_ids: pd.Series) -> None:
    copied = 0
    for mid in match_ids.unique():
        src = SOURCE_JSON_DIR / f"{mid}.json"
        dst = MALE_JSON_DIR / f"{mid}.json"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"Copied {copied} male JSON files into {MALE_JSON_DIR.name}/")


def pitch_attributes(venue: str, city: str) -> Dict[str, str]:
    v = (venue or "").lower()
    c = (city or "").lower()
    attr = {
        "Soil_Type": "Loam",
        "Pace_Bounce": "Medium",
        "Seam_Swing": "Medium",
        "Spin_Potential": "Minimal",
        "Abrasion_Rate": "Low",
    }
    if any(x in v or x in c for x in ["perth", "brisbane", "melbourne", "sydney", "hobart", "adelaide", "canberra", "geelong"]):
        attr.update({
            "Soil_Type": "Black Clay",
            "Pace_Bounce": "High",
            "Seam_Swing": "Medium",
            "Spin_Potential": "Late",
            "Abrasion_Rate": "Medium",
        })
    if any(x in v or x in c for x in ["dubai", "abu dhabi", "sharjah", "lahore", "karachi", "rawalpindi", "multan", "mirpur", "chattogram", "chennai", "mumbai", "kolkata", "dhaka", "colombo", "galle", "pallekele"]):
        attr.update({
            "Soil_Type": "Silt/Clay",
            "Pace_Bounce": "Low-Medium",
            "Seam_Swing": "Low",
            "Spin_Potential": "High",
            "Abrasion_Rate": "High",
        })
    if any(x in v or x in c for x in ["london", "lord's", "edgbaston", "trent bridge", "headingley", "oval", "cardiff", "manchester", "southampton", "bristol", "dublin", "belfast", "bready"]):
        attr.update({
            "Soil_Type": "Loam",
            "Pace_Bounce": "Medium",
            "Seam_Swing": "High",
            "Spin_Potential": "Late",
            "Abrasion_Rate": "Low",
        })
    if any(x in v or x in c for x in ["wanderers", "centurion", "newlands", "kingsmead", "boland", "paarl", "port elizabeth", "bloemfontein"]):
        attr.update({
            "Soil_Type": "Red Clay",
            "Pace_Bounce": "High",
            "Seam_Swing": "Medium-High",
            "Spin_Potential": "Medium",
            "Abrasion_Rate": "Medium",
        })
    if any(x in v or x in c for x in ["barbados", "trinidad", "guyana", "st kitts", "grenada", "antigua", "jamaica"]):
        attr.update({
            "Soil_Type": "Clay",
            "Pace_Bounce": "Low-Medium",
            "Seam_Swing": "Medium",
            "Spin_Potential": "Medium",
            "Abrasion_Rate": "Medium",
        })
    return attr


def infer_country_from_venue(venue: str, city: str) -> str:
    v = (venue or "").lower()
    c = (city or "").lower()
    mapping = {
        "karachi": "Pakistan",
        "lahore": "Pakistan",
        "rawalpindi": "Pakistan",
        "multan": "Pakistan",
        "islamabad": "Pakistan",
        "dubai": "United Arab Emirates",
        "abu dhabi": "United Arab Emirates",
        "sharjah": "United Arab Emirates",
        "melbourne": "Australia",
        "sydney": "Australia",
        "brisbane": "Australia",
        "adelaide": "Australia",
        "perth": "Australia",
        "hobart": "Australia",
        "auckland": "New Zealand",
        "wellington": "New Zealand",
        "christchurch": "New Zealand",
        "london": "England",
        "manchester": "England",
        "birmingham": "England",
        "leeds": "England",
        "cardiff": "England",
        "nottingham": "England",
        "cape town": "South Africa",
        "johannesburg": "South Africa",
        "centurion": "South Africa",
        "durban": "South Africa",
        "port elizabeth": "South Africa",
        "colombo": "Sri Lanka",
        "galle": "Sri Lanka",
        "pallekele": "Sri Lanka",
        "mumbai": "India",
        "chennai": "India",
        "delhi": "India",
        "hyderabad": "India",
        "ahmedabad": "India",
        "dhaka": "Bangladesh",
        "chattogram": "Bangladesh",
        "mirpur": "Bangladesh",
        "barbados": "West Indies",
        "trinidad": "West Indies",
        "guyana": "West Indies",
        "jamaica": "West Indies",
    }
    for key, country in mapping.items():
        if key in v or key in c:
            return country
    return "Unknown"


def is_legal_delivery(delivery: Dict) -> bool:
    extras = delivery.get("extras", {})
    return not ("wides" in extras or "noballs" in extras)


def empty_player(team: str) -> Dict:
    return {
        "Team": team,
        "Runs": 0,
        "Balls_Faced": 0,
        "Fours": 0,
        "Sixes": 0,
        "Wickets": 0,
        "Balls_Bowled": 0,
        "Runs_Conceded": 0,
        "Maidens": 0,
        "Catches": 0,
        "Stumpings": 0,
        "Run_Outs": 0,
    }


def process_match(file_path: Path) -> Tuple[List[Dict], Dict]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    if info.get("gender") != "male":
        return [], {}

    match_id = file_path.stem
    date = info.get("dates", [None])[0]
    venue = info.get("venue", "Unknown")
    city = info.get("city", "Unknown")
    teams = info.get("teams", [])
    toss = info.get("toss", {})
    toss_winner = toss.get("winner")
    toss_decision = toss.get("decision")
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    neutral = info.get("neutral_venue", 0)

    players_by_team = info.get("players", {})
    player_stats: Dict[str, Dict] = {}
    for team, roster in players_by_team.items():
        for p in roster:
            player_stats[p] = empty_player(team)

    over_runs = defaultdict(int)
    innings = data.get("innings", [])
    chasing_team = innings[1].get("team") if len(innings) > 1 else None

    for inning in innings:
        batting_team = inning.get("team")
        fielding_team = [t for t in teams if t != batting_team]
        fielding_team = fielding_team[0] if fielding_team else batting_team
        for over in inning.get("overs", []):
            over_no = over.get("over", -1)
            for delivery in over.get("deliveries", []):
                batter = delivery.get("batter")
                bowler = delivery.get("bowler")
                runs = delivery.get("runs", {})
                batter_runs = runs.get("batter", 0)
                total_runs = runs.get("total", 0)
                legal = is_legal_delivery(delivery)

                if batter not in player_stats:
                    player_stats[batter] = empty_player(batting_team)
                if bowler not in player_stats:
                    player_stats[bowler] = empty_player(fielding_team)

                bstat = player_stats[batter]
                bstat["Runs"] += batter_runs
                if legal:
                    bstat["Balls_Faced"] += 1
                if batter_runs == 4:
                    bstat["Fours"] += 1
                if batter_runs == 6:
                    bstat["Sixes"] += 1

                bowl_stat = player_stats[bowler]
                bowl_stat["Runs_Conceded"] += total_runs
                if legal:
                    bowl_stat["Balls_Bowled"] += 1
                    over_runs[(bowler, over_no, fielding_team)] += total_runs

                for wicket in delivery.get("wickets", []):
                    kind = wicket.get("kind", "")
                    if kind not in ["run out", "retired hurt", "obstructing the field"]:
                        bowl_stat["Wickets"] += 1
                    for f in wicket.get("fielders", []):
                        fname = f.get("name")
                        if fname not in player_stats:
                            player_stats[fname] = empty_player(fielding_team)
                        fstat = player_stats[fname]
                        if kind == "catch":
                            fstat["Catches"] += 1
                        elif kind == "stumping":
                            fstat["Stumpings"] += 1
                        elif kind == "run out":
                            fstat["Run_Outs"] += 1

    for (bowler, over_no, team), runs in over_runs.items():
        if runs == 0:
            player_stats[bowler]["Maidens"] += 1

    rows = []
    pitch_attr = pitch_attributes(venue, city)
    host_country = infer_country_from_venue(venue, city)
    opponent_lookup = {}
    if len(teams) == 2:
        opponent_lookup[teams[0]] = teams[1]
        opponent_lookup[teams[1]] = teams[0]

    for player, stats in player_stats.items():
        runs = stats["Runs"]
        balls_faced = stats["Balls_Faced"] or 1
        strike_rate = runs * 100.0 / balls_faced
        bat_pi = (runs + stats["Fours"] + stats["Sixes"]) * (strike_rate / 100.0)

        balls_bowled = stats["Balls_Bowled"]
        overs_float = balls_bowled / 6.0 if balls_bowled else 0.0
        economy = stats["Runs_Conceded"] / overs_float if overs_float else 0.0
        bowl_base = stats["Wickets"] * 25 + stats["Maidens"] * 10
        econ_adj = 6.0 / economy if economy > 0 else 0.0
        econ_adj = min(econ_adj, 10)
        bowl_pi = bowl_base * (econ_adj if econ_adj > 0 else 1.0)

        field_pi = stats["Catches"] * 8 + stats["Stumpings"] * 12 + stats["Run_Outs"] * 10

        team = stats["Team"]
        rows.append(
            {
                "Match_ID": match_id,
                "Date": pd.to_datetime(date),
                "Team": team,
                "Opponent": opponent_lookup.get(team, "Unknown"),
                "Player": player,
                "Venue": venue,
                "City": city,
                "Neutral_Venue": neutral,
                "Toss_Win": 1 if team == toss_winner else 0,
                "Chase": 1 if team == chasing_team else 0,
                "Home_Advantage": 1 if host_country == team else 0,
                "Winner": 1 if winner == team else 0,
                "Runs": runs,
                "Balls_Faced": stats["Balls_Faced"],
                "Fours": stats["Fours"],
                "Sixes": stats["Sixes"],
                "Wickets": stats["Wickets"],
                "Balls_Bowled": balls_bowled,
                "Overs": round(overs_float, 2),
                "Runs_Conceded": stats["Runs_Conceded"],
                "Maidens": stats["Maidens"],
                "Catches": stats["Catches"],
                "Stumpings": stats["Stumpings"],
                "Run_Outs": stats["Run_Outs"],
                "Bat_PI": round(bat_pi, 2),
                "Bowl_PI": round(bowl_pi, 2),
                "Field_PI": field_pi,
                "Pace_Bounce": pitch_attr["Pace_Bounce"],
                "Seam_Swing": pitch_attr["Seam_Swing"],
                "Spin_Potential": pitch_attr["Spin_Potential"],
                "Abrasion_Rate": pitch_attr["Abrasion_Rate"],
                "Soil_Type": pitch_attr["Soil_Type"],
            }
        )

    meta = {
        "Match_ID": match_id,
        "Date": pd.to_datetime(date),
        "Teams": teams,
        "Venue": venue,
        "City": city,
        "Chasing_Team": chasing_team,
        "Toss_Winner": toss_winner,
        "Toss_Decision": toss_decision,
        "Winner": winner,
    }
    return rows, meta


def build_player_level_df(json_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_rows: List[Dict] = []
    meta_rows: List[Dict] = []
    for fp in json_dir.glob("*.json"):
        rows, meta = process_match(fp)
        if rows:
            player_rows.extend(rows)
            meta_rows.append(meta)

    player_df = pd.DataFrame(player_rows)
    meta_df = pd.DataFrame(meta_rows)

    audit_cols = [
        "Match_ID",
        "Date",
        "Team",
        "Player",
        "Runs",
        "Balls_Faced",
        "Wickets",
        "Overs",
        "Runs_Conceded",
    ]
    audit_df = player_df[audit_cols].copy()

    return player_df, audit_df, meta_df


def add_player_forms(player_df: pd.DataFrame) -> pd.DataFrame:
    df = player_df.sort_values(["Player", "Date", "Match_ID"]).copy()
    for col, out_col in [("Bat_PI", "Bat_Form"), ("Bowl_PI", "Bowl_Form"), ("Field_PI", "Field_Form")]:
        rolling = (
            df.groupby("Player")[col]
            .apply(lambda s: s.shift(1).rolling(window=10, min_periods=1).mean())
        )
        df[out_col] = rolling.reset_index(level=0, drop=True)
        df[out_col].fillna(df[col].mean(), inplace=True)
    return df


def aggregate_team_level(player_df: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for (match_id, team), grp in player_df.groupby(["Match_ID", "Team"]):
        first_row = grp.iloc[0]
        team_row = {
            "Match_ID": match_id,
            "Date": first_row["Date"],
            "Team": team,
            "Opponent": first_row["Opponent"],
            "Venue": first_row["Venue"],
            "City": first_row["City"],
            "Toss_Win": int(first_row["Toss_Win"]),
            "Chase": int(first_row["Chase"]),
            "Home_Advantage": int(first_row["Home_Advantage"]),
            "Winner": int(first_row["Winner"]),
            "Pace_Bounce": first_row["Pace_Bounce"],
            "Seam_Swing": first_row["Seam_Swing"],
            "Spin_Potential": first_row["Spin_Potential"],
            "Abrasion_Rate": first_row["Abrasion_Rate"],
            "Soil_Type": first_row["Soil_Type"],
            "Team_Bat_PI": grp["Bat_PI"].sum(),
            "Team_Bowl_PI": grp["Bowl_PI"].sum(),
            "Team_Field_PI": grp["Field_PI"].sum(),
            "Team_Bat_Form": grp["Bat_Form"].sum(),
            "Team_Bowl_Form": grp["Bowl_Form"].sum(),
            "Team_Field_Form": grp["Field_Form"].sum(),
            "Top3_Bat_Form_Sum": grp["Bat_Form"].dropna().nlargest(3).sum() if len(grp["Bat_Form"].dropna()) > 0 else 0,
            "Top2_Bowl_Form_Sum": grp["Bowl_Form"].dropna().nlargest(2).sum() if len(grp["Bowl_Form"].dropna()) > 0 else 0,
        }
        groups.append(team_row)
    team_df = pd.DataFrame(groups)
    team_df.sort_values(["Team", "Date", "Match_ID"], inplace=True)

    team_df["Prior_Matches"] = team_df.groupby("Team").cumcount()
    team_df["Prior_Wins"] = team_df.groupby("Team")["Winner"].cumsum() - team_df["Winner"]
    team_df["Prior_WL_Ratio"] = team_df["Prior_Wins"] / team_df["Prior_Matches"].replace(0, np.nan)
    for col, out_col in [
        ("Team_Bat_PI", "Prior_Avg_Bat_PI"),
        ("Team_Bowl_PI", "Prior_Avg_Bowl_PI"),
        ("Team_Field_PI", "Prior_Avg_Field_PI"),
    ]:
        rolling = (
            team_df.groupby("Team")[col]
            .expanding()
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        team_df[out_col] = rolling
    overall_means = {
        "Prior_WL_Ratio": team_df["Winner"].mean(),
        "Prior_Avg_Bat_PI": team_df["Team_Bat_PI"].mean(),
        "Prior_Avg_Bowl_PI": team_df["Team_Bowl_PI"].mean(),
        "Prior_Avg_Field_PI": team_df["Team_Field_PI"].mean(),
    }
    for col, default in overall_means.items():
        team_df[col].fillna(default, inplace=True)

    opp_baseline = team_df[
        [
            "Match_ID",
            "Team",
            "Prior_WL_Ratio",
            "Prior_Avg_Bat_PI",
            "Prior_Avg_Bowl_PI",
            "Prior_Avg_Field_PI",
        ]
    ].rename(
        columns={
            "Team": "Opponent",
            "Prior_WL_Ratio": "Opp_WL_Ratio",
            "Prior_Avg_Bat_PI": "Opp_Avg_Bat_PI",
            "Prior_Avg_Bowl_PI": "Opp_Avg_Bowl_PI",
            "Prior_Avg_Field_PI": "Opp_Avg_Field_PI",
        }
    )

    team_df = team_df.merge(opp_baseline, on=["Match_ID", "Opponent"], how="left")
    for col in ["Opp_WL_Ratio", "Opp_Avg_Bat_PI", "Opp_Avg_Bowl_PI", "Opp_Avg_Field_PI"]:
        team_df[col].fillna(team_df[col].mean(), inplace=True)

    return team_df


def build_country_baseline(team_df: pd.DataFrame) -> pd.DataFrame:
    return (
        team_df.groupby("Team")
        .agg(
            Matches=("Match_ID", "count"),
            Wins=("Winner", "sum"),
            WL_Ratio=("Winner", "mean"),
            Avg_Bat_PI=("Team_Bat_PI", "mean"),
            Avg_Bowl_PI=("Team_Bowl_PI", "mean"),
            Avg_Field_PI=("Team_Field_PI", "mean"),
        )
        .reset_index()
    )


def layer1_verification(team_df: pd.DataFrame) -> Dict:
    features = team_df[["Team_Bat_PI", "Team_Bowl_PI", "Team_Field_PI", "Chase"]]
    target = team_df["Winner"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 3),
        "precision": round(precision_score(y_test, preds), 3),
        "recall": round(recall_score(y_test, preds), 3),
    }
    pd.DataFrame(metrics, index=[0]).to_json(OUTPUT_DIR / "layer1_metrics.json", orient="records", indent=2)
    return metrics


def layer2_pre_match(team_df: pd.DataFrame) -> pd.DataFrame:
    pak_df = team_df[team_df["Team"] == "Pakistan"].copy()
    feature_cols = [
        "Team_Bat_Form",
        "Team_Bowl_Form",
        "Team_Field_Form",
        "Top3_Bat_Form_Sum",
        "Top2_Bowl_Form_Sum",
        "Pace_Bounce",
        "Seam_Swing",
        "Spin_Potential",
        "Abrasion_Rate",
        "Soil_Type",
        "Toss_Win",
        "Chase",
        "Home_Advantage",
        "Opp_WL_Ratio",
        "Opp_Avg_Bat_PI",
        "Opp_Avg_Bowl_PI",
        "Opp_Avg_Field_PI",
    ]

    cat_cols = ["Pace_Bounce", "Seam_Swing", "Spin_Potential", "Abrasion_Rate", "Soil_Type"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    targets = {
        "Pred_Bat_PI": "Team_Bat_PI",
        "Pred_Bowl_PI": "Team_Bowl_PI",
        "Pred_Field_PI": "Team_Field_PI",
    }

    results = pak_df[["Match_ID", "Date", "Opponent"] + feature_cols].copy()
    mae_scores: Dict[str, float] = {}

    for out_col, target_col in targets.items():
        y = pak_df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(pak_df[feature_cols], y, test_size=0.2, random_state=42)
        model = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=150,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mae_scores[out_col] = float(mae)
        print(f"{out_col} MAE: {mae:.2f}")
        results[out_col] = model.predict(pak_df[feature_cols])

    results["Pred_Total_PI"] = results[["Pred_Bat_PI", "Pred_Bowl_PI", "Pred_Field_PI"]].sum(axis=1)
    
    # Add opponent baseline and match metadata
    opp_data = pak_df[[
        "Match_ID",
        "Opp_Avg_Bat_PI",
        "Opp_Avg_Bowl_PI",
        "Opp_Avg_Field_PI",
        "Chase",
        "Toss_Win",
        "Home_Advantage",
        "Winner",
    ]].drop_duplicates(subset=["Match_ID"])
    
    results = results.merge(opp_data, on="Match_ID", how="left", suffixes=("", "_opp"))
    
    results["Opp_Total_PI"] = results["Opp_Avg_Bat_PI"] + results["Opp_Avg_Bowl_PI"] + results["Opp_Avg_Field_PI"]
    results["Environmental_Weight"] = (
        ENV_WEIGHTS["CHASE"] * results["Chase"]
        + ENV_WEIGHTS["NOT_TOSS_WIN"] * (1 - results["Toss_Win"])
        + ENV_WEIGHTS["HOME"] * results["Home_Advantage"]
    )
    results["Pred_Win"] = (results["Pred_Total_PI"] > (results["Opp_Total_PI"] + results["Environmental_Weight"])).astype(int)
    results["Correct"] = (results["Pred_Win"] == results["Winner"]).astype(int)
    
    # Layer 2 performance summary
    acc = accuracy_score(results["Winner"], results["Pred_Win"]) if len(results) else 0.0
    prec = precision_score(results["Winner"], results["Pred_Win"]) if results["Pred_Win"].sum() > 0 else 0.0
    rec = recall_score(results["Winner"], results["Pred_Win"]) if results["Winner"].sum() > 0 else 0.0
    layer2_metrics = {
        "accuracy": round(float(acc), 3),
        "precision": round(float(prec), 3),
        "recall": round(float(rec), 3),
        "mae_pred_bat": round(mae_scores.get("Pred_Bat_PI", 0.0), 2),
        "mae_pred_bowl": round(mae_scores.get("Pred_Bowl_PI", 0.0), 2),
        "mae_pred_field": round(mae_scores.get("Pred_Field_PI", 0.0), 2),
    }
    pd.DataFrame([layer2_metrics]).to_json(OUTPUT_DIR / "layer2_metrics.json", orient="records", indent=2)
    return results


def main() -> None:
    print("Parsing match list...")
    match_df = parse_match_list(MATCH_LIST_PATH)
    copy_male_jsons(match_df["Match_ID"])

    print("Processing JSON scorecards...")
    player_df, audit_df, _meta_df = build_player_level_df(MALE_JSON_DIR)

    print("Adding player rolling form (shifted, last 10 matches)...")
    player_df = add_player_forms(player_df)

    player_df.to_csv(OUTPUT_DIR / "player_match_pi.csv", index=False)
    audit_df.to_csv(OUTPUT_DIR / "player_raw_audit.csv", index=False)

    print("Building team-level aggregates...")
    team_df = aggregate_team_level(player_df)
    team_df.to_csv(OUTPUT_DIR / "team_match_aggregates.csv", index=False)

    print("Building country baseline table (no leakage in per-match features)...")
    country_baseline = build_country_baseline(team_df)
    country_baseline.to_csv(OUTPUT_DIR / "country_baseline.csv", index=False)

    print("Layer 1 verification: PIs vs victory...")
    metrics = layer1_verification(team_df)
    print(f"Layer 1 metrics: {metrics}")

    print("Layer 2 pre-match PI predictors for Pakistan...")
    pre_match = layer2_pre_match(team_df)
    pre_match.to_csv(OUTPUT_DIR / "pre_match_predictions.csv", index=False)

    print("Done. Artifacts saved to outputs/.")


if __name__ == "__main__":
    main()
