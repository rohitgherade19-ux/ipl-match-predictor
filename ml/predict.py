"""
Prediction module — loads trained model and provides utility functions.
"""

import os
import sys
import joblib
import pandas as pd
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from database.db_setup import get_db_path
from ml.feature_engineering import compute_live_features

_model = None  # cache loaded model


def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')
        _model = joblib.load(model_path)
    return _model


def predict_winner(team1, team2, venue, toss_winner, toss_decision):
    """
    Predict the winner of a match.
    Returns a dict with predicted_winner, team1, team2, team1_win_prob, team2_win_prob.
    """
    model = get_model()
    features = compute_live_features(team1, team2, venue, toss_winner, toss_decision)

    prob = model.predict_proba(features)[0]
    pred = model.predict(features)[0]

    # Class 0 = team2 wins, Class 1 = team1 wins
    team1_win_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
    team2_win_prob = float(prob[0]) if len(prob) > 1 else 0.0

    predicted_winner = team1 if pred == 1 else team2

    return {
        'predicted_winner': predicted_winner,
        'team1': team1,
        'team2': team2,
        'team1_win_prob': team1_win_prob,
        'team2_win_prob': team2_win_prob,
    }


def get_all_teams():
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT team FROM ("
        "  SELECT team1 AS team FROM matches UNION SELECT team2 AS team FROM matches"
        ") WHERE team IS NOT NULL ORDER BY team"
    )
    teams = [row[0] for row in cursor.fetchall()]
    conn.close()
    return teams


def get_all_venues():
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT venue FROM matches WHERE venue IS NOT NULL ORDER BY venue")
    venues = [row[0] for row in cursor.fetchall()]
    conn.close()
    return venues


def get_head_to_head(team1, team2):
    """
    Return head-to-head stats as a dict with keys: team1_wins, team2_wins, total.
    """
    conn = sqlite3.connect(get_db_path())
    query = """
    SELECT winner, COUNT(*) as wins
    FROM matches
    WHERE ((team1 = ? AND team2 = ?) OR (team1 = ? AND team2 = ?))
      AND winner IS NOT NULL
    GROUP BY winner
    """
    cursor = conn.cursor()
    cursor.execute(query, (team1, team2, team2, team1))
    rows = cursor.fetchall()
    conn.close()

    result = {'team1_wins': 0, 'team2_wins': 0, 'total': 0}
    for row in rows:
        winner, wins = row
        if winner == team1:
            result['team1_wins'] = wins
        elif winner == team2:
            result['team2_wins'] = wins
    result['total'] = result['team1_wins'] + result['team2_wins']
    return result


def get_team_stats(team):
    """Return overall stats for a team."""
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) as total, "
        "SUM(CASE WHEN winner = ? THEN 1 ELSE 0 END) as wins "
        "FROM matches WHERE (team1 = ? OR team2 = ?) AND winner IS NOT NULL",
        (team, team, team)
    )
    row = cursor.fetchone()
    conn.close()

    if row and row[0] > 0:
        total, wins = row
        return {'total_matches': total, 'wins': wins, 'win_pct': round(wins / total * 100, 1)}
    return {'total_matches': 0, 'wins': 0, 'win_pct': 0.0}


if __name__ == '__main__':
    print("Testing prediction...")
    result = predict_winner('Chennai Super Kings', 'Mumbai Indians',
                            'Wankhede Stadium, Mumbai', 'Mumbai Indians', 'field')
    print(result)
