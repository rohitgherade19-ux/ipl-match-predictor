"""
Flask Web Application — IPL Match Winner Prediction System.
"""

import os
import sys
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ml.predict import predict_winner, get_all_teams, get_all_venues, get_head_to_head, get_team_stats
from database.db_setup import get_db_path

app = Flask(__name__)
app.secret_key = 'ipl-predictor-secret-key-2024'


def get_db():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    teams = get_all_teams()
    venues = get_all_venues()
    return render_template('index.html', teams=teams, venues=venues)


@app.route('/predict', methods=['POST'])
def predict():
    team1 = request.form.get('team1')
    team2 = request.form.get('team2')
    venue = request.form.get('venue')
    toss_winner = request.form.get('toss_winner')
    toss_decision = request.form.get('toss_decision')

    if team1 == team2:
        flash("Team 1 and Team 2 cannot be the same.", "warning")
        return redirect(url_for('index'))

    prediction = predict_winner(team1, team2, venue, toss_winner, toss_decision)
    h2h = get_head_to_head(team1, team2)

    confidence = max(prediction.get('team1_win_prob', 0), prediction.get('team2_win_prob', 0))

    # Save to database
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (team1, team2, venue, toss_winner, toss_decision, predicted_winner, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (team1, team2, venue, toss_winner, toss_decision, prediction['predicted_winner'], confidence))
        conn.commit()
    except Exception as e:
        print(f"Error saving prediction: {e}")
    finally:
        conn.close()

    return render_template(
        'result.html',
        prediction=prediction,
        h2h=h2h,
        team1=team1,
        team2=team2,
        venue=venue,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
    )


@app.route('/stats', methods=['GET'])
def stats():
    conn = get_db()

    # ---- Team Stats ----
    team_stats = []
    teams_query = conn.execute(
        "SELECT DISTINCT team FROM ("
        "  SELECT team1 AS team FROM matches UNION SELECT team2 AS team FROM matches"
        ") WHERE team IS NOT NULL ORDER BY team"
    ).fetchall()

    for row in teams_query:
        team = row['team']
        r = conn.execute(
            "SELECT COUNT(*) as played, "
            "SUM(CASE WHEN winner = ? THEN 1 ELSE 0 END) as wins "
            "FROM matches WHERE (team1 = ? OR team2 = ?) AND winner IS NOT NULL",
            (team, team, team)
        ).fetchone()
        played = r['played']
        wins = r['wins']
        losses = played - wins

        titles = conn.execute(
            "SELECT COUNT(*) as cnt FROM matches WHERE match_type = 'Final' AND winner = ?",
            (team,)
        ).fetchone()['cnt']

        win_pct = round(wins / played * 100, 1) if played > 0 else 0
        team_stats.append({
            'team': team,
            'matches': played,
            'wins': wins,
            'losses': losses,
            'win_percentage': win_pct,
            'titles': titles,
        })

    # Sort by win %
    team_stats.sort(key=lambda x: x['win_percentage'], reverse=True)

    # ---- Top Batsmen ----
    top_batsmen_raw = conn.execute('''
        SELECT batter,
               SUM(batsman_runs) AS total_runs,
               COUNT(DISTINCT match_id) AS matches,
               ROUND(SUM(batsman_runs) * 1.0 / COUNT(DISTINCT match_id), 2) AS average,
               ROUND(SUM(batsman_runs) * 100.0 / COUNT(*), 2) AS strike_rate
        FROM deliveries
        WHERE batsman_runs >= 0
        GROUP BY batter
        HAVING total_runs > 500
        ORDER BY total_runs DESC
        LIMIT 15
    ''').fetchall()

    top_batsmen = []
    for i, r in enumerate(top_batsmen_raw, 1):
        top_batsmen.append({
            'rank': i,
            'player': r['batter'],
            'runs': r['total_runs'],
            'matches': r['matches'],
            'average': r['average'],
            'strike_rate': r['strike_rate'],
        })

    # ---- Top Bowlers ----
    top_bowlers_raw = conn.execute('''
        SELECT bowler,
               SUM(CASE WHEN is_wicket = 1 AND dismissal_kind NOT IN ('run out', 'retired hurt', 'obstructing the field') THEN 1 ELSE 0 END) AS wickets,
               COUNT(DISTINCT match_id) AS matches,
               ROUND(SUM(total_runs) * 6.0 / COUNT(*), 2) AS economy
        FROM deliveries
        GROUP BY bowler
        HAVING wickets > 30
        ORDER BY wickets DESC
        LIMIT 15
    ''').fetchall()

    top_bowlers = []
    for i, r in enumerate(top_bowlers_raw, 1):
        top_bowlers.append({
            'rank': i,
            'player': r['bowler'],
            'wickets': r['wickets'],
            'matches': r['matches'],
            'economy': r['economy'],
        })

    conn.close()

    return render_template('stats.html', team_stats=team_stats, top_batsmen=top_batsmen, top_bowlers=top_bowlers)


@app.route('/history', methods=['GET'])
def history():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()

    predictions = []
    for r in rows:
        predictions.append({
            'id': r['id'],
            'team1': r['team1'],
            'team2': r['team2'],
            'venue': r['venue'],
            'toss_winner': r['toss_winner'],
            'toss_decision': r['toss_decision'],
            'predicted_winner': r['predicted_winner'],
            'confidence': r['confidence'],
            'date': r['timestamp'],
        })

    return render_template('history.html', predictions=predictions)


@app.route('/api/teams', methods=['GET'])
def api_teams():
    return jsonify(get_all_teams())


@app.route('/api/venues', methods=['GET'])
def api_venues():
    return jsonify(get_all_venues())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
