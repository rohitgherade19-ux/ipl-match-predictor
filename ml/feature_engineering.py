"""
Feature Engineering for IPL Match Winner Prediction — v4 (Elo-based).

Key innovation: Elo rating system computed chronologically for each team,
supplemented with venue-specific Elo, toss advantage, and momentum features.
Elo is purpose-built for predicting competitive outcomes.
"""

import os
import sys
import pandas as pd
import numpy as np
import sqlite3
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from database.db_setup import get_db_path

# ---------------------------------------------------------------------------
#  Elo Rating System
# ---------------------------------------------------------------------------
INITIAL_ELO = 1500
K_FACTOR = 32
SEASON_DECAY = 0.15  # Regress toward mean between seasons


def _expected_score(elo_a, elo_b):
    """Expected probability that A wins."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _update_elo(elo_winner, elo_loser, k=K_FACTOR, margin=None):
    """Update Elo ratings. Optional margin-of-victory multiplier."""
    expected_w = _expected_score(elo_winner, elo_loser)
    expected_l = 1.0 - expected_w

    # Margin-of-victory multiplier (log-scaled)
    if margin is not None and margin > 0:
        mov_mult = np.log(1 + margin) * 0.8
        mov_mult = max(mov_mult, 0.5)
        mov_mult = min(mov_mult, 2.5)
    else:
        mov_mult = 1.0

    new_winner = elo_winner + k * mov_mult * (1 - expected_w)
    new_loser = elo_loser + k * mov_mult * (0 - expected_l)
    return new_winner, new_loser


def _season_reset(elo, initial=INITIAL_ELO, decay=SEASON_DECAY):
    """Regress Elo toward mean between seasons to account for roster changes."""
    return elo * (1 - decay) + initial * decay


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

HOME_GROUND = {
    'Chennai Super Kings': ['Chennai'],
    'Mumbai Indians': ['Mumbai'],
    'Royal Challengers Bengaluru': ['Bengaluru', 'Bangalore'],
    'Kolkata Knight Riders': ['Kolkata'],
    'Delhi Capitals': ['Delhi'],
    'Punjab Kings': ['Mohali', 'Chandigarh', 'Dharamsala'],
    'Rajasthan Royals': ['Jaipur'],
    'Sunrisers Hyderabad': ['Hyderabad'],
    'Lucknow Super Giants': ['Lucknow'],
    'Gujarat Titans': ['Ahmedabad'],
}


def _is_home(team, city):
    cities = HOME_GROUND.get(team, [])
    if not city or not cities:
        return 0
    return 1 if str(city).strip() in cities else 0


def _safe_div(a, b, default=0.5):
    return a / b if b > 0 else default


def _parse_season(s):
    s = str(s)
    if '/' in s:
        return int(s.split('/')[0])
    try:
        return int(s)
    except:
        return 2008


# ---------------------------------------------------------------------------
#  MAIN: Build features with Elo
# ---------------------------------------------------------------------------

def build_features():
    conn = sqlite3.connect(get_db_path())

    matches_df = pd.read_sql_query("SELECT * FROM matches", conn)
    matches_df = matches_df.dropna(subset=['winner'])
    matches_df = matches_df[matches_df['winner'].str.strip() != '']
    matches_df['date'] = pd.to_datetime(matches_df['date'], format='mixed', errors='coerce')
    matches_df = matches_df.dropna(subset=['date'])
    matches_df = matches_df.sort_values('date').reset_index(drop=True)
    matches_df['season_num'] = matches_df['season'].apply(_parse_season)

    deliveries_df = pd.read_sql_query(
        "SELECT match_id, inning, batting_team, bowling_team, total_runs, is_wicket FROM deliveries",
        conn
    )
    conn.close()

    # Pre-aggregate
    match_runs = deliveries_df.groupby(['match_id', 'batting_team'])['total_runs'].sum().reset_index()
    match_runs_dict = {}
    for _, r in match_runs.iterrows():
        match_runs_dict.setdefault(r['match_id'], {})[r['batting_team']] = r['total_runs']

    # ---- Compute Elo ratings chronologically ----
    elo = {}  # team -> elo
    venue_elo = {}  # (team, venue) -> elo
    toss_field_wins = {'won': 0, 'total': 0}  # toss winner choosing field -> did they win?
    toss_bat_wins = {'won': 0, 'total': 0}

    # Track recent results for momentum
    recent_results = {}  # team -> list of (1=win, 0=loss), last N

    features_list = []
    prev_season = None

    # Collect all unique teams/venues for encoding
    all_teams = sorted(set(matches_df['team1'].dropna()) | set(matches_df['team2'].dropna()))
    all_venues = sorted(matches_df['venue'].dropna().unique())
    team_enc = {t: i for i, t in enumerate(all_teams)}
    venue_enc = {v: i for i, v in enumerate(all_venues)}

    for i, row in matches_df.iterrows():
        team1, team2 = row['team1'], row['team2']
        venue = row['venue']
        city = row.get('city', '')
        winner = row['winner']
        loser = team2 if winner == team1 else team1
        season = row['season_num']
        match_type = row.get('match_type', 'League')

        # Season reset
        if prev_season is not None and season != prev_season:
            for t in list(elo.keys()):
                elo[t] = _season_reset(elo[t])
        prev_season = season

        # Get current Elo BEFORE this match
        elo1 = elo.get(team1, INITIAL_ELO)
        elo2 = elo.get(team2, INITIAL_ELO)
        velo1 = venue_elo.get((team1, venue), INITIAL_ELO)
        velo2 = venue_elo.get((team2, venue), INITIAL_ELO)

        # Expected win probability
        expected1 = _expected_score(elo1, elo2)

        # Recent form (momentum)
        t1_recent = recent_results.get(team1, [])
        t2_recent = recent_results.get(team2, [])
        t1_form_5 = np.mean(t1_recent[-5:]) if len(t1_recent) >= 3 else 0.5
        t2_form_5 = np.mean(t2_recent[-5:]) if len(t2_recent) >= 3 else 0.5
        t1_form_10 = np.mean(t1_recent[-10:]) if len(t1_recent) >= 5 else 0.5
        t2_form_10 = np.mean(t2_recent[-10:]) if len(t2_recent) >= 5 else 0.5

        # Win/loss streak
        def _streak(results):
            if not results:
                return 0
            s = 0
            last = results[-1]
            for r in reversed(results):
                if r == last:
                    s += 1
                else:
                    break
            return s if last == 1 else -s

        t1_streak = _streak(t1_recent)
        t2_streak = _streak(t2_recent)

        # H2H from Elo perspective (track separately)
        # Head-to-head recent: how many of last encounters did team1 win?
        h2h_key = tuple(sorted([team1, team2]))
        # We'll compute this from stored match data
        h2h_matches = matches_df.iloc[:i]
        h2h_sub = h2h_matches[
            ((h2h_matches['team1'] == team1) & (h2h_matches['team2'] == team2)) |
            ((h2h_matches['team1'] == team2) & (h2h_matches['team2'] == team1))
        ]
        t1_h2h_wins = int((h2h_sub['winner'] == team1).sum())
        t2_h2h_wins = int((h2h_sub['winner'] == team2).sum())
        h2h_total = len(h2h_sub)
        h2h_ratio = _safe_div(t1_h2h_wins, t1_h2h_wins + t2_h2h_wins, 0.5)

        # Toss
        toss_is_t1 = 1 if row['toss_winner'] == team1 else 0
        chose_field = 1 if row['toss_decision'] == 'field' else 0

        # Historical toss advantage
        toss_total = toss_field_wins['total'] + toss_bat_wins['total']
        toss_adv = _safe_div(
            toss_field_wins['won'] + toss_bat_wins['won'],
            toss_total, 0.5
        ) if toss_total > 20 else 0.5

        # Home advantage
        t1_home = _is_home(team1, city)
        t2_home = _is_home(team2, city)

        # Is playoff
        is_playoff = 0 if match_type == 'League' else 1

        # Margin (for updating Elo after)
        try:
            margin = float(row.get('result_margin', 0))
            if pd.isna(margin):
                margin = 0
        except:
            margin = 0

        # ---- Only build features after we have enough history ----
        total_played_min = len(t1_recent) + len(t2_recent)
        if total_played_min >= 6:
            target = 1 if winner == team1 else 0

            features_list.append({
                # Elo features (most predictive)
                'elo_diff': elo1 - elo2,
                'elo_expected': expected1,
                'venue_elo_diff': velo1 - velo2,

                # Form & momentum
                'form_diff_5': t1_form_5 - t2_form_5,
                'form_diff_10': t1_form_10 - t2_form_10,
                'streak_diff': t1_streak - t2_streak,

                # Head-to-head
                'h2h_ratio': h2h_ratio,
                'h2h_total': min(h2h_total, 30),

                # Toss
                'toss_winner_is_team1': toss_is_t1,
                'toss_chose_field': chose_field,
                'toss_advantage': toss_adv,

                # Home
                'home_advantage': t1_home - t2_home,

                # Match context
                'is_playoff': is_playoff,

                # Identity (encoded)
                'team1_code': team_enc.get(team1, 0),
                'team2_code': team_enc.get(team2, 0),
                'venue_code': venue_enc.get(venue, 0),

                # Target
                'team1_wins': target,
            })

        # ---- Update Elo AFTER recording features (no leakage) ----
        new_w, new_l = _update_elo(elo.get(winner, INITIAL_ELO),
                                    elo.get(loser, INITIAL_ELO),
                                    margin=margin)
        elo[winner] = new_w
        elo[loser] = new_l

        # Venue Elo
        vw = venue_elo.get((winner, venue), INITIAL_ELO)
        vl = venue_elo.get((loser, venue), INITIAL_ELO)
        new_vw, new_vl = _update_elo(vw, vl, k=K_FACTOR * 0.5, margin=margin)
        venue_elo[(winner, venue)] = new_vw
        venue_elo[(loser, venue)] = new_vl

        # Track toss
        toss_w = row['toss_winner']
        if row['toss_decision'] == 'field':
            toss_field_wins['total'] += 1
            if toss_w == winner:
                toss_field_wins['won'] += 1
        else:
            toss_bat_wins['total'] += 1
            if toss_w == winner:
                toss_bat_wins['won'] += 1

        # Track recent results
        recent_results.setdefault(team1, []).append(1 if winner == team1 else 0)
        recent_results.setdefault(team2, []).append(1 if winner == team2 else 0)

    # Save state for live predictions
    _save_state(elo, venue_elo, toss_field_wins, toss_bat_wins,
                recent_results, team_enc, venue_enc)

    return pd.DataFrame(features_list)


# ---------------------------------------------------------------------------
#  Persist / Load state
# ---------------------------------------------------------------------------

def _save_state(elo, venue_elo, toss_field, toss_bat, recent, team_enc, venue_enc):
    """Save Elo ratings and other state for live prediction."""
    state = {
        'elo': elo,
        'venue_elo': {f"{k[0]}|||{k[1]}": v for k, v in venue_elo.items()},
        'toss_field_wins': toss_field,
        'toss_bat_wins': toss_bat,
        'recent_results': recent,
        'team_enc': team_enc,
        'venue_enc': venue_enc,
    }
    path = os.path.join(BASE_DIR, 'models', 'elo_state.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(state, f)


def _load_state():
    path = os.path.join(BASE_DIR, 'models', 'elo_state.json')
    with open(path, 'r') as f:
        state = json.load(f)
    state['venue_elo'] = {
        tuple(k.split('|||')): v for k, v in state['venue_elo'].items()
    }
    return state


# ---------------------------------------------------------------------------
#  LIVE FEATURE BUILDER
# ---------------------------------------------------------------------------

def compute_live_features(team1, team2, venue, toss_winner, toss_decision):
    """
    Compute features for a new prediction using latest Elo state.
    """
    state = _load_state()
    elo = state['elo']
    venue_elo = state['venue_elo']
    recent_results = state['recent_results']
    toss_field = state['toss_field_wins']
    toss_bat = state['toss_bat_wins']
    team_enc = state['team_enc']
    venue_enc = state['venue_enc']

    elo1 = elo.get(team1, INITIAL_ELO)
    elo2 = elo.get(team2, INITIAL_ELO)
    velo1 = venue_elo.get((team1, venue), INITIAL_ELO)
    velo2 = venue_elo.get((team2, venue), INITIAL_ELO)
    expected1 = _expected_score(elo1, elo2)

    t1_recent = recent_results.get(team1, [])
    t2_recent = recent_results.get(team2, [])
    t1_form_5 = np.mean(t1_recent[-5:]) if len(t1_recent) >= 3 else 0.5
    t2_form_5 = np.mean(t2_recent[-5:]) if len(t2_recent) >= 3 else 0.5
    t1_form_10 = np.mean(t1_recent[-10:]) if len(t1_recent) >= 5 else 0.5
    t2_form_10 = np.mean(t2_recent[-10:]) if len(t2_recent) >= 5 else 0.5

    def _streak(results):
        if not results:
            return 0
        s = 0
        last = results[-1]
        for r in reversed(results):
            if r == last:
                s += 1
            else:
                break
        return s if last == 1 else -s

    t1_streak = _streak(t1_recent)
    t2_streak = _streak(t2_recent)

    # H2H from DB
    conn = sqlite3.connect(get_db_path())
    h2h_df = pd.read_sql_query(
        "SELECT winner FROM matches WHERE "
        "((team1=? AND team2=?) OR (team1=? AND team2=?)) AND winner IS NOT NULL",
        conn, params=(team1, team2, team2, team1)
    )
    conn.close()
    t1_h2h = int((h2h_df['winner'] == team1).sum())
    t2_h2h = int((h2h_df['winner'] == team2).sum())
    h2h_total = len(h2h_df)
    h2h_ratio = _safe_div(t1_h2h, t1_h2h + t2_h2h, 0.5)

    toss_is_t1 = 1 if toss_winner == team1 else 0
    chose_field = 1 if toss_decision == 'field' else 0

    toss_total = toss_field['total'] + toss_bat['total']
    toss_adv = _safe_div(toss_field['won'] + toss_bat['won'], toss_total, 0.5) if toss_total > 20 else 0.5

    # Home advantage
    conn2 = sqlite3.connect(get_db_path())
    city_row = pd.read_sql_query(
        "SELECT city FROM matches WHERE venue=? AND city IS NOT NULL LIMIT 1",
        conn2, params=(venue,)
    )
    conn2.close()
    city = city_row.iloc[0]['city'] if len(city_row) > 0 else ''
    t1_home = _is_home(team1, city)
    t2_home = _is_home(team2, city)

    return pd.DataFrame([{
        'elo_diff': elo1 - elo2,
        'elo_expected': expected1,
        'venue_elo_diff': velo1 - velo2,
        'form_diff_5': t1_form_5 - t2_form_5,
        'form_diff_10': t1_form_10 - t2_form_10,
        'streak_diff': t1_streak - t2_streak,
        'h2h_ratio': h2h_ratio,
        'h2h_total': min(h2h_total, 30),
        'toss_winner_is_team1': toss_is_t1,
        'toss_chose_field': chose_field,
        'toss_advantage': toss_adv,
        'home_advantage': t1_home - t2_home,
        'is_playoff': 0,
        'team1_code': team_enc.get(team1, 0),
        'team2_code': team_enc.get(team2, 0),
        'venue_code': venue_enc.get(venue, 0),
    }])
