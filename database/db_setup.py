import pandas as pd
import sqlite3
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def get_db_path():
    return os.path.join(BASE_DIR, 'database', 'ipl.db')

def normalize_teams(df, columns):
    replacements = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Rising Pune Supergiants': 'Rising Pune Supergiants',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Kings XI Punjab': 'Punjab Kings',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
    }
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(replacements)
    return df

def init_db():
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # Read CSVs
    data_dir = os.path.join(BASE_DIR, 'data')
    try:
        matches_df = pd.read_csv(os.path.join(data_dir, 'matches.csv'))
        matches_df = normalize_teams(matches_df, ['team1', 'team2', 'toss_winner', 'winner'])
        matches_df.to_sql('matches', conn, if_exists='replace', index=False)
        
        deliveries_df = pd.read_csv(os.path.join(data_dir, 'deliveries.csv'))
        deliveries_df = normalize_teams(deliveries_df, ['batting_team', 'bowling_team'])
        deliveries_df.to_sql('deliveries', conn, if_exists='replace', index=False)
        
        players_df = pd.read_csv(os.path.join(data_dir, 'Players Data.csv'))
        players_df.to_sql('players', conn, if_exists='replace', index=False)
        
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team1 TEXT,
            team2 TEXT,
            venue TEXT,
            toss_winner TEXT,
            toss_decision TEXT,
            predicted_winner TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_del_match_id ON deliveries (match_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_del_batting ON deliveries (batting_team)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_del_bowling ON deliveries (bowling_team)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mat_team1 ON matches (team1)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mat_team2 ON matches (team2)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mat_winner ON matches (winner)")
        
        conn.commit()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing DB: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    init_db()
