from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import torch
import torch.nn as nn
import os
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import numpy as np
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
import time
from datetime import datetime
import requests
import soccerdata as sd
from pathlib import Path

# Define the team name replacements
teamname_replacements = {
    "Liverpool": ["Liverpool"],
    "Arsenal": ["Arsenal"],
    "Manchester City": ["Manchester City"],
    "Newcastle": ["Newcastle Utd"],
    "Chelsea": ["Chelsea"],
    "Aston Villa": ["Aston Villa"],
    "Nottingham": ["Nott'ham Forest"],
    "Bournemouth": ["Bournemouth"],
    "Tottenham": ["Tottenham"],
    "Fulham": ["Fulham"],
    "Brighton": ["Brighton"],
    "Manchester Utd": ["Manchester Utd"],
    "Crystal Palace": ["Crystal Palace"],
    "Brentford": ["Brentford"],
    "West Ham": ["West Ham"],
    "Everton": ["Everton"],
    "Wolves": ["Wolves"],
    "Ipswich": ["Ipswich Town"],
    "Leicester": ["Leicester City"],
    "Southampton": ["Southampton"]}

# Function to get canonical team name or variations
def get_canonical_team_name(input_team_name, replacements):
    """
    Find the canonical team name and variations from the teamname_replacements dictionary.
    """
    for canonical_name, variations in replacements.items():
        if input_team_name in variations:
            return canonical_name
    raise ValueError(f"Team name '{input_team_name}' not found in teamname_replacements.")

# Load Model
class NCFBinary2(nn.Module):
    def __init__(self, num_teams, num_features, embedding_dim=16, dropout_rate=0.2, num_sports=3):
        super(NCFBinary2, self).__init__()
        
        # Team embeddings
        self.home_team_embed = nn.Embedding(num_teams, embedding_dim)
        self.away_team_embed = nn.Embedding(num_teams, embedding_dim)
        
        # Additional game features
        self.fc_game_features = nn.Linear(num_features, embedding_dim)
        
        # Shared dense layers
        self.shared_fc = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        
        # Sport-specific layers
        self.sport_specific_layers = nn.ModuleDict({
            f"sport_{i}": nn.Sequential(
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout_rate))
            for i in range(num_sports)})
        
        # Output layers for home and away win probabilities
        self.final_output = nn.Linear(64, 1)

    def forward(self, home_team, away_team, game_features, sport_id):
        home_embed = self.home_team_embed(home_team)
        away_embed = self.away_team_embed(away_team)
        
        # Process game-specific features
        game_features_embed = torch.relu(self.fc_game_features(game_features))
        
        # Concatenate all inputs
        x = torch.cat([home_embed, away_embed, game_features_embed], dim=-1)
        
        # Pass through shared layers
        shared_output = self.shared_fc(x)
        
        # Initialize tensor to store sport-specific outputs
        sport_outputs = torch.zeros(shared_output.size(0), 64, device=shared_output.device)

        # Process sport-specific logic
        for i in range(len(self.sport_specific_layers)):
            mask = (sport_id == i)
            if mask.any():
                sport_specific_layer = self.sport_specific_layers[f"sport_{i}"]
                sport_outputs[mask] = sport_specific_layer(shared_output[mask])
        
        # Apply sigmoid activation to final output
        home_win_prob = torch.sigmoid(self.final_output(sport_outputs))
        
        return home_win_prob

# EPL HELPER FUNCTIONS
fbref = sd.FBref(leagues="ENG-Premier League", seasons=2024)

# Fetch the full schedule for the league
epl_schedule = fbref.read_schedule()
epl_schedule = epl_schedule.reset_index()

# Parse the date column
epl_schedule["date"] = pd.to_datetime(epl_schedule["date"])

# Function to calculate game result for a specific team
def calculate_epl_result(row, team_name):
    """
    Calculate the result for a given team in a match.
    """
    if pd.isna(row["score"]):
        return None  # Skip games with missing scores
    if (row["home_team"] == team_name and row["score"][0] > row["score"][-1]):
        return "W"
    elif (row["away_team"] == team_name and row["score"][-1] > row["score"][0]):
        return "W"
    elif (row["home_team"] == team_name and row["score"][0] < row["score"][-1]):
        return "L"
    elif (row["away_team"] == team_name and row["score"][-1] < row["score"][0]):
        return "L"
    else:
        return "D"

# Function to calculate streaks for home/away games
def calculate_epl_streaks(games):
    """
    Calculate win/loss streaks for a team from a subset of games (home or away).
    """
    if len(games) < 3:
        return {"win_streak_3": False, "loss_streak_3": False}

    # Exclude draws
    epl_streak_games = games[games["result"] != "D"]

    # Sort games by date in descending order and take the last 3 non-draw games
    epl_streak_games = epl_streak_games.sort_values(by="date", ascending=False).head(3)

    # Determine win/loss
    epl_streak_games["win"] = epl_streak_games["result"] == "W"

    # Calculate win streak: True if all last 3 games are wins
    epl_win_streak = epl_streak_games["win"].all()

    # Calculate loss streak: True if all last 3 games are losses
    epl_loss_streak = (~epl_streak_games["win"]).all()

    return {"win_streak_3": epl_win_streak, "loss_streak_3": epl_loss_streak}

# Analyze streaks for a team
def epl_analyze_team_streaks(input_team_name):
    """
    Analyze home and away streaks for a given team.
    """
    team_name = get_canonical_team_name(input_team_name, teamname_replacements)
    
    team_epl_games = epl_schedule[
        (epl_schedule["home_team"] == team_name) | (epl_schedule["away_team"] == team_name)]

    today = pd.Timestamp(datetime.now().date())
    past_epl_games = team_epl_games[team_epl_games["date"] < today]

    # Apply result calculation
    past_epl_games = past_epl_games.copy()  # Avoid modifying the original DataFrame
    past_epl_games["result"] = past_epl_games.apply(lambda row: calculate_epl_result(row, team_name), axis=1)
    past_epl_games = past_epl_games.dropna(subset=["result"])

    # Separate home and away games
    home_epl_games = past_epl_games[past_epl_games["home_team"] == team_name]
    away_epl_games = past_epl_games[past_epl_games["away_team"] == team_name]

    # Calculate streaks for home and away games
    home_epl_streak_metrics = calculate_epl_streaks(home_epl_games)
    away_epl_streak_metrics = calculate_epl_streaks(away_epl_games)

    return home_epl_streak_metrics, away_epl_streak_metrics

# NFL HELPER FUNCTIONS
def fetch_nfl_scoreboard_data(start_date=None, end_date=None):
    """
    Fetch NFL scoreboard data for a specific date range or year.
    """
    nfl_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    nfl_params = {"dates": start_date if not end_date else f"{start_date}-{end_date}", "seasontype": 2}
    nfl_response = requests.get(nfl_url, params=nfl_params)

    if nfl_response.status_code == 200:
        return nfl_response.json()
    else:
        raise ValueError(f"Failed to fetch NFL data. Status code: {nfl_response.status_code}")

def extract_nfl_recent_games(scoreboard_data, team_name, n_games=18):
    """
    Extract the most recent NFL games for a specific team.
    """
    games = scoreboard_data.get('events', [])
    nfl_recent_games = []

    for game in games:
        competition = game.get('competitions', [])[0]
        competitors = competition.get('competitors', [])
        
        # Match the team name robustly
        for competitor in competitors:
            if competitor['team']['displayName'].strip().lower() == team_name.strip().lower():
                # Found a match for the team
                opponent = next(
                    comp['team']['displayName']
                    for comp in competitors if comp['team']['displayName'].strip().lower() != team_name.strip().lower())
                nfl_team_game = {
                    "date": pd.to_datetime(game['date']),
                    "opponent": opponent,
                    "team": competitor['team']['displayName'],
                    "team_score": int(competitor.get('score', 0)),
                    "opponent_score": int(
                        next(
                            comp['score'] for comp in competitors
                            if comp['team']['displayName'].strip().lower() != team_name.strip().lower())),
                    "win": competitor.get('winner', False),
                    "home_away": "home" if competitor['homeAway'] == "home" else "away"}
                nfl_recent_games.append(nfl_team_game)
                break

    # Sort games by date in descending order (most recent to oldest)
    nfl_recent_games = sorted(nfl_recent_games, key=lambda x: x['date'], reverse=True)

    # Debugging after sorting
    print("\nExtracted Games After Sorting:")
    for game in nfl_recent_games:
        print(f"Date: {game['date']}, Opponent: {game['opponent']}, Home/Away: {game['home_away']}, Win: {game['win']}")
        
    return nfl_recent_games[:n_games]  # Return only the most recent `n_games`

def calculate_nfl_streaks(nfl_games, nfl_home_or_away=None):
    """
    Calculate win/loss streaks for a specific team from its recent games.
    """
    # Filter games based on home or away criteria if provided
    if nfl_home_or_away:
        nfl_games = [game for game in nfl_games if game["home_away"] == nfl_home_or_away]

    # Sort games chronologically (oldest to most recent)
    nfl_games = sorted(nfl_games, key=lambda x: x['date'])

    # Exclude draws
    nfl_non_draw_games = [game for game in nfl_games if game["team_score"] != game["opponent_score"]]

    if len(nfl_non_draw_games) < 3:
        return {"win_streak_3": False, "loss_streak_3": False}

    # Take the most recent 3 non-draw games (chronological order maintained)
    nfl_streak_games = nfl_non_draw_games[-3:]

    # Determine win/loss
    nfl_win_streak = all(game["win"] for game in nfl_streak_games)
    nfl_loss_streak = all(not game["win"] for game in nfl_streak_games)

    return {"win_streak_3": nfl_win_streak, "loss_streak_3": nfl_loss_streak}

def nfl_analyze_team_streaks(nfl_team_name, start_date=None, end_date=None):
    """
    Analyze home and away streaks for a given NFL team.
    """
    nfl_scoreboard_data = fetch_nfl_scoreboard_data(start_date=20241201, end_date=20250131)
    nfl_recent_games = extract_nfl_recent_games(nfl_scoreboard_data, nfl_team_name)

    # Separate home and away games
    nfl_home_games = [game for game in nfl_recent_games if game["home_away"] == "home"]
    nfl_away_games = [game for game in nfl_recent_games if game["home_away"] == "away"]

    # Exclude draws
    nfl_home_games_non_draw = [game for game in nfl_home_games if game["team_score"] != game["opponent_score"]]
    nfl_away_games_non_draw = [game for game in nfl_away_games if game["team_score"] != game["opponent_score"]]

    # Debug: Print the last 3 home and away games (excluding draws)
    print("\nLast 3 Home Games (Excluding Draws):")
    for game in nfl_home_games_non_draw[:3]:  # Most recent 3 home games
        print(f"Date: {game['date']}, Opponent: {game['opponent']}, Score: {game['team_score']}-{game['opponent_score']}, Win: {game['win']}")

    print("\nLast 3 Away Games (Excluding Draws):")
    for game in nfl_away_games_non_draw[:3]:  # Most recent 3 away games
        print(f"Date: {game['date']}, Opponent: {game['opponent']}, Score: {game['team_score']}-{game['opponent_score']}, Win: {game['win']}")

    # Calculate streaks for home and away games
    nfl_home_streak_metrics = calculate_nfl_streaks(nfl_home_games_non_draw)
    nfl_away_streak_metrics = calculate_nfl_streaks(nfl_away_games_non_draw)

    return nfl_home_streak_metrics, nfl_away_streak_metrics

# NBA HELPER FUNCTIONS
# Initialize team lookup table
nba_team_lookup = pd.DataFrame(teams.get_teams())

# Helper Functions
def get_nba_team_id_by_fullname(fullname):
    """
    Get the NBA team ID from the full team name using nba_team_lookup.
    """
    # Check for matches in the full_name column
    matches_full_name = nba_team_lookup[nba_team_lookup['full_name'].str.contains(fullname, case=False, na=False)]
    if not matches_full_name.empty:
        return matches_full_name.iloc[0]['id']
    else:
        raise ValueError(f"NBA team full name '{fullname}' not found in lookup.")

def fetch_nba_team_recent_games(team_id, n_games=3, season='2024-25'):
    """
    Fetch the most recent games for an NBA team using nba_api.
    """
    team_games = teamgamelog.TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
    
    # Convert 'GAME_DATE' to datetime format for proper sorting
    team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'], errors='coerce')
    
    # Drop rows with invalid dates
    team_games = team_games.dropna(subset=['GAME_DATE'])

    # Sort by GAME_DATE in descending order and fetch the most recent `n_games`
    nba_recent_games = team_games.sort_values(by='GAME_DATE', ascending=False).head(n_games)
    return nba_recent_games

def calculate_nba_streak_metrics(recent_games):
    """
    Calculate win/loss streaks for the last 3 NBA games.
    """
    # Add a binary win/loss indicator
    recent_games['NBA_WIN'] = recent_games['WL'].apply(lambda x: 1 if x == 'W' else 0)
    recent_games['NBA_LOSS'] = recent_games['WL'].apply(lambda x: 1 if x == 'L' else 0)

    # Calculate win streak: True if all last 3 games are wins
    recent_games['NBA_WIN_STREAK_3'] = (
        len(recent_games) >= 3 and
        all(recent_games['WL'].iloc[:3] == 'W'))

    # Calculate loss streak: True if all last 3 games are losses
    recent_games['NBA_LOSS_STREAK_3'] = (
        len(recent_games) >= 3 and
        all(recent_games['WL'].iloc[:3] == 'L'))

    return recent_games

def get_nba_team_recent_stats(fullname, season='2024-25'):
    """
    Fetch recent NBA games and calculate metrics for a specific team.
    """
    team_id = get_nba_team_id_by_fullname(fullname)
    nba_recent_games = fetch_nba_team_recent_games(team_id, n_games=3, season=season)
    nba_recent_games = calculate_nba_streak_metrics(nba_recent_games)
    return nba_recent_games

# Load function to calculate Avg_odds_ratio
def adjusted_odds_ratio(home_odds, away_odds):
    # Ensure inputs are floats and handle cases where odds might be zero or non-numeric
    try:
        home_odds = float(home_odds)
        away_odds = float(away_odds)
        if home_odds == 0 or away_odds == 0:
            raise ValueError("Odds cannot be zero.")
    except ValueError as e:
        return None  # or return a default value or message indicating the issue

    if (home_odds > 0 and away_odds < 0) or (home_odds < 0 and away_odds > 0):
        return abs(home_odds) / abs(away_odds)
    else:
        return home_odds / away_odds

def scale_features(features_dict, home_team, away_team, home_odds, away_odds, loaded_params, game_features, sport_id):
    try:
        # Elo ratings dictionaries
        elo_ratings = {
            1: nba_elo_ratings,
            0: nfl_elo_ratings,
            2: epl_elo_ratings}

        # Scale home and away odds
        mean_avg_home_odds = loaded_params['means'][0]
        std_avg_home_odds = loaded_params['stds'][0]
        scaled_avg_home_odds = (home_odds - mean_avg_home_odds) / std_avg_home_odds

        mean_avg_away_odds = loaded_params['means'][1]
        std_avg_away_odds = loaded_params['stds'][1]
        scaled_avg_away_odds = (away_odds - mean_avg_away_odds) / std_avg_away_odds

        # Compute odds_ratio and scale it
        odds_ratio = adjusted_odds_ratio(home_odds, away_odds)
        mean_avg_odds_ratio = loaded_params['means'][2]
        std_avg_odds_ratio = loaded_params['stds'][2]
        scaled_avg_odds_ratio = (odds_ratio - mean_avg_odds_ratio) / std_avg_odds_ratio

        # Initialize streaks to defaults
        hteam_w_streak, hteam_l_streak, ateam_w_streak, ateam_l_streak = 0, 0, 0, 0

        # Fetch sport-specific streaks
        if sport_id == 2:  # EPL
            home_team = get_canonical_team_name(home_team, teamname_replacements)
            away_team = get_canonical_team_name(away_team, teamname_replacements)
            home_streaks, _ = epl_analyze_team_streaks(home_team)
            away_streaks, _ = epl_analyze_team_streaks(away_team)
            hteam_w_streak = int(home_streaks['win_streak_3'])
            hteam_l_streak = int(home_streaks['loss_streak_3'])
            ateam_w_streak = int(away_streaks['win_streak_3'])
            ateam_l_streak = int(away_streaks['loss_streak_3'])
        elif sport_id == 1:  # NBA
            nba_home_stats = get_nba_team_recent_stats(home_team)
            nba_away_stats = get_nba_team_recent_stats(away_team)
            hteam_w_streak = int(nba_home_stats['NBA_WIN_STREAK_3'].iloc[-1])
            hteam_l_streak = int(nba_home_stats['NBA_LOSS_STREAK_3'].iloc[-1])
            ateam_w_streak = int(nba_away_stats['NBA_WIN_STREAK_3'].iloc[-1])
            ateam_l_streak = int(nba_away_stats['NBA_LOSS_STREAK_3'].iloc[-1])
        elif sport_id == 0:  # NFL
            home_streaks, away_streaks = nfl_analyze_team_streaks(home_team), nfl_analyze_team_streaks(away_team)
            hteam_w_streak = int(home_streaks['win_streak_3'])
            hteam_l_streak = int(home_streaks['loss_streak_3'])
            ateam_w_streak = int(away_streaks['win_streak_3'])
            ateam_l_streak = int(away_streaks['loss_streak_3'])

        # Fetch Elo ratings, default to 1500 if not found
        home_team_elo = elo_ratings.get(sport_id, {}).get(home_team, 1500)
        away_team_elo = elo_ratings.get(sport_id, {}).get(away_team, 1500)
        elo_diff = home_team_elo - away_team_elo
        elo_discrepancy = int(pd.cut([elo_diff], bins=[-float('inf'), -50, 50, float('inf')], labels=[-1, 0, 1])[0])

        home_favored_by_odds = int(home_odds < away_odds)
        home_favored_by_elo = int(home_team_elo > away_team_elo)
        away_favored_by_odds = int(home_odds > away_odds)
        away_favored_by_elo = int(away_team_elo > home_team_elo)
        home_odds_elo_mismatch = int(home_favored_by_odds != home_favored_by_elo)
        away_odds_elo_mismatch = int(away_favored_by_odds != away_favored_by_elo)

        # Extract unscaled features
        unscaled_features = [
            hteam_w_streak, hteam_l_streak, ateam_w_streak, ateam_l_streak,
            features_dict['home_wr_favored.json'].get(home_team, 0),
            features_dict['away_wr_favored.json'].get(away_team, 0),
            features_dict['home_wr_underdog.json'].get(home_team, 0),
            features_dict['away_wr_underdog.json'].get(away_team, 0),
            features_dict['home_team_upset.json'].get(home_team, 0),
            features_dict['away_team_upset.json'].get(away_team, 0),
            home_team_elo, away_team_elo, elo_diff, elo_discrepancy, home_odds_elo_mismatch, away_odds_elo_mismatch]

        # Scale these features
        scaler = StandardScaler()
        scaler.mean_ = np.array(loaded_params['means'][3:])
        scaler.scale_ = np.array(loaded_params['stds'][3:])
        unscaled_features = np.array(unscaled_features).reshape(1, -1)
        scaled_features = scaler.transform(unscaled_features)[0]

        # Combine all features
        complete_features = [scaled_avg_home_odds, scaled_avg_away_odds, scaled_avg_odds_ratio] + list(scaled_features)
        return complete_features

    except Exception as e:
        raise RuntimeError(f"Error in scale_features: {e}")

# Load function to convert inputs into model
def prepare_inputs(home_team, away_team, home_odds, away_odds):
    # Get team IDs and sport ID
    home_team_id = teams_df.loc[teams_df['team_name'] == home_team, 'team_id'].iloc[0]
    away_team_id = teams_df.loc[teams_df['team_name'] == away_team, 'team_id'].iloc[0]
    sport_id = teams_df.loc[teams_df['team_name'] == home_team, 'sport_id'].iloc[0]

    complete_features = scale_features(
        features_dict=features_dict,
        home_team=home_team,
        away_team=away_team,
        home_odds=home_odds,
        away_odds=away_odds,
        loaded_params=loaded_params,
        game_features=None,  
        sport_id=sport_id)

    inputs = {
        'home_team': torch.tensor([home_team_id], dtype=torch.long),
        'away_team': torch.tensor([away_team_id], dtype=torch.long),
        'game_features': torch.tensor([complete_features], dtype=torch.float32),
        'sport_id': torch.tensor([sport_id], dtype=torch.long)}
    
    return inputs

# Load function to update output
def update_output(n_clicks, home_team, away_team, home_odds, away_odds):
    if n_clicks is None or n_clicks == 0:
        return ""  # No output initially
    
    if n_clicks > 0:
        # Validate odds
        odds_ratio = adjusted_odds_ratio(home_odds, away_odds)
        if odds_ratio is None:
            return "Invalid odds entered. Please enter valid odds."

        decimal_home_odds = (home_odds / 100 + 1) if home_odds > 0 else (100 / abs(home_odds) + 1)
        decimal_away_odds = (away_odds / 100 + 1) if away_odds > 0 else (100 / abs(away_odds) + 1)
        
        # Prepare input features for prediction
        inputs = prepare_inputs(home_team, away_team, home_odds, away_odds)
        home_probs = []

        # Collect predictions from each model in the ensemble
        for model in ensemble_models:
            with torch.no_grad():  # Ensure no gradients are computed
                home_prob = model(
                    inputs['home_team'],
                    inputs['away_team'],
                    inputs['game_features'],
                    inputs['sport_id'])
                home_probs.append(home_prob.item())
        
        # Compute average home win probability
        avg_home_prob = sum(home_probs) / len(home_probs)
        avg_away_prob = 1 - avg_home_prob

        # Calculate the model value bets to identify value bets
        model_home_value = (avg_home_prob * decimal_home_odds) - 1
        model_away_value = (avg_away_prob * decimal_away_odds) - 1
        home_vb = model_home_value > 0.05
        away_vb = model_away_value > 0.05
        ev_home = 10 * ((avg_home_prob * (decimal_home_odds - 1)) - (1 - avg_home_prob))
        ev_away = 10 * ((avg_away_prob * (decimal_away_odds - 1)) - (1 - avg_away_prob))

        # Construct output message
        output = f"Predicted Home Win Probability: {avg_home_prob:.4f}, Predicted Away Win Probability: {avg_away_prob:.4f}"

        if home_vb:
            output += f"\n\nHome Moneyline Bet is a value bet\n\nEV for a $10 wager: ${ev_home:.2f}"
        else:
            output += "\n\nHome Moneyline Bet is NOT a value bet"

        if away_vb:
            output += f"\n\nAway Moneyline Bet is a value bet\n\nEV for a $10 wager: ${ev_away:.2f}"
        else:
            output += "\n\nAway Moneyline Bet is NOT a value bet"

        return output

 # Load Ensemble Models
def load_models(model_dir, model_count, hyperparams):
    models = []
    for i in range(model_count):
        model_path = os.path.join(model_dir, f"model_{i}.pt")
        model = NCFBinary2(
            num_teams=hyperparams['num_teams'],
            num_features=hyperparams['num_features'],
            num_sports=hyperparams['num_sports'],
            embedding_dim=hyperparams['embedding_dim'],
            dropout_rate=hyperparams['dropout_rate'])
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval() 
        models.append(model)

    return models

# Load team names from CSV
teams_df = pd.read_csv('team_names_with_sports_and_ids.csv')

# Load ELO Ratings - Jan 6 2025
nba_elo_ratings = {
    "Oklahoma City Thunder": 1727.67,
    "Cleveland Cavaliers": 1689.42,
    "Boston Celtics": 1683.85,
    "New York Knicks": 1632.22,
    "Denver Nuggets": 1592.67,
    "Dallas Mavericks": 1585.23,
    "Houston Rockets": 1578.68,
    "Los Angeles Clippers": 1545.26,
    "Minnesota Timberwolves": 1554.03,
    "Indiana Pacers": 1551.24,
    "Memphis Grizzlies": 1561.02,
    "Los Angeles Lakers": 1525.59,
    "Orlando Magic": 1514.48,
    "Milwaukee Bucks": 1524.36,
    "Golden State Warriors": 1506.90,
    "Sacramento Kings": 1524.15,
    "Miami Heat": 1482.19,
    "Philadelphia 76ers": 1486.01,
    "Atlanta Hawks": 1480.56,
    "Phoenix Suns": 1474.09,
    "San Antonio Spurs": 1465.75,
    "Chicago Bulls": 1447.29,
    "Detroit Pistons": 1447.10,
    "Utah Jazz": 1439.04,
    "Brooklyn Nets": 1368.24,
    "Portland Trail Blazers": 1331.90,
    "New Orleans Pelicans": 1344.04,
    "Toronto Raptors": 1323.88,
    "Charlotte Hornets": 1291.98,
    "Washington Wizards": 1247.57}

nfl_elo_ratings = {
    "Kansas City Chiefs": 1663.60,
    "Detroit Lions": 1633.90,
    "Buffalo Bills": 1642.03,
    "Baltimore Raves": 1636.80,
    "Philadelphia Eagles": 1628.26,
    "Minnesota Vikings": 1571.93,
    "Green Bay Packers": 1578.91,
    "Cincinnati Bengals": 1551.59,
    "Tampa Bay Buccaneers": 1549.08,
    "Pittsburgh Steelers": 1535.29,
    "Los Angeles Rams": 1524.62,
    "Los Angeles Chargers": 1534.40,
    "Denver Broncos": 1528.54,
    "Seattle Seahawks": 1515.43,
    "San Francisco 49ers": 1533.09,
    "Washington Commanders": 1491.04,
    "Miami Dolphins": 1511.25,
    "Houston Texans": 1496.34,
    "Dallas Cowboys": 1495.64,
    "Indianapolis Colts": 1453.56,
    "Atlanta Falcons": 1441.19,
    "Arizona Cardinals": 1438.39,
    "New Orleans Saints": 1447.17,
    "New York Jets": 1428.28,
    "Cleveland Brows": 1408.85,
    "Jacksonville Jaguars": 1420.75,
    "Las Vegas Raiders": 1409.40,
    "Chicago Bears": 1411.22,
    "New England Patriots": 1398.20,
    "Tennessee Titans": 1384.54,
    "New York Giants": 1377.40,
    "Carolina Panthers": 1359.33}

epl_elo_ratings = {
    "Liverpool": 1707.96,
    "Arsenal": 1693.95,
    "Manchester City": 1646.77,
    "Newcastle": 1583.88,
    "Chelsea": 1581.35,
    "Aston Villa": 1524.47,
    "Nottingham": 1525.99,
    "Bournemouth": 1528.34,
    "Tottenham": 1514.64,
    "Fulham": 1501.51,
    "Brighton": 1488.99,
    "Manchester Utd": 1469.88,
    "Crystal Palace": 1474.63,
    "Brentford": 1466.56,
    "West Ham": 1440.42,
    "Everton": 1430.22,
    "Wolves": 1414.84,
    "Ipswich": 1377.64,
    "Leicester": 1339.31,
    "Southampton": 1288.65}

# Setup the Dropdown in Dash
dropdown = dcc.Dropdown(options=[{'label': team, 'value': team} for team in teams_df['team_name']], searchable=True, placeholder="Select a team", id='team-dropdown')
# Load team names
teams = [{'label': team, 'value': team} for team in teams_df['team_name'].unique()]

# Define game features in the exact order used during training
game_features = [
    'avg_home_odds', 'avg_away_odds', 'avg_odds_ratio', 'hteam_w_streak', 'ateam_w_streak', 'hteam_l_streak', 'ateam_l_streak', 
    'home_team_elo', 'away_team_elo', 'home_wr_favored', 'away_wr_favored', 'home_wr_underdog', 'away_wr_underdog',
    'home_team_upset', 'away_team_upset', 'elo_diff', 'elo_discrepancy', 'home_odds_elo_mismatch', 'away_odds_elo_mismatch']

# Initialize your Dash app using the standard Dash class
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        html.H1("ML Model to Identify Value Bets", style={'textAlign': 'center'}),
        html.P("You can use my trained model to identify profitable moneyline bets. Simply enter the home and away teams and input the odds. The model will then provide predictions for the home and away team win probabilities based on the information entered, as well as calculate the expected value of placing a $10 wager on either team.",
               style={'textAlign': 'left'}),
        dbc.Row([
            dbc.Col([
                html.Div("Home Team:", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='input-home-team',
                    options=teams,
                    searchable=True,
                    placeholder="Select Home Team"
                ),
                html.Br(),
                html.Div("Home Odds:", style={'textAlign': 'center'}),
                dcc.Input(
                    id='input-home-odds', type='number', placeholder='(American Odds format)', step=0.01, debounce=True, style={'width': '100%'}
                )
            ], width=6),
            dbc.Col([
                html.Div("Away Team:", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='input-away-team',
                    options=teams,
                    searchable=True,
                    placeholder="Select Away Team"
                ),
                html.Br(),
                html.Div("Away Odds:", style={'textAlign': 'center'}),
                dcc.Input(
                    id='input-away-odds', type='number', placeholder='(American Odds format)', step=0.01, debounce=True, style={'width': '100%'}
                )
            ], width=6),
        ]),
        html.Hr(),  
        dbc.Row([
            dbc.Col(html.Button('Submit', id='submit-button', n_clicks=0), width={"size": 2, "offset": 5}, className="d-flex justify-content-center")
        ]),
        html.Br(),  
        dbc.Row([
            dbc.Col(dcc.Markdown(id='output-col-1', style={'display': 'none', 'border': '1px solid #ddd', 'padding': '10px'}), width=6),
            dbc.Col(dcc.Markdown(id='output-col-2', style={'display': 'none', 'border': '1px solid #ddd', 'padding': '10px'}), width=6),
        ])
    ])
])

@app.callback(
    [Output('output-col-1', 'children'),
     Output('output-col-2', 'children'),
     Output('output-col-1', 'style'),
     Output('output-col-2', 'style'),
     Output('submit-button', 'n_clicks')],
    [Input('submit-button', 'n_clicks')],
    [State('input-home-team', 'value'),
     State('input-away-team', 'value'),
     State('input-home-odds', 'value'),
     State('input-away-odds', 'value')])

def update_output_callback(n_clicks, home_team, away_team, home_odds, away_odds):
    if n_clicks is None or n_clicks == 0:
        return "", "", {'display': 'none'}, {'display': 'none'}, n_clicks  

    # Error check: Ensure all fields are filled
    if not all([home_team, away_team, home_odds, away_odds]):
        return "Please fill in all fields.", "", {'display': 'block'}, {'display': 'none'}, n_clicks

    # Error check: Ensure the teams are not the same
    if home_team == away_team:
        return "The same team cannot be home and away. Try again.", "", {'display': 'block'}, {'display': 'none'}, n_clicks

    # Error check: Ensure the teams belong to the same sport
    home_sport = teams_df.loc[teams_df['team_name'] == home_team, 'sport'].iloc[0]
    away_sport = teams_df.loc[teams_df['team_name'] == away_team, 'sport'].iloc[0]
    if home_sport != away_sport:
        return "The two teams are not in the same sport.", "", {'display': 'block'}, {'display': 'none'}, n_clicks

    # Error check: Validate odds range
    if -100 < home_odds < 100 or -100 < away_odds < 100:
        return "Odds cannot be within the range -100 to 100. Please enter valid odds.", "", {'display': 'block'}, {'display': 'none'}, n_clicks

    try:
        # Validate and prepare odds
        odds_ratio = adjusted_odds_ratio(home_odds, away_odds)
        if odds_ratio is None:
            return "Invalid odds entered.", "", {'display': 'block'}, {'display': 'none'}, n_clicks

        decimal_home_odds = (home_odds / 100 + 1) if home_odds > 0 else (100 / abs(home_odds) + 1)
        decimal_away_odds = (away_odds / 100 + 1) if away_odds > 0 else (100 / abs(away_odds) + 1)

        # Prepare model inputs
        inputs = prepare_inputs(home_team, away_team, home_odds, away_odds)
        home_probs = []

        # Collect predictions from each model
        for model in ensemble_models:
            with torch.no_grad():
                home_prob = model(
                    inputs['home_team'],
                    inputs['away_team'],
                    inputs['game_features'],
                    inputs['sport_id'])
                home_probs.append(home_prob.item())

        # Compute the average probabilities
        avg_home_prob = sum(home_probs) / len(home_probs)
        avg_away_prob = 1 - avg_home_prob

        # Calculate value bets and EVs
        model_home_value = (avg_home_prob * decimal_home_odds) - 1
        model_away_value = (avg_away_prob * decimal_away_odds) - 1
        home_vb = model_home_value > 0.05
        away_vb = model_away_value > 0.05
        ev_home = 10 * ((avg_home_prob * (decimal_home_odds - 1)) - (1 - avg_home_prob))
        ev_away = 10 * ((avg_away_prob * (decimal_away_odds - 1)) - (1 - avg_away_prob))

    except Exception as e:
        return f"Error during prediction: {str(e)}", "", {'display': 'block'}, {'display': 'none'}, n_clicks

    # Construct output content
    col1_content = f"Predicted Home Win Probability: {avg_home_prob:.2%}"
    col2_content = f"Predicted Away Win Probability: {avg_away_prob:.2%}"

    if home_vb:
        col1_content += f"\n\nHome Moneyline Bet is a value bet\n\nEV for a $10 wager: ${ev_home:.2f}"
    else:
        col1_content += "\n\nHome Moneyline Bet is NOT a value bet"

    if away_vb:
        col2_content += f"\n\nAway Moneyline Bet is a value bet\n\nEV for a $10 wager: ${ev_away:.2f}"
    else:
        col2_content += "\n\nAway Moneyline Bet is NOT a value bet"

    visible_style = {'display': 'block'}
    return col1_content, col2_content, visible_style, visible_style, 0

if __name__ == '__main__':
    # Load hyperparameters
    with open('model_hyperparameters.json', 'r') as f:
        hyperparams = json.load(f)

    # Load features
    features_json = [
    'home_wr_favored.json',
    'away_wr_favored.json',
    'home_wr_underdog.json',
    'away_wr_underdog.json',
    'home_team_upset.json',
    'away_team_upset.json']

    # Dictionary to store loaded data
    features_dict = {}
    
    # Iterate over the files and load their contents
    for features in features_json:
        with open(features, 'r') as f:
            features_dict[features] = json.load(f)

    # Load Scaler
    with open('scaler_params.json', 'r') as f:
        loaded_params = json.load(f)
        scaler = StandardScaler()
        scaler.mean_ = np.array(loaded_params['means'])
        scaler.scale_ = np.array(loaded_params['stds'])
    
    # Run models
    ensemble_models = load_models('ensemble_modelsNCFBinary2', 10, hyperparams)

    # Start server
    app.run_server(debug=True)