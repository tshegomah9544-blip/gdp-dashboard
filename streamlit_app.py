#!/bin/bash

# ------------------------------
# Fully Automated Football Goal Predictor Deployment (With Preloaded Matches)
# ------------------------------

# ----------- CONFIGURATION -----------
GITHUB_USER="your_github_username"        # Replace
GITHUB_TOKEN="your_github_token"          # Replace
STREAMLIT_TOKEN="your_streamlit_token"    # Replace
REPO_NAME="football-goal-predictor"
LOCAL_DIR="$HOME/$REPO_NAME"
APP_FILE="football_goal_predictor_streamlit.py"

# ----------- STEP 1: Create repo directory -----------
mkdir -p "$LOCAL_DIR"
cd "$LOCAL_DIR" || exit

# ----------- STEP 2: Generate files -----------

# README.md
cat <<EOL > README.md
# Football Goal Predictor

A Streamlit web app for predicting football match outcomes:

- First-half and second-half expected goals  
- Over/Under probabilities  
- Takes into account player form, injuries, and last 5 matches  
- Preloaded example matches (TeamA vs TeamB, TeamB vs TeamA)

## How to Run Locally

\`\`\`bash
git clone https://github.com/$GITHUB_USER/$REPO_NAME.git
cd $REPO_NAME
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run $APP_FILE
\`\`\`
EOL

# requirements.txt
cat <<EOL > requirements.txt
streamlit
pandas
numpy
scipy
EOL

# .gitignore
cat <<EOL > .gitignore
venv/
__pycache__/
*.pyc
*.pyo
.DS_Store
EOL

# App file with preloaded example matches
cat <<'EOL' > "$APP_FILE"
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# --- Mock lineup fetch with injuries ---
def fetch_lineup_mock(team_id):
    data = [
        {'player': 'Player1', 'rating': 7.5, 'form': 0.9, 'injured': False},
        {'player': 'Player2', 'rating': 8.0, 'form': 0.85, 'injured': True},
        {'player': 'Player3', 'rating': 7.0, 'form': 0.8, 'injured': False}
    ]
    lineup_data = []
    for player in data:
        injury_factor = 0 if player['injured'] else 1
        lineup_data.append({
            'player': player['player'],
            'rating': player['rating']*injury_factor,
            'form': player['form']*injury_factor
        })
    return pd.DataFrame(lineup_data)

# --- Sample last 5 matches ---
last5_matches_data = pd.DataFrame({
    'team': ['TeamA']*5 + ['TeamB']*5,
    '1H_scored': [1,0,2,1,1, 0,1,1,0,2],
    '1H_conceded': [0,1,1,1,0, 1,0,1,2,1],
    '2H_scored': [0,1,1,0,2, 1,0,1,1,1],
    '2H_conceded': [1,0,0,1,1, 0,1,1,0,1]
})

coach_stats = {('TeamA','TeamB'): 1.0, ('TeamB','TeamA'): -0.5}

class GoalEstimatorApp:
    def __init__(self):
        self.last5_matches_data = last5_matches_data
        self.coach_stats = coach_stats

    def get_team_strength(self, lineup_df):
        return (lineup_df['rating'] * lineup_df['form']).sum()

    def get_expected_goals_half(self, teamA_stats, teamB_stats, half='1H', lineup_strength=0, h2h_advantage=0):
        avg_scored = teamA_stats[f'{half}_scored'].mean()
        avg_conceded = teamB_stats[f'{half}_conceded'].mean()
        return (avg_scored + avg_conceded)*(1 + 0.05*lineup_strength + 0.05*h2h_advantage)

    def poisson_matrix(self, lambdaA, lambdaB, max_goals=6):
        matrix = np.zeros((max_goals+1, max_goals+1))
        for i in range(max_goals+1):
            for j in range(max_goals+1):
                matrix[i,j] = poisson.pmf(i, lambdaA)*poisson.pmf(j, lambdaB)
        return matrix

    def predict_half_probabilities(self, teamA_id, teamB_id):
        lineupA = fetch_lineup_mock(teamA_id)
        lineupB = fetch_lineup_mock(teamB_id)
        lsA = self.get_team_strength(lineupA)
        lsB = self.get_team_strength(lineupB)

        teamA_stats = self.last5_matches_data[self.last5_matches_data['team']==teamA_id]
        teamB_stats = self.last5_matches_data[self.last5_matches_data['team']==teamB_id]

        h2h_adv = self.coach_stats.get((teamA_id, teamB_id),0)

        teamA_1H = self.get_expected_goals_half(teamA_stats, teamB_stats, '1H', lsA, h2h_adv)
        teamB_1H = self.get_expected_goals_half(teamB_stats, teamA_stats, '1H', lsB, -h2h_adv)
        teamA_2H = self.get_expected_goals_half(teamA_stats, teamB_stats, '2H', lsA, h2h_adv)
        teamB_2H = self.get_expected_goals_half(teamB_stats, teamA_stats, '2H', lsB, -h2h_adv)

        m_total = self.poisson_matrix(teamA_1H+teamA_2H, teamB_1H+teamB_2H)
        threshold = 2.5
        over_total = np.sum([m_total[i,j] for i in range(7) for j in range(7) if i+j>threshold])

        return {
            '1H_expected_goals': {'teamA': round(teamA_1H,2),'teamB': round(teamB_1H,2)},
            '2H_expected_goals': {'teamA': round(teamA_2H,2),'teamB': round(teamB_2H,2)},
            'over_under_total': {'over': round(over_total,2), 'under': round(1-over_total,2)}
        }

def match_prediction_generator(estimator, matches):
    for match in matches:
        pred = estimator.predict_half_probabilities(match['teamA_id'], match['teamB_id'])
        yield {'match_id': match['match_id'], 'teamA': match['teamA_id'], 'teamB': match['teamB_id'], 'predictions': pred}

st.title("Football Goal Predictor")

estimator = GoalEstimatorApp()

# --- Preloaded matches ---
matches = [
    {'match_id': '001', 'teamA_id': 'TeamA', 'teamB_id': 'TeamB'},
    {'match_id': '002', 'teamA_id': 'TeamB', 'teamB_id': 'TeamA'}
]

# --- User form to add new matches ---
with st.form("match_input_form"):
    st.subheader("Add Your Own Match")
    match_id = st.text_input("Match ID")
    teamA = st.text_input("Team A")
    teamB = st.text_input("Team B")
    submitted = st.form_submit_button("Add Match")
    if submitted:
        matches.append({'match_id': match_id, 'teamA_id': teamA, 'teamB_id': teamB})
        st.success(f"Added Match: {teamA} vs {teamB}")

st.subheader("Predictions")
results = list(match_prediction_generator(estimator, matches))
for result in results:
    st.write(f"**Match {result['match_id']}: {result['teamA']} vs {result['teamB']}**")
    st.json(result['predictions'])
EOL

# ----------- STEP 3: Virtual environment and install packages -----------

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade streamlit

# ----------- STEP 4: Initialize Git and push to GitHub -----------

git init
git add .
git commit -m "Initial commit: Football Goal Predictor Streamlit App"

curl -u $GITHUB_USER:$GITHUB_TOKEN https://api.github.com/user/repos \
    -d "{\"name\":\"$REPO_NAME\",\"description\":\"Football Goal Predictor Streamlit App with Preloaded Matches\"}"

git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git
git branch -M main