import sys
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


# Commonly used functions are kept in one script to be re-used
# Use the sys library to import the ones needed for a given script
toolbox_path = '../Utility_Functions'
sys.path.append(toolbox_path)
from functions import Cal_rolling_mean_var

# Viewing preferences for Pycharm console
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 12)


# Load in all relevant data for EDA
os.chdir('../Data')

games = pd.read_csv('Raw/games.csv')
games_teams_stats = pd.read_csv('Raw/games_teams_stats.csv')
teams = pd.read_csv('Raw/teams.csv')
games_players_stats = pd.read_csv('Raw/games_players_stats.csv')
team_players = pd.read_csv('Raw/team_players.csv')
games_elo = pd.read_csv('Raw/games_elo.csv')
games_odds = pd.read_csv('Raw/games_odds.csv')

gs = pd.read_csv('Processed/gs_all_columns.csv')
gst = pd.read_csv('Processed/gst_all_columns.csv')


# Check player names and minutes played to make sure that values here match official NBA box scores
games[games.id == 6450]
games_players_stats[games_players_stats.id == 6450]
games_players_stats[games_players_stats.id == 6450]['teamId']
games_players_stats[(games_players_stats.id == 6450) & (games_players_stats.teamId == 14.0)]['min']
games_players_stats[(games_players_stats.id == 6450) & (games_players_stats.teamId == 41.0)][['min', 'playerId']]
team_players[team_players.id == 2207]
team_players[team_players.id == 356]

games[games.id == 3000]
games_players_stats[games_players_stats.id == 3000]
games_players_stats[games_players_stats.id == 3000]['teamId']
games_players_stats[(games_players_stats.id == 3000) & (games_players_stats.teamId == 27.0)][['min', 'playerId']]
games_players_stats[(games_players_stats.id == 3000) & (games_players_stats.teamId == 8.0)][['min', 'playerId']]


# Look at how a team's point differential changes over the season to see if it can be modelled mathematically
plt.plot(gs[(gs['teamId'] == 1) & (gs['seasonYear'] == 2015)]['startDate'],gs[(gs['teamId'] == 1) & (gs['seasonYear'] == 2015)]['total_point_differential'])
plt.show()
plt.plot(gs[(gs['teamId'] == 1) & (gs['seasonYear'] == 2015)]['startDate'],gs[(gs['teamId'] == 1) & (gs['seasonYear'] == 2015)]['avg_point_differential'])
plt.show()

# Look at rolling mean and variance of average point differential to see if it's stationary
team1_2015_netpointdiff = gs[(gs['teamId'] == 1) & (gs['seasonYear'] == 2015)]['avg_point_differential']
Cal_rolling_mean_var(team1_2015_netpointdiff)

team2_2015_netpointdiff = gs[(gs['teamId'] == 2) & (gs['seasonYear'] == 2015)]['avg_point_differential']
Cal_rolling_mean_var(team2_2015_netpointdiff)

# Plot effective field goal percentage and do rolling mean to see if variable is stationary
team1_2015_efg = gst[(gst['teamId'] == 1) & (gst['seasonYear'] == 2015)][['startDate','team_efg']]
plot_acf(team1_2015_efg['team_efg'])
plt.show()

plt.plot(team1_2015_efg['startDate'],team1_2015_efg['team_efg'])
plt.show()

plot_acf(team1_2015_efg['team_efg'])
plt.show()
Cal_rolling_mean_var(team1_2015_efg['team_efg'])

# Look at three pointers made to see if variable is stationary over a season
team1_2015_tpm = gst[(gst['teamId'] == 1) & (gst['seasonYear'] == 2015)][['startDate','tpm']]
Cal_rolling_mean_var(team1_2015_tpm['tpm'])
plot_acf(team1_2015_tpm['tpm'])
plt.show()

plt.plot(team1_2015_tpm['startDate'], team1_2015_tpm['tpm'])
plt.show()

# Attempt to smooth over early season variable values using NBA league average from prior season
# This may reduce the noise and provide cleaner features into our models
# Replace NaN with previous season average, do weighted average for first 15 games of the season
conditions = [gst['season_games_played'] >= 15,
              (gst['season_games_played'] <= 14) & (gst['season_games_played'] >= 1),
              gst['season_games_played'] == 0]
choices = [gst['team_efg_shifted'],
           (1.0/15.0) * (gst['team_efg_shifted'].multiply(gst['season_games_played']) + gst['efg_baseline'].multiply(15 - gst['season_games_played'])),
           gst['efg_baseline']]

gst['team_efg_shifted_smooth'] = np.select(conditions, choices, default = np.nan)
team1_2015_smooth = gst[(gst['teamId'] == 1) & (gst['seasonYear'] == 2015)][['startDate','team_efg_shifted','team_efg_shifted_smooth']]
plt.plot(team1_2015_smooth['startDate'], team1_2015_smooth['team_efg_shifted_smooth'])
plt.plot(team1_2015_smooth['startDate'], team1_2015_smooth['team_efg_shifted'])
plt.show()

for i in range(1, 42):
    team_2015_smooth = gst[(gst['teamId'] == i) & (gst['seasonYear'] == 2015)][['startDate','team_efg_shifted','team_efg_shifted_smooth']]
    plt.plot(team_2015_smooth['startDate'], team_2015_smooth['team_efg_shifted_smooth'])
    plt.plot(team_2015_smooth['startDate'], team_2015_smooth['team_efg_shifted'])
    plt.show()

team4_2015_smooth = gst[(gst['teamId'] == 4) & (gst['seasonYear'] == 2015)][['startDate','team_efg_shifted','team_efg_shifted_smooth']]
plt.plot(team4_2015_smooth['startDate'], team4_2015_smooth['team_efg_shifted_smooth'])
plt.plot(team4_2015_smooth['startDate'], team4_2015_smooth['team_efg_shifted'])
plt.show()

team9_2015_smooth = gst[(gst['teamId'] == 9) & (gst['seasonYear'] == 2015)][['startDate','team_efg_shifted','team_efg_shifted_smooth']]
plt.plot(team9_2015_smooth['startDate'], team9_2015_smooth['team_efg_shifted_smooth'])
plt.plot(team9_2015_smooth['startDate'], team9_2015_smooth['team_efg_shifted'])
plt.show()


