import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

toolbox_path = '../Utility_Functions'
sys.path.append(toolbox_path)
from functions import efg, oreb_rate, ft_rate, to_rate, four_factors_averages, f_multi

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 12)

os.chdir('../Data')

games = pd.read_csv('Raw/games.csv')
games_teams_stats = pd.read_csv('Raw/games_teams_stats.csv')
teams = pd.read_csv('Raw/teams.csv')
games_players_stats = pd.read_csv('Raw/games_players_stats.csv')
team_players = pd.read_csv('Raw/team_players.csv')
games_elo = pd.read_csv('Raw/games_elo.csv')
games_odds = pd.read_csv('Raw/games_odds.csv')
player_ratings = pd.read_csv('Raw/player_ratings.csv')


games_h = games[['id', 'seasonYear', 'startDate', 'hTeamId']]
games_h.columns = ['id', 'seasonYear', 'startDate', 'teamId']
games_h['home_flag'] = 1
games_v = games[['id', 'seasonYear', 'startDate', 'vTeamId']]
games_v.columns = ['id', 'seasonYear', 'startDate', 'teamId']
games_v['home_flag'] = 0
games_final = pd.concat([games_h, games_v])


stats = games_teams_stats[['id', 'teamId', 'plusMinus']]
gs = pd.merge(left=games_final, right=stats, on=['id', 'teamId'])
gs.sort_values(by=['seasonYear', 'teamId', 'startDate'], ascending=True, inplace=True)
gs['result_for_streak_var'] = np.where(gs['plusMinus'] > 0, 1, -1)
gs['start'] = gs.groupby(by=['seasonYear', 'teamId'])['result_for_streak_var'].apply(lambda x: x.ne(x.shift()))
gs['streakid'] = gs.groupby(by=['seasonYear', 'teamId'])['start'].apply(lambda x: x.cumsum())
gs['streak_with_game'] = gs.groupby(by=['seasonYear', 'teamId', 'streakid']).cumcount() + 1
gs['streak_with_game'] = gs['streak_with_game'] * gs['result_for_streak_var']
gs['streak_entering'] = gs.groupby(by=['seasonYear', 'teamId'])['streak_with_game'].apply(lambda x: x.shift(1))

# Streak for home and away
gs['start_ha'] = gs.groupby(by=['seasonYear', 'teamId', 'home_flag'])['result_for_streak_var'].apply(lambda x: x.ne(x.shift()))
gs['streakid_ha'] = gs.groupby(by=['seasonYear', 'teamId', 'home_flag'])['start_ha'].apply(lambda x: x.cumsum())
gs['streak_with_game_ha'] = gs.groupby(by=['seasonYear', 'teamId', 'home_flag', 'streakid_ha']).cumcount() + 1
gs['streak_with_game_ha'] = gs['streak_with_game_ha'] * gs['result_for_streak_var']
gs['streak_entering_ha'] = gs.groupby(by=['seasonYear', 'teamId', 'home_flag'])['streak_with_game_ha'].apply(lambda x: x.shift(1))

gs['result_binary'] = np.where(gs['plusMinus'] > 0, 1, -1)
gs['win_percentage_last10'] = gs.groupby(by=['seasonYear', 'teamId'])['result_binary'].transform(lambda x: x.rolling(10, 1).mean())
gs['win_percentage_ha_last10'] = gs.groupby(by=['seasonYear', 'teamId', 'home_flag'])['result_binary'].transform(lambda x: x.rolling(10, 1).mean())

gs['win_percentage_last10_shifted'] = gs.groupby(by=['seasonYear', 'teamId'])['win_percentage_last10'].apply(lambda x: x.shift(1))
gs['win_percentage_ha_last10_shifted'] = gs.groupby(by=['seasonYear', 'teamId', 'home_flag'])['win_percentage_ha_last10'].apply(lambda x: x.shift(1))

# Create days rest
gs['previous_game_date'] = gs.groupby(by=['seasonYear', 'teamId'])['startDate'].apply(lambda x: x.shift(1))
gs['days_rest'] = (pd.to_datetime(gs['startDate']) - pd.to_datetime(gs['previous_game_date'])).dt.days - 1
gs['b2b_flag'] = gs['days_rest'] == 0

# Create point differential
gs['counting_value'] = 1
gs['total_point_differential'] = gs.groupby(by=['seasonYear','teamId'])['plusMinus'].cumsum()
gs['avg_point_differential'] = gs.groupby(by=['seasonYear','teamId'])['plusMinus'].cumsum() / gs.groupby(by=['seasonYear','teamId'])['counting_value'].cumsum()
gs['avg_point_differential_shifted'] = gs.groupby(by=['seasonYear', 'teamId'])['avg_point_differential'].apply(lambda x: x.shift(1))

gs['avg_point_differential_last10'] = gs.groupby(by=['seasonYear','teamId'])['plusMinus'].transform(lambda x: x.rolling(10, 1).mean())
gs['avg_point_differential_last10_shifted'] = gs.groupby(by=['seasonYear', 'teamId'])['avg_point_differential_last10'].apply(lambda x: x.shift(1))

# Point differential for home and away
gs['total_point_differential_ha'] = gs.groupby(by=['seasonYear','teamId', 'home_flag'])['plusMinus'].cumsum()
gs['avg_point_differential_ha'] = gs.groupby(by=['seasonYear','teamId', 'home_flag'])['plusMinus'].cumsum() / gs.groupby(by=['seasonYear','teamId', 'home_flag'])['counting_value'].cumsum()
gs['avg_point_differential_ha_shifted'] = gs.groupby(by=['seasonYear', 'teamId', 'home_flag'])['avg_point_differential_ha'].apply(lambda x: x.shift(1))

gs['avg_point_differential_last10_ha'] = gs.groupby(by=['seasonYear','teamId', 'home_flag'])['plusMinus'].transform(lambda x: x.rolling(10, 1).mean())
gs['avg_point_differential_last10_ha_shifted'] = gs.groupby(by=['seasonYear','teamId', 'home_flag'])['avg_point_differential_last10_ha'].apply(lambda x: x.shift(1))

gs.to_csv('Processed/gs_all_columns.csv', index_label=False)


gs_features = gs[['id','seasonYear','startDate','teamId','home_flag','streak_entering', 'streak_entering_ha',
                  'days_rest', 'avg_point_differential_shifted', 'avg_point_differential_ha_shifted',
                  'win_percentage_last10_shifted',
                  'win_percentage_ha_last10_shifted',
                  'b2b_flag',
                  'avg_point_differential_last10_shifted',
                  'avg_point_differential_last10_ha_shifted'
                  ]]
gs_features_home = gs_features[gs_features['home_flag'] == 1]
gs_features_away = gs_features[gs_features['home_flag'] == 0]

gs_features_home = gs_features_home.add_prefix('home_')
gs_features_away = gs_features_away.add_prefix('away_')




# Need to join to game team stats
gst = pd.merge(left=games_final, right=games_teams_stats, on=['id','teamId']).sort_values(by=['seasonYear_x', 'teamId', 'startDate'], ascending=True)

# Filter out records with missing box score stats
gst = gst[(gst.offReb.isna() == False) & (gst.fga.isna() == False) & (gst.turnovers.isna() == False)]

# Only use 2015-2020 seasons, not current season
gst = gst[(gst.seasonYear_x <= 2020) & (gst.seasonYear_x >= 2015)]

# cumsum fgm fga 3pm 3pa for themselves
gst['cum_fga'] = gst.groupby(by=['seasonYear_x','teamId'])['fga'].cumsum()
gst['cum_fgm'] = gst.groupby(by=['seasonYear_x','teamId'])['fgm'].cumsum()
gst['cum_tpm'] = gst.groupby(by=['seasonYear_x','teamId'])['tpm'].cumsum()

gst['cum_offReb'] = gst.groupby(by=['seasonYear_x','teamId'])['offReb'].cumsum()
gst['cum_fta'] = gst.groupby(by=['seasonYear_x','teamId'])['fta'].cumsum()
gst['cum_turnovers'] = gst.groupby(by=['seasonYear_x','teamId'])['turnovers'].cumsum()


four_factors_baseline_df = four_factors_averages()

gst.rename(columns={'seasonYear_x': 'seasonYear'}, inplace=True)


gst['team_efg'] = gst.apply(lambda x: efg(x.cum_fgm, x.cum_fga, x.cum_tpm), axis=1)
gst['team_oreb_rate'] = gst.apply(lambda x: oreb_rate(x.cum_offReb, x.cum_fgm, x.cum_fga), axis=1)
gst['team_ft_rate'] = gst.apply(lambda x: ft_rate(x.cum_fta, x.cum_fga), axis=1)
gst['team_to_rate'] = gst.apply(lambda x: to_rate(x.cum_fga, x.cum_turnovers, x.cum_fta, x.cum_offReb), axis=1)

# Same metrics but for home or away (depending on that specific game)
gst['cum_ha_fga'] = gst.groupby(by=['seasonYear','teamId', 'home_flag'])['fga'].cumsum()
gst['cum_ha_fgm'] = gst.groupby(by=['seasonYear','teamId', 'home_flag'])['fgm'].cumsum()
gst['cum_ha_tpm'] = gst.groupby(by=['seasonYear','teamId', 'home_flag'])['tpm'].cumsum()

gst['cum_ha_offReb'] = gst.groupby(by=['seasonYear','teamId', 'home_flag'])['offReb'].cumsum()
gst['cum_ha_fta'] = gst.groupby(by=['seasonYear','teamId', 'home_flag'])['fta'].cumsum()
gst['cum_ha_turnovers'] = gst.groupby(by=['seasonYear','teamId', 'home_flag'])['turnovers'].cumsum()

gst['team_ha_efg'] = gst.apply(lambda x: efg(x.cum_ha_fgm, x.cum_ha_fga, x.cum_ha_tpm), axis=1)
gst['team_ha_oreb_rate'] = gst.apply(lambda x: oreb_rate(x.cum_ha_offReb, x.cum_ha_fgm, x.cum_ha_fga), axis=1)
gst['team_ha_ft_rate'] = gst.apply(lambda x: ft_rate(x.cum_ha_fta, x.cum_ha_fga), axis=1)
gst['team_ha_to_rate'] = gst.apply(lambda x: to_rate(x.cum_ha_fga, x.cum_ha_turnovers, x.cum_ha_fta, x.cum_ha_offReb), axis=1)


gst['team_efg_shifted'] = gst.groupby(by=['seasonYear', 'teamId'])['team_efg'].apply(lambda x: x.shift(1))
gst['team_oreb_rate_shifted'] = gst.groupby(by=['seasonYear', 'teamId'])['team_oreb_rate'].apply(lambda x: x.shift(1))
gst['team_ft_rate_shifted'] = gst.groupby(by=['seasonYear', 'teamId'])['team_ft_rate'].apply(lambda x: x.shift(1))
gst['team_to_rate_shifted'] = gst.groupby(by=['seasonYear', 'teamId'])['team_to_rate'].apply(lambda x: x.shift(1))

gst['team_ha_efg_shifted'] = gst.groupby(by=['seasonYear', 'teamId', 'home_flag'])['team_ha_efg'].apply(lambda x: x.shift(1))
gst['team_ha_oreb_rate_shifted'] = gst.groupby(by=['seasonYear', 'teamId', 'home_flag'])['team_ha_oreb_rate'].apply(lambda x: x.shift(1))
gst['team_ha_ft_rate_shifted'] = gst.groupby(by=['seasonYear', 'teamId', 'home_flag'])['team_ha_ft_rate'].apply(lambda x: x.shift(1))
gst['team_ha_to_rate_shifted'] = gst.groupby(by=['seasonYear', 'teamId', 'home_flag'])['team_ha_to_rate'].apply(lambda x: x.shift(1))
gst['season_games_played'] = gst.groupby(by=['seasonYear', 'teamId']).cumcount()
gst['season_ha_games_played'] = gst.groupby(by=['seasonYear', 'teamId', 'home_flag']).cumcount()

# Calculate four factors as moving average instead of full historical data for the season
# Do this as a 10 game moving average

gst['moving_average_fga'] = gst.groupby(by=['seasonYear','teamId'])['fga'].transform(lambda x: x.rolling(10, 1).mean())
gst['moving_average_fgm'] = gst.groupby(by=['seasonYear','teamId'])['fgm'].transform(lambda x: x.rolling(10, 1).mean())
gst['moving_average_tpm'] = gst.groupby(by=['seasonYear','teamId'])['tpm'].transform(lambda x: x.rolling(10, 1).mean())
gst['moving_average_offReb'] = gst.groupby(by=['seasonYear','teamId'])['offReb'].transform(lambda x: x.rolling(10, 1).mean())
gst['moving_average_fta'] = gst.groupby(by=['seasonYear','teamId'])['fta'].transform(lambda x: x.rolling(10, 1).mean())
gst['moving_average_turnovers'] = gst.groupby(by=['seasonYear','teamId'])['turnovers'].transform(lambda x: x.rolling(10, 1).mean())

gst['team_efg_ma'] = gst.apply(lambda x: efg(x.moving_average_fgm, x.moving_average_fga, x.moving_average_tpm), axis=1)
gst['team_oreb_rate_ma'] = gst.apply(lambda x: oreb_rate(x.moving_average_offReb, x.moving_average_fgm, x.moving_average_fga), axis=1)
gst['team_ft_rate_ma'] = gst.apply(lambda x: ft_rate(x.moving_average_fta, x.moving_average_fga), axis=1)
gst['team_to_rate_ma'] = gst.apply(lambda x: to_rate(x.moving_average_fga, x.moving_average_turnovers, x.moving_average_fta, x.moving_average_offReb), axis=1)

gst['team_efg_ma_shifted'] = gst.groupby(by=['seasonYear', 'teamId'])['team_efg_ma'].apply(lambda x: x.shift(1))
gst['team_oreb_rate_ma_shifted'] = gst.groupby(by=['seasonYear', 'teamId'])['team_oreb_rate_ma'].apply(lambda x: x.shift(1))
gst['team_ft_rate_ma_shifted'] = gst.groupby(by=['seasonYear', 'teamId'])['team_ft_rate_ma'].apply(lambda x: x.shift(1))
gst['team_to_rate_ma_shifted'] = gst.groupby(by=['seasonYear', 'teamId'])['team_to_rate_ma'].apply(lambda x: x.shift(1))

gst['result'] = np.where(gst['plusMinus'] > 0, 1, 0)
gst['previous_seasonYear'] = gst['seasonYear'] - 1

gst = pd.merge(gst, four_factors_baseline_df, left_on=['previous_seasonYear'], right_on=['year'])

gst.to_csv('Processed/gst_all_columns.csv', index_label=False)


# gst_features = gst[['id', 'teamId', 'home_flag', 'result', 'plusMinus', 'seasonYear', 'startDate', 'season_games_played', 'season_ha_games_played',
#                     'team_efg','team_oreb_rate','team_ft_rate','team_to_rate', 'team_ha_efg','team_ha_oreb_rate','team_ha_ft_rate','team_ha_to_rate']]
#
#
# gst_features_home = gst_features[gst_features['home_flag'] == 1]
# gst_features_away = gst_features[gst_features['home_flag'] == 0]

gst_features = gst[['id', 'teamId', 'home_flag', 'plusMinus','result','seasonYear', 'startDate', 'season_games_played', 'season_ha_games_played',
                    'team_efg_shifted','team_oreb_rate_shifted','team_ft_rate_shifted','team_to_rate_shifted',
                    'team_efg_ma_shifted','team_oreb_rate_ma_shifted','team_ft_rate_ma_shifted','team_to_rate_ma_shifted',
                    'team_ha_efg_shifted',
                    'team_ha_oreb_rate_shifted','team_ha_ft_rate_shifted','team_ha_to_rate_shifted', 'points']]

gst_features_home = gst_features[gst_features['home_flag'] == 1]
gst_features_away = gst_features[gst_features['home_flag'] == 0]

gst_features_home = gst_features_home.add_prefix('home_')
gst_features_away = gst_features_away.add_prefix('away_')

full_features = pd.merge(gst_features_home, gst_features_away,
                         left_on=['home_id', 'home_seasonYear', 'home_startDate'],
                         right_on=['away_id', 'away_seasonYear', 'away_startDate'],
                         how='inner')

# Merge gs features home and away - win streak, point differential on the year, etc
full_features = pd.merge(full_features, gs_features_home, left_on=['home_id', 'home_seasonYear', 'home_startDate', 'home_teamId'], right_on=['home_id', 'home_seasonYear', 'home_startDate', 'home_teamId'])
full_features = pd.merge(full_features, gs_features_away, left_on=['home_id', 'home_seasonYear', 'home_startDate', 'away_teamId'], right_on=['away_id', 'away_seasonYear', 'away_startDate', 'away_teamId'])

# Add elo here - don't need home and away since they are perfect correlated
# elo_pre, elo_win_prob for home and away
# join elo for home and away teams
elo_info = pd.merge(teams, games_elo, left_on=['shortName'], right_on=['team'])
elo_info.drop(['shortName', 'createdAt_x', 'fullName', 'logo', 'allStar', 'city', 'confName', 'nickname', 'updatedAt_x',
               'divName', 'nbaFranchise', 'raptor_pre', 'importance',
       'neutral', 'carm_elo_pre', 'score', 'createdAt_y', 'playoff', 'elo_post', 'quality',
               'updatedAt_y', 'carm_elo_post',  'carm_elo_prob', 'team', 'raptor_prob', 'total_rating', 'season',
       'game_id'], axis=1, inplace=True)
elo_info.rename(columns={'id': 'teamId'}, inplace=True) # 'id', 'date', 'elo_pre', 'elo_prob'
elo_home = elo_info.add_prefix('home_')
elo_away = elo_info.add_prefix('away_')

full_features = pd.merge(full_features, elo_home, left_on=['home_startDate', 'home_teamId'], right_on=['home_date', 'home_teamId'], how='inner')
full_features = pd.merge(full_features, elo_away, left_on=['home_startDate', 'away_teamId'], right_on=['away_date', 'away_teamId'], how='inner')

# NBA 2k ratings
# Make sure it's not data leakage
# Season on 2k game is one year ahead of season in other data, so fix that first
gp = games_players_stats[['id', 'playerId', 'teamId', 'seasonYear']]
player_ratings['seasonYear_shifted'] = player_ratings['seasonYear'] - 1
pr = player_ratings[['id', 'seasonYear_shifted', 'rating']]
pr.rename(columns={'id': 'playerId'}, inplace=True)
game_player_ratings = pd.merge(gp, pr, how='left', left_on=['playerId', 'seasonYear'], right_on=['playerId', 'seasonYear_shifted'])
# Replace unknown ratings with 69 (as that seems to be the baseline replacement player)
replacement_level = 69
game_player_ratings['rating'].replace(np.nan, replacement_level, inplace=True)
game_player_ratings.to_csv('Processed/game_player_ratings.csv', index_label=False)

# Get exactly 10 player ratings per team per game to use for our model
def pad_player_ratings(vals, n=10, pad=replacement_level):
    vals = sorted(vals, reverse=True)
    return vals[:n] + [pad for _ in range(n - len(vals))]

# Weigh top players more as they 1) play more minutes 2) do more to affect the outcome
player_weights = [.2, .15, .12, .11, .10, .09, .08, .06, .05, .04]

ratings_agg = f_multi(game_player_ratings[['id', 'teamId', 'rating']], list(['id', 'teamId']))
ratings_agg['id'] = [x[0] for x in ratings_agg.index]
ratings_agg['teamId'] = [x[1] for x in ratings_agg.index]
ratings_agg = ratings_agg.reset_index(drop=True)
ratings_agg['team_size'] = [len(x) for x in ratings_agg['rating']]
# Get list of player ratings for top 10 players for each team
# Cut off if more than 10 players listed or pad with replacement value up to 10 values
ratings_agg['ratings_cleaned'] = [pad_player_ratings(x) for x in ratings_agg['rating']]
# Take mean and weighted average for features
ratings_agg['avg_2k_rating'] = np.mean(ratings_agg['ratings_cleaned'].tolist(), axis=1)
ratings_agg['weighted_avg_2k_rating'] = np.average(ratings_agg['ratings_cleaned'].tolist(), weights=player_weights, axis=1)
ratings_agg['best_player_2k_rating'] = np.max(ratings_agg['ratings_cleaned'].to_list(), axis=1)
ratings_df = ratings_agg[['id', 'teamId', 'avg_2k_rating', 'weighted_avg_2k_rating', 'best_player_2k_rating']]

# Spot checking missing players to make sure they are not in the 2k ratings
# games[games.id ==6097]
# team_players[team_players.id == 83]
# player_ratings[player_ratings.id == 83]
# game_player_ratings[game_player_ratings.id == 6097]
# game_player_ratings.iloc[100:120, :]
# games[games.id == 3435]
# team_players[team_players.id == 739]
# team_players[team_players.id == 802]

full_features = pd.merge(full_features, ratings_df, left_on=['home_id', 'home_teamId'], right_on=['id', 'teamId'])
full_features.rename(columns={'avg_2k_rating': 'home_avg_2k_rating',
                              'weighted_avg_2k_rating':'home_weighted_avg_2k_rating',
                              'best_player_2k_rating': 'home_best_player_2k_rating'}, inplace=True)
full_features = pd.merge(full_features, ratings_df, left_on=['home_id', 'away_teamId'], right_on=['id', 'teamId'])
full_features.rename(columns={'avg_2k_rating': 'away_avg_2k_rating',
                              'weighted_avg_2k_rating':'away_weighted_avg_2k_rating',
                              'best_player_2k_rating': 'away_best_player_2k_rating'}, inplace=True)

# Get a playoff flag
games_playoff_flag = games[['id', 'seasonStage']]
games_playoff_flag['playoff_flag'] = np.where(games_playoff_flag['seasonStage'] == 4, 1, 0)
games_playoff_flag = games_playoff_flag[['id', 'playoff_flag']]
#print(games_playoff_flag.head(5))


games_odds_cleaned = games_odds[['id', 'seasonYear','hTeamId', 'vTeamId', 'commenceDate', 'hH2h', 'vH2h',
                                 'hSpreadPoints', 'vSpreadPoints', 'hSpreadOdds', 'vSpreadOdds']][games_odds['siteKey'] == 'unknown']
home_spread_points = games_odds_cleaned[['seasonYear', 'commenceDate', 'id', 'hTeamId', 'hSpreadPoints']]
away_spread_points = games_odds_cleaned[['seasonYear', 'commenceDate', 'id', 'vTeamId', 'vSpreadPoints']]
combined_spread_points = pd.concat([home_spread_points.rename(columns={'hTeamId':'teamId', 'hSpreadPoints':'spread'}),
                                    away_spread_points.rename(columns={'vTeamId':'teamId', 'vSpreadPoints':'spread'})], ignore_index=True)
combined_spread_points.sort_values(by=['commenceDate', 'teamId'], inplace=True)
combined_spread_points['spread_last10'] = combined_spread_points.groupby(by=['seasonYear','teamId'])['spread'].transform(lambda x: x.rolling(10, 1).mean())
#print(combined_spread_points.head(20))
#print(combined_spread_points.tail(20))
combined_spread_points = combined_spread_points[['id', 'teamId', 'spread_last10']]
#print(combined_spread_points.head(20))


# Remove sample that
# 1) happened in NBA Bubble environment (as home/away framework doesn't truly apply)
# 2) where both teams haven't played at least 10 games in the season (statistics are too noisy)
full_features = full_features[full_features.home_startDate <= '2020-03-11']
full_features = full_features[full_features.home_season_games_played >= 10]
full_features = full_features[full_features.away_season_games_played >= 10]

# Merge features together by merging to the existing dataframe
final_df = pd.merge(full_features, games_odds_cleaned,
                    left_on=['home_id'],
                    right_on=['id'],
                    how='inner')

final_df = pd.merge(final_df, combined_spread_points,
                    left_on=['home_id','home_teamId'],
                    right_on=['id','teamId'],
                    how='inner')
final_df.rename(columns={'spread_last10': 'home_spread_last10'}, inplace=True)
final_df = pd.merge(final_df, combined_spread_points,
                    left_on=['home_id', 'away_teamId'],
                    right_on=['id','teamId'],
                    how='inner')
final_df.rename(columns={'spread_last10': 'away_spread_last10'}, inplace=True)

final_df = pd.merge(final_df, games_playoff_flag,
                    left_on=['home_id'],
                    right_on=['id'],
                    how='inner')

cols_to_keep = ['home_team_efg_shifted',
                               'home_team_oreb_rate_shifted', 'home_team_ft_rate_shifted', 'home_team_to_rate_shifted',
                               'home_team_ha_efg_shifted', 'home_team_ha_oreb_rate_shifted', 'home_team_ha_ft_rate_shifted',
                               'home_team_ha_to_rate_shifted',
                               'home_streak_entering', 'home_streak_entering_ha', 'home_days_rest', 'home_avg_point_differential_shifted', 'home_avg_point_differential_ha_shifted',
                               'home_elo_pre', 'home_elo_prob',
                             'home_win_percentage_last10_shifted','home_win_percentage_ha_last10_shifted','home_b2b_flag',
                               'home_avg_point_differential_last10_shifted','home_avg_point_differential_last10_ha_shifted',
                             'home_team_efg_ma_shifted','home_team_oreb_rate_ma_shifted','home_team_ft_rate_ma_shifted','home_team_to_rate_ma_shifted',
                                'home_avg_2k_rating', 'home_weighted_avg_2k_rating', 'home_best_player_2k_rating', 'home_best_player_2k_rating',
                               'away_team_efg_shifted',
                               'away_team_oreb_rate_shifted', 'away_team_ft_rate_shifted', 'away_team_to_rate_shifted', 'away_team_ha_efg_shifted',
                               'away_team_ha_oreb_rate_shifted', 'away_team_ha_ft_rate_shifted', 'away_team_ha_to_rate_shifted',
                               'away_streak_entering', 'away_streak_entering_ha', 'away_days_rest', 'away_avg_point_differential_shifted', 'away_avg_point_differential_ha_shifted',
                               'away_elo_pre', 'away_elo_prob',
                             'away_win_percentage_last10_shifted','away_win_percentage_ha_last10_shifted','away_b2b_flag',
                               'away_avg_point_differential_last10_shifted','away_avg_point_differential_last10_ha_shifted',
                             'away_team_efg_ma_shifted','away_team_oreb_rate_ma_shifted','away_team_ft_rate_ma_shifted','away_team_to_rate_ma_shifted',
                                'away_avg_2k_rating', 'away_weighted_avg_2k_rating', 'away_best_player_2k_rating', 'away_best_player_2k_rating',
                            'hH2h', 'vH2h', 'home_result', 'home_plusMinus','hSpreadPoints','hSpreadOdds', 'vSpreadOdds', 'home_id',
                             'home_spread_last10', 'away_spread_last10',
                'home_season_games_played', 'away_season_games_played', 'home_seasonYear', 'home_startDate', 'playoff_flag', 'home_points', 'away_points']


final_df_for_mod = final_df[cols_to_keep]


final_df_for_mod.to_csv('Processed/base_file_for_model.csv', index_label=False)

