import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.api import OLS
from statsmodels.tools import add_constant

# Ignore one specific warning we know this script will encounter
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Approach here
# 1) get team strength for each team-season using closing spread (shown to be highly predictive)
# 2) Optional next step: apply kalman filter to it

# Reference Links
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
# https://datascience.stackexchange.com/questions/71804/how-to-perform-one-hot-encoding-on-multiple-categorical-columns
# https://datascience.stackexchange.com/questions/94038/how-to-create-multi-hot-encoding-from-a-list-column-in-dataframe

### This is Proof of Concept, actual function used is in utilities ###

# os.chdir('../Data')
# df = pd.read_csv('Processed/base_file_for_model.csv')
#
# # Test for one season to validate concept
# # Columns needed: home team, away team, closing line
# df = df[['home_teamId', 'away_teamId', 'hSpreadPoints', 'home_seasonYear', 'playoff_flag']]
# df = df[df['home_seasonYear'] == 2015]
#
# # Stack home and away team id into a array format
# # This sets up the multilabel binarizer
# df['both_teams'] = [[x, y] for x, y in zip(df.home_teamId, df.away_teamId)]
#
# mlb = MultiLabelBinarizer()
# mlb.fit(df['both_teams'])
# #col_names = ['team' + str(x) for x in mlb.classes_]
# col_names = [str(x) for x in mlb.classes_]
#
#
# encoded = pd.DataFrame(mlb.fit_transform(df['both_teams']), columns=col_names)
#
# mod_df_pre = pd.concat([df, encoded], axis=1)
#
# # flip to negative for home team since negative number means home team is stronger
# # ie. home team id is 10 so mod_df_pre[str(10)] = -1 * value (1)
# # ie. home team id is 25 so mod_df_pre[str(25)] = -1 * value (1)
# for idx, row in mod_df_pre.iterrows():
#        home_team_value = row['home_teamId']
#        mod_df_pre.loc[idx, f'{str(home_team_value)}'] = -1
#
#
#
# mod_df_pre.drop([
#        'home_teamId',
#        'away_teamId',
#        'home_seasonYear',
#        'both_teams'], axis=1, inplace=True)
#
# mod_df = add_constant(mod_df_pre, prepend=False, has_constant='add')
# Y = mod_df['hSpreadPoints']
# X = mod_df.drop(['hSpreadPoints'], axis=1)
# mod = OLS(Y, X)
# results = mod.fit()
# print(results.params)
# print(type(results.params))
# print(results.params['11'])

# Explanation of process
# Load in data and get relevant columns
os.chdir('../Data')
df = pd.read_csv('Processed/base_file_for_model.csv')
df = df[['home_startDate', 'home_teamId', 'away_teamId', 'hSpreadPoints', 'home_seasonYear', 'playoff_flag']]

# Create empty list to append team coefficients to using the upcoming loop
team_coef_list = []


# Get unique season values to use in loop
# Then initiate loop
season_values = df['home_seasonYear'].unique()

for season in season_values:
       df_season = df[df['home_seasonYear'] == season]
       season_dates = np.sort(df_season['home_startDate'].unique())
       for date_until in season_dates:
              df_date = df_season[df_season['home_startDate'] <= date_until]

              # Zip home and away team to do multi-label binarizer (i.e. get [1, 3] when team 1 plays team 3]
              df_date['both_teams'] = [[x, y] for x, y in zip(df_date.home_teamId, df_date.away_teamId)]

              # Encode team values, so if team 1 plays team 3
              # team1 team2 team3
              #   1     0     1
              mlb = MultiLabelBinarizer()
              mlb.fit(df_date['both_teams'])
              col_names = [str(x) for x in mlb.classes_]
              encoded = pd.DataFrame(mlb.fit_transform(df_date['both_teams']), columns=col_names)

              # Join together encoded values back to original data to set up our regression
              mod_df_pre = pd.concat([df_date.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)

              # flip to negative for home team since negative number means home team is stronger
              # ie. home team id is 10 so mod_df_pre[str(10)] = -1 * value (1)
              # ie. home team id is 25 so mod_df_pre[str(25)] = -1 * value (1)
              for idx, row in mod_df_pre.iterrows():
                     home_team_value = row['home_teamId']
                     mod_df_pre.loc[idx, f'{str(home_team_value)}'] = -1


              mod_df_pre.drop([
                     'home_teamId',
                     'away_teamId',
                     'home_seasonYear',
                     'both_teams',
              'home_startDate'], axis=1, inplace=True)

              # Add constant for home court advantage and do OLS regression since spread is continuous varible
              # OLS coefficients refer to how each team impacts the spread
              # team1 team2 team3 const    y
              #   1     0     1    1      -5
              mod_df = add_constant(mod_df_pre, prepend=False, has_constant='add')
              Y = mod_df['hSpreadPoints']
              X = mod_df.drop(['hSpreadPoints', 'playoff_flag'], axis=1) # Can choose to add playoff flag later
              mod = OLS(Y, X)
              results = mod.fit()
              params_series = results.params

              # date, team id, coefficient
              for i, v in params_series.items():
                     team_coef_list.append([season, date_until, i, v])


team_coef_df = pd.DataFrame(team_coef_list, columns=['season', 'date', 'teamId', 'coef'])