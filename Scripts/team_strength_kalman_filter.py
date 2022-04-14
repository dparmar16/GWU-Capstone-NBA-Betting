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



# fit on all
# get unique dates in a season and loop through using all games up to that date
# transform on a limited subset
#
# linear regression on all the labels and the closing spread
# get team specific coefficient at each date -- put into a pandas dataframe or dict or other
#
# save these as features to use in a one off model to get f1 score and log loss

# create a df
# for season in seasons:
#        for date in season_dates:
#               keep games in that season up to that date
#               run this process
#               write to a DF
#
# get a df with date on rows and team id on columns
#
# stack to get df with date and teamid on rows and coefficient in a column

os.chdir('../Data')
df = pd.read_csv('Processed/base_file_for_model.csv')
df = df[['home_startDate', 'home_teamId', 'away_teamId', 'hSpreadPoints', 'home_seasonYear', 'playoff_flag']]

df['both_teams'] = [[x, y] for x, y in zip(df.home_teamId, df.away_teamId)]

mlb = MultiLabelBinarizer()
mlb.fit(df['both_teams'])
col_names = [str(x) for x in mlb.classes_]

season_values = df['home_seasonYear'].unique()

team_coef_list = []

for season in season_values:
       df_season = df[df['home_seasonYear'] == season]
       season_dates = np.sort(df_season['home_startDate'].unique())
       for date_until in season_dates:
              df_date = df_season[df_season['home_startDate'] <= date_until]
              # print(date_until)

              df_date['both_teams'] = [[x, y] for x, y in zip(df_date.home_teamId, df_date.away_teamId)]
              # print('date shape',df_date.shape)

              mlb = MultiLabelBinarizer()
              mlb.fit(df_date['both_teams'])
              col_names = [str(x) for x in mlb.classes_]
              encoded = pd.DataFrame(mlb.fit_transform(df_date['both_teams']), columns=col_names)
              # print('encoded shape', encoded.shape)

              mod_df_pre = pd.concat([df_date.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
              # print('mod shape',mod_df_pre.shape)

              # flip to negative for home team since negative number means home team is stronger
              # ie. home team id is 10 so mod_df_pre[str(10)] = -1 * value (1)
              # ie. home team id is 25 so mod_df_pre[str(25)] = -1 * value (1)
              for idx, row in mod_df_pre.iterrows():
                     home_team_value = row['home_teamId']
                     mod_df_pre.loc[idx, f'{str(home_team_value)}'] = -1

              # print('mod shape',mod_df_pre.shape)

              mod_df_pre.drop([
                     'home_teamId',
                     'away_teamId',
                     'home_seasonYear',
                     'both_teams',
              'home_startDate'], axis=1, inplace=True)

              mod_df = add_constant(mod_df_pre, prepend=False, has_constant='add')
              Y = mod_df['hSpreadPoints']
              X = mod_df.drop(['hSpreadPoints', 'playoff_flag'], axis=1) # Can choose to add playoff flag later
              # print(X.columns)
              # print(len(Y))
              # print(len(df_date['hSpreadPoints']))
              # print(X.shape)
              mod = OLS(Y, X)
              results = mod.fit()

              params_series = results.params
              # print(season, date_until, params_series.index[0], params_series.values[0])
              # print(params_series)
              # print(list(range(len(params_series))))

              # date, team id, coefficient
              for i, v in params_series.items():
                     team_coef_list.append([season, date_until, i, v])



print(team_coef_list)
team_coef_df = pd.DataFrame(team_coef_list, columns=['season', 'date', 'teamId', 'coef'])
print(team_coef_df)