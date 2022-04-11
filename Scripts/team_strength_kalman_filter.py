import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.api import OLS
from statsmodels.tools import add_constant

# Approach here
# 1) get team strength for each team-season using closing spread (shown to be highly predictive)
# 2) Optional next step: apply kalman filter to it

# Reference Links
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
# https://datascience.stackexchange.com/questions/71804/how-to-perform-one-hot-encoding-on-multiple-categorical-columns
# https://datascience.stackexchange.com/questions/94038/how-to-create-multi-hot-encoding-from-a-list-column-in-dataframe

os.chdir('../Data')
df = pd.read_csv('Processed/base_file_for_model.csv')

# Columns needed: home team, away team, closing line
df = df[['home_teamId', 'away_teamId', 'hSpreadPoints', 'home_seasonYear']]
df = df[df['home_seasonYear'] == 2015]

# Stack home and away team id into a array format
# This sets up the multilabel binarizer
df['both_teams'] = [[x, y] for x, y in zip(df.home_teamId, df.away_teamId)]

mlb = MultiLabelBinarizer()
mlb.fit(df['both_teams'])
col_names = ['team' + str(x) for x in mlb.classes_]

encoded = pd.DataFrame(mlb.fit_transform(df['both_teams']), columns=col_names)

mod_df_pre = pd.concat([df, encoded], axis=1).drop(['home_teamId', 'away_teamId', 'home_seasonYear',
       'both_teams'], axis=1)

mod_df = add_constant(mod_df_pre, prepend=False, has_constant='add')
Y = mod_df['hSpreadPoints']
X = mod_df.drop(['hSpreadPoints'], axis=1)
mod = OLS(Y, X)
results = mod.fit()
print(results.params)



# fit on all
# get unique dates in a season and loop through using all games up to that date
# transform on a limited subset
#
# linear regression on all the labels and the closing spread
# get team specific coefficient at each date -- put into a pandas dataframe or dict or other
#
# save these as features to use in a one off model to get f1 score and log loss

