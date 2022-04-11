# get team strength for each team-season using points
#
# point difference home, home team year, away team year
#
# apply kalman filter to it
#
# home team, away team, closing line
# one hot home team

import pandas as pd
import os

os.chdir('../Data')
df = pd.read_csv('Processed/base_file_for_model.csv')

one hot home team id
one hot away team id
keep closing line

do regression on this using each unique game date to separate the loop

pull each team coefficient as of each game date to use in the model
