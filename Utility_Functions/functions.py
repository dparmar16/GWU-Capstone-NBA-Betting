# Functions used across repository
# Use cases include exploratory data analysis, preprocessing, etc.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re


def Cal_rolling_mean_var(timeseries):
    rolling_mean = list()
    rolling_var = list()

    for i in range(1, len(timeseries)):
        rolling_mean.append(np.mean(timeseries[0:i]))
        rolling_var.append(np.var(timeseries[0:i]))

    fig, axs = plt.subplots(2)
    fig.suptitle('Rolling mean and variance')
    axs[0].plot(rolling_mean, c='red')
    axs[0].set_title('Rolling Mean')
    axs[1].plot(rolling_var, c='green')
    axs[1].set_title('Rolling Variance')
    plt.show()

    return rolling_mean, rolling_var

def efg(fgm, fga, tpm):
    efg = (float(fgm) + 0.5 * float(tpm)) / float(fga)
    return efg

def oreb_rate(offReb, fgm, fga):
    offReb_rate = float(offReb) / (float(fga) - float(fgm))
    return offReb_rate

def ft_rate(fta, fga):
    ft_rate = float(fta) / float(fga)
    return ft_rate

def to_rate(fga, to, fta, offReb):
    # Possession calculation
    # https://www.nbastuffer.com/analytics101/possession/
    # 0.96*[(Field Goal Attempts)+(Turnovers)+0.44*(Free Throw Attempts)-(Offensive Rebounds)]
    poss = 0.96 * (fga + to + 0.44 * fta - offReb)
    to_rate = float(to) / float(poss)
    return to_rate

def four_factors_averages():
    # Season averages
    # https://www.nba.com/stats/teams/four-factors/?sort=OREB_PCT&dir=-1&Season=2020-21&SeasonType=Regular%20Season
    # https://www.basketball-reference.com/about/factors.html
    # efg, ftr, tov, oreb
    four_factors_season_avg = [
        [2014, .494, .269, .147, .289],
        [2015, .501, .271, .145, .280],
        [2016, .512, .275, .141, .273],
        [2017, .518, .254, .145, .271],
        [2018, .527, .257, .144, .269],
        [2019, .532, .263, .144, .267],
        [2020, .540, .242, .136, .267],
    ]

    four_factors_baseline_df = pd.DataFrame(four_factors_season_avg,
                                            columns=['year', 'efg_baseline', 'ftr_baseline',
                                                     'tov_baseline', 'oreb_baseline'])

    return four_factors_baseline_df

# Correlation Reduction function takes in a dataframe
# It finds correlated columns and drops one of them
# The goal is to reduce multi-collinearity that causes model issues
def correlation_reduction(dataset, threshold):
   col_corr = set()  # Set of all the names of deleted columns
   corr_matrix = dataset.corr()
   for i in range(len(corr_matrix.columns)):
      for j in range(i):
         if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
            colname = corr_matrix.columns[i]  # getting the name of column
            col_corr.add(colname)
            print(corr_matrix.columns[i] + ' - ' + corr_matrix.columns[j])
            if colname in dataset.columns:
               del dataset[colname]  # deleting the column from the dataset

   cols_kept = pd.Series(dataset.columns).values
   return dataset, cols_kept

# For betting data only
# Combine raw files into a single file for processing
def combine_files(path: str):
    '''

    :param path: a directory with all raw odds data
    :return: one combined file of all games (with season var)
    '''

    df = pd.DataFrame()

    os.chdir(path)

    files_list = glob.glob(os.getcwd() + "/nba_odds_20" + "*.xlsx")
    for file in files_list:
        file_years = [x for x in re.findall('\d{4}', file)]
        temp = pd.read_excel(file)
        temp['season'] = file_years[0] # Or do min? file_years.min()
        df = df.append(temp, ignore_index=True)

    min_year = df['season'].min()
    max_year = df['season'].max()
    df.to_excel(f'nba_combined_odds_{min_year}-{max_year}.xlsx')
    return df


# Function to process raw betting data
# https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm
# Write function to extract spread from open_h, close_h, open_v, close_v
# Logic to identify favorite and get to home_spread, away_spread, home_ml, away_ml
# https://stackoverflow.com/questions/16236684/apply-pandas-function-to-column-to-create-multiple-new-columns
def get_home_line_and_total(Close_h, Close_v) -> float:
    '''

    :param Close_h: Column within game dataframe, signifies spread or total depending on context
    :param Close_v: Column within game dataframe, signifies spread or total depending on context
    :return home_line: Home line (number of points that home team is favored by in specific game)
    :return total: Total points to bet on for over/under
    '''

    # Slim the dataset for easier handling and processing
    #df = df[['Open_h','Close_h','Open_v','Close_v']]

    # These columns could have either the spread or the game point totals
    # Game point totals are very large (180 or more)
    # Game point spreads are much smaller (30 or less)

    # Close is the only thing that matters, find the lower number
    # If the lower number is in home side, it's home favored by that amount
    # If the lower number is in away side, it's away favored by that amount
    # Spreads are not provided with minus in front, i.e. favored by 4 points is 4 not -4
    # Need to correct this in this function
    if not(Close_h >= 0 and Close_v >= 0):
        raise ValueError(f'Invalid input values for get_home_line_and_total function. Close_h is {Close_h} and Close_v is {Close_v}.')
    if Close_h <= Close_v:
        home_line = -1 * Close_h
        total = Close_v
    else:
        home_line = Close_v
        total = Close_h

    return home_line, total

# Function to process raw betting data
# https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm
# Cleaning function
def cleaning(df: pd.DataFrame) -> pd.DataFrame:
    '''

    :param df: Raw dataframe of spreads, totals, odds, etc
    :return clean: Cleaned dataframe with essential information as season, date, home spread, etc
    '''

    # Replace PK in open/close
    # Cast open/close to a float
    df['Open'] = df['Open'].replace('pk', 0).replace('PK', 0)
    df['Close'] = df['Close'].replace('pk', 0).replace('PK', 0)
    df = df.astype({'Open': 'float', 'Close': 'float'})

    # Replace Nan values in Open and Close with the value in the other column
    # This makes the assumption that the other was excluded because the value didn't change
    # There was only one NaN value found in exploratory data analysis, so this is safe for now
    df.Close.fillna(df.Open, inplace=True)
    df.Open.fillna(df.Close, inplace=True)

    # Get separate h (home team data) and v (visiting team data)
    # Add suffix to column names
    # Join h and v
    # For names that are on neutral site, we use index (since v and h rows always alternate)
    # The index that is even is the visitor
    #print(df.head(5))

    dfv = df[(df['VH'] == 'V')| ((df['VH'] == 'N') & (df.index % 2 == 0))]
    dfv = dfv.add_suffix('_v')
    dfv.reset_index(inplace=True)
    dfh = df[(df['VH'] == 'H') | ((df['VH'] == 'N') & (df.index % 2 == 1))]
    dfh = dfh.add_suffix('_h')
    dfh.reset_index(inplace=True)
    game = pd.concat([dfv, dfh], axis=1)

    # Obtain or derive all metrics needed for each game
    # Apply home line function to get home line, away line, home line odds, away line odds, total for over/under
    game['home_spread'], game['total_points_over_under'] = zip(*game.apply(lambda x: get_home_line_and_total(x['Close_h'], x['Close_v']), axis=1))
    game['away_spread'] = -1 * game['home_spread']

    # Assume that betting on spread and over/under has return of -110
    game['home_spread_odds'], game['away_spread_odds'] = -110, -110


    # Want actual date, not just month-day
    game['calendar_year'] = np.where(game['Date_v'] <= 1012, game['season_v'].astype(int) + 1, game['season_v'].astype(int))
    game['game_date'] = game['calendar_year'].astype(str) + game['Date_v'].astype(str).str.zfill(4)


    # Remove all unnecessary columns from the output (ones that won't go in DynamoDB)
    game.rename(columns={'season_v': 'season'}, inplace=True)
    game = game[['season','game_date', 'Team_v', 'Team_h', 'away_spread', 'away_spread_odds', 'home_spread', 'home_spread_odds', 'ML_v', 'ML_h', 'total_points_over_under', 'Final_v', 'Final_h']]

    # Sort data by date for easy troubleshooting
    game.sort_values(by=['game_date'], axis=0, inplace=True)

    # Write final dataset to Excel
    min_year = game['season'].min()
    max_year = game['season'].max()
    game.to_excel(f'nba_cleaned_odds_{min_year}-{max_year}.xlsx')

    return game