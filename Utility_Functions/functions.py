# Functions used across repository
# Use cases include exploratory data analysis, preprocessing, etc.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import statsmodels
from statsmodels.discrete.discrete_model import Logit
from sklearn.naive_bayes import GaussianNB

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

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


# Translate odds to probability
# https://help.smarkets.com/hc/en-gb/articles/214058369-How-to-calculate-implied-probability-in-betting
# This will help us find mispriced market prices to bet on
def odds_to_implied_prob(value):

    assert (type(value) == float) | (type(value) == int), 'Value must be integer or float'

    if value < 0:
        prob = -1 * value / (-1 * value + 100) # Minus odds
    else:
        prob = 100 / (value + 100) # Plus odds

    return prob

# Formula for conversion here:
# http://www.betsmart.co/odds-conversion-formulas/#americantodecimal
def american_converter_to_decimal(american):

    assert (type(american) == float) | (type(american) == int), 'Value must be integer or float'

    if american < 0:
        return 1 + (100 / - american) # Minus odds
    else:
        return (american / 100) + 1 # Plus odds

def kelly_criterion(model_prob_win, market_odds, multiplier=1, max=None):
    # First convert to decimal odds
    # But b in this function is market return, which is decimal - 1
    b = american_converter_to_decimal(market_odds) - 1
    p, q = model_prob_win, 1- model_prob_win

    # Calculate kelly criterion but also multply by multiplier
    # For example, in "half kelly" we would bet half the recommendation
    f = multiplier * (b * p - q) / b

    # If max bet percentage is given, ensure we are not betting more than that
    # But value should not be less than 0
    if max is not None:
        out_pre = np.where(f > max, max, f)
        out = np.where(out_pre < 0, 0, out_pre)
        return out
    else:
        return np.where(f < 0, 0, f)

# Correlation Reduction function takes in a dataframe
# It finds correlated columns and drops one of them
# The goal is to reduce multi-collinearity that causes model issues
def correlation_reduction(dataset, threshold, verbose=True):
   col_corr = set()  # Set of all the names of deleted columns
   corr_matrix = dataset.corr()
   for i in range(len(corr_matrix.columns)):
      for j in range(i):
         if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
            colname = corr_matrix.columns[i]  # getting the name of column
            col_corr.add(colname)
            if verbose:
                print(corr_matrix.columns[i] + ' - ' + corr_matrix.columns[j])
            if colname in dataset.columns:
               del dataset[colname]  # deleting the column from the dataset

   cols_kept = pd.Series(dataset.columns).values
   return dataset, cols_kept

# Run logistic regression given data, features, target, and multicollinearity threshold
# Returns performance metrics such as f1-score and classifcation matrix
def logistic_model_process(df_train, df_test, features, target, threshold=0.75):
    # Take feature columns for our X train matrix
    X_train = df_train[features]
    # Remove correlated features to reduce multicollinearity in linear model
    X_train, x_cols_kept = correlation_reduction(X_train, threshold=threshold, verbose=False)
    # Standardize (set to mean of 0 and standard deviation of 1) for all features
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)

    # Get remaining features after dimension reduction and do same processing steps to X test
    X_test = df_test[x_cols_kept]
    X_test = ss.transform(X_test)

    # Process y vector for train and test
    y_train = df_train[target]
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)

    y_test = df_test[target]
    y_test = le.transform(y_test)

    basemod = LogisticRegression(fit_intercept=True)
    basemod.fit(X_train, y_train)
    y_pred = basemod.predict(X_test)
    print(f'Logistic regression output for target {target} and features {features}.')
    print('F1 score is',f1_score(y_pred, y_test))
    print('Accuracy score is', accuracy_score(y_pred, y_test))
    print('Confusion matrix is\n', confusion_matrix(y_pred, y_test))

    # Get relevant columns for output
    df_out = df_test[['home_id', 'home_startDate', 'home_result', 'hH2h', 'vH2h']]
    # Plan is to use model prediction to compare to market prices
    # Get predictions and attach to rest of relevant data
    y_pred_proba = basemod.predict_proba(X_test)
    df_out['logistic_pred_home'] = y_pred_proba[:, 1]
    df_out['logistic_pred_away'] = 1 - df_out['logistic_pred_home']

    # Also want implied probability of winning given by odds to compare to our model
    df_out['home_implied_prob'] = df_out['hH2h'].map(lambda x: odds_to_implied_prob(x))
    df_out['away_implied_prob'] = df_out['vH2h'].map(lambda x: odds_to_implied_prob(x))

    # Sort by date
    df_out.sort_values(by='home_startDate', axis=0, ascending=True, inplace=True)

    return df_out

def classical_model_process(df_train, df_test, features, target, type, threshold=0.75, pca=False):

    # If PCA is true, then return PCA features as way to reduce dimensionality
    if pca:
        pass # Insert here
    # If PCA is false, then reduce dimensionality by removing highly correlated columns
    # This preserves the original features for explainability purposes
    else:
        # Take feature columns for our X train matrix
        X_train = df_train[features]
        # Remove correlated features to reduce multicollinearity in linear model
        X_train, x_cols_kept = correlation_reduction(X_train, threshold=threshold, verbose=False)
        # Standardize (set to mean of 0 and standard deviation of 1) for all features
        ss = StandardScaler()
        ss.fit(X_train)
        X_train = ss.transform(X_train)

        # Get remaining features after dimension reduction and do same processing steps to X test
        X_test = df_test[x_cols_kept]
        X_test = ss.transform(X_test)

    # Process y vector for train and test
    y_train = df_train[target]
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)

    y_test = df_test[target]
    y_test = le.transform(y_test)

    if type == 'logistic':
        basemod = LogisticRegression(fit_intercept=True)
        basemod.fit(X_train, y_train)
        y_pred = basemod.predict(X_test)
        print(f'Logistic regression output for target {target} and features {features}.')
        print('F1 score is', f1_score(y_pred, y_test))
        print('Accuracy score is', accuracy_score(y_pred, y_test))
        print('Confusion matrix is\n', confusion_matrix(y_pred, y_test))

    if type == 'logit':
        X_train1 = statsmodels.tools.add_constant(X_train, has_constant='add')
        logit_mod = Logit(endog=y_train, exog=X_train1)
        logit_result = logit_mod.fit()
        X_test1 = statsmodels.tools.add_constant(X_test, has_constant='add')
        logit_probs = logit_result.predict(X_test1)
        logit_preds = np.where(logit_probs >= 0.5, 1, 0)
        print(f'Logit MLE output for target {target} and features {features}.')
        print('F1 score is', f1_score(y_true=y_test, y_pred=logit_preds))
        print('Accuracy score is', accuracy_score(y_true=y_test, y_pred=logit_preds))
        print('Confusion matrix is\n', confusion_matrix(y_true=y_test, y_pred=logit_preds))


    if type == 'naive-bayes':
        clf = GaussianNB()
        clf.fit(X=X_train, y=y_train)
        nb_preds = clf.predict(X=X_test)
        print(f'Gaussian Naive Bayes output for target {target} and features {features}.')
        print('F1 score is', f1_score(y_true=y_test, y_pred=nb_preds))
        print('Accuracy score is', accuracy_score(y_true=y_test, y_pred=nb_preds))
        print('Confusion matrix is\n', confusion_matrix(y_true=y_test, y_pred=nb_preds))

    return None

# Function to run MLP model given data, features, target, and model parameters
# Returns model metrics such as:
# 1) loss function (binary cross entropy)
# 2) performance metric (accuracy)
def mlp_model_process(df_train, df_test, features, target, model, epochs,
                      early_stopping=True, decorrelate=False, threshold=0.75, title=None):
    # Take feature columns for our X train matrix
    # No need to drop due to multi-collinearity
    # Use all features available
    X_train = df_train[features]
    X_test = df_test[features]

    # Include option to remove multi-collinearity features
    if decorrelate:
        # Remove correlated features to reduce multicollinearity in linear model
        X_train, x_cols_kept = correlation_reduction(X_train, threshold=threshold, verbose=False)
        X_test = X_test[x_cols_kept]

    # Need to standardize features as neural networks expect such data
    # Fit on train and transform both train and test
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

    # Get target value
    y_train = df_train[target]
    y_test = df_test[target]

    # Encode target value
    # Fit on train and transform both train and test
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    # Fit the model given
    # Early stopping is a callback
    # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    if early_stopping:
        es = EarlyStopping(monitor='val_accuracy',
                           min_delta=0.0001,
                           restore_best_weights=True,
                           patience=1000,
                           mode='max',
                           verbose=1)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=epochs, verbose=0, callbacks=[es])
    else:
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=epochs, verbose=0)

    # Look at metrics
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    # plot loss during training
    # Binary Cross entropy loss
    # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
    plt.subplot(211)
    if title == None:
        plt.title('Loss')
    else:
        plt.title(f'{title} Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    if title == None:
        plt.title('Accuracy')
    else:
        plt.title(f'{title} Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

    print('\n')


    # Get relevant columns for output
    df_out = df_test[['home_id', 'home_startDate', 'home_result', 'hH2h', 'vH2h']]
    # Plan is to use model prediction to compare to market prices
    # Get predictions and attach to rest of relevant data
    y_pred = model.predict(X_test)

    df_out['mlp_pred_home'] = y_pred
    df_out['mlp_pred_away'] = 1 - df_out['mlp_pred_home']

    # Also want implied probability of winning given by odds to compare to our model
    df_out['home_implied_prob'] = df_out['hH2h'].map(lambda x: odds_to_implied_prob(x))
    df_out['away_implied_prob'] = df_out['vH2h'].map(lambda x: odds_to_implied_prob(x))

    # Sort by date
    df_out.sort_values(by='home_startDate', axis=0, ascending=True, inplace=True)

    # Print metrics
    y_pred_for_scores = np.where(y_pred >= 0.5, 1, 0)
    print('Ran MLP model')
    print('Accuracy score is', accuracy_score(y_pred_for_scores, y_test))
    print('F1 score is', f1_score(y_pred_for_scores, y_test))

    return df_out

# Group columns values into a list
# https://stackoverflow.com/questions/22219004/how-to-group-dataframe-rows-into-list-in-pandas-groupby/66018377#66018377
# Use this to combine video game ratings by team into one column
def f_multi(df, col_names):
    if not isinstance(col_names, list):
        col_names = [col_names]

    values = df.sort_values(col_names).values.T

    col_idcs = [df.columns.get_loc(cn) for cn in col_names]
    other_col_names = [name for idx, name in enumerate(df.columns) if idx not in col_idcs]
    other_col_idcs = [df.columns.get_loc(cn) for cn in other_col_names]

    # split df into indexing colums(=keys) and data colums(=vals)
    keys = values[col_idcs, :]
    vals = values[other_col_idcs, :]

    # list of tuple of key pairs
    multikeys = list(zip(*keys))

    # remember unique key pairs and ther indices
    ukeys, index = np.unique(multikeys, return_index=True, axis=0)

    # split data columns according to those indices
    arrays = np.split(vals, index[1:], axis=1)

    # resulting list of subarrays has same number of subarrays as unique key pairs
    # each subarray has the following shape:
    #    rows = number of non-grouped data columns
    #    cols = number of data points grouped into that unique key pair

    # prepare multi index
    idx = pd.MultiIndex.from_arrays(ukeys.T, names=col_names)

    list_agg_vals = dict()
    for tup in zip(*arrays, other_col_names):
        col_vals = tup[:-1]  # first entries are the subarrays from above
        col_name = tup[-1]  # last entry is data-column name

        list_agg_vals[col_name] = col_vals

    df2 = pd.DataFrame(data=list_agg_vals, index=idx)
    return df2

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