import pandas as pd
import os
import sys
import numpy as np


import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set preferences for Pycharm editor
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 12)

# Import re-usable functions from utility folder
toolbox_path = '../Utility_Functions'
sys.path.append(toolbox_path)
from functions import american_converter_to_decimal, kelly_criterion

# Load logistic and MLP files (the two models created)
os.chdir('../Data')
logistic = pd.read_csv('Processed/logistic_model_traintest1_output.csv')
mlp = pd.read_csv('Processed/mlp_model_traintest1_output.csv')

# Check that each dataset has the same number of records and game id's
print(logistic.shape)
print(logistic['home_id'].nunique())
print(mlp.shape)
print(mlp['home_id'].nunique())

# Merge the two together into one file for betting simulation
mlp_slim = mlp[['home_id', 'mlp_pred_home', 'mlp_pred_away']]
combined = pd.merge(logistic, mlp_slim, on=['home_id'])
# Check that data is correct to proceed
print(combined.head(5))

#    home_id home_startDate  home_result   hH2h   vH2h  logistic_pred_home  logistic_pred_away  home_implied_prob  away_implied_prob  mlp_pred_home  mlp_pred_away
# 0     4529     2018-11-05            0 -150.0  130.0            0.574808            0.425192           0.600000           0.434783       0.638044       0.361956
# 1     4541     2018-11-07            0  100.0 -120.0            0.429815            0.570185           0.500000           0.545455       0.329435       0.670565
# 2     4547     2018-11-07            0  400.0 -550.0            0.286634            0.713366           0.200000           0.846154       0.141408       0.858592
# 3     4548     2018-11-07            1 -270.0  220.0            0.764635            0.235365           0.729730           0.312500       0.768728       0.231272
# 4     4546     2018-11-07            1 -550.0  400.0            0.814913            0.185087           0.846154           0.200000       0.797763       0.202237

# figure out rule set to bet on and see return
# 1) Use MLP, Logistic or both? --> 1) avg of both 2) pessimistic model (less delta) 3) optimistic model (more delta)
# 2) Use absolute delta (2% market to 4% model is 2% edge), relative delta (2% to 4% is considered 100% edge)
# 3) Wager size - Kelly criterion, half kelly, static percentage

combined['home_opt'] = combined.apply(lambda x: max(x.logistic_pred_home, x.mlp_pred_home), axis=1)
combined['home_avg'] = combined.apply(lambda x: np.mean([x.logistic_pred_home, x.mlp_pred_home]), axis=1)
combined['home_pes'] = combined.apply(lambda x: min(x.logistic_pred_home, x.mlp_pred_home), axis=1)

combined['away_opt'] = combined.apply(lambda x: max(x.logistic_pred_away, x.mlp_pred_away), axis=1)
combined['away_avg'] = combined.apply(lambda x: np.mean([x.logistic_pred_away, x.mlp_pred_away]), axis=1)
combined['away_pes'] = combined.apply(lambda x: min(x.logistic_pred_away, x.mlp_pred_away), axis=1)

def betting_deployment(df,
                       model_optimism,
                       bet_strategy,
                       kelly_multiplier=1,
                       percentage_stake=0.01,
                       bankroll_amount=1,
                       alpha_type='relative', # 'absolute' or 'relative'
                       alpha_threshold=0.075,
                       max_percentage=0.1,
                       ):
    '''

    :param df: Dataframe of betting odds and information
    :param model_optimism: Based on various models, choose most aggressive, mean, or conservative
        Aggressive: Choose model which has highest "alpha" or mismatch to market price
        Mean: Average out difference between models and market price
        Conservative: Choose model which has lowest "alpha" or mismatch to market price
    :param bet_strategy: kelly or percentage stake
    :param kelly_multiplier: If using kelly, what percent of kelly recommendation to bet
    :param percentage_stake: If using percentage stake strategy, what percent to bet each time
    :param bankroll_amount: Amount of money to start with, used to track return

    :return: result: daily change in capital
    '''
    # 'home_opt'
    # 'home_avg'
    # 'home_pes'
    # 'away_opt'
    # 'away_avg'
    # 'away_pes'
    # 'home_implied_prob'
    # 'away_implied_prob'
    # 'home_startDate'
    # 'home_result'
    # 'hH2h' #--> to decimal
    # 'vH2h' #--> to decimal

    # df['home_opt_edge'] = df['home_opt'] - df['home_implied_prob']
    # df['home_avg_edge'] = df['home_avg'] - df['home_implied_prob']
    # df['home_pes_edge'] = df['home_pes'] - df['home_implied_prob']
    #
    # df['home_opt_edge_relative'] = (df['home_opt'] - df['home_implied_prob']) / df['home_implied_prob']
    # df['home_avg_edge_relative'] = (df['home_avg'] - df['home_implied_prob']) / df['home_implied_prob']
    # df['home_pes_edge_relative'] = (df['home_pes'] - df['home_implied_prob']) / df['home_implied_prob']
    #
    # df['away_opt_edge'] = df['away_opt'] - df['away_implied_prob']
    # df['away_avg_edge'] = df['away_avg'] - df['away_implied_prob']
    # df['away_pes_edge'] = df['away_pes'] - df['away_implied_prob']
    #
    # df['away_opt_edge_relative'] = (df['away_opt'] - df['away_implied_prob']) / df['away_implied_prob']
    # df['away_avg_edge_relative'] = (df['away_avg'] - df['away_implied_prob']) / df['away_implied_prob']
    # df['away_pes_edge_relative'] = (df['away_pes'] - df['away_implied_prob']) / df['away_implied_prob']

    print(df.columns)
    # Convert odds to decimal to use in betting
    df['hH2h_decimal'] = df['hH2h'].map(lambda x: american_converter_to_decimal(x))
    df['vH2h_decimal'] = df['vH2h'].map(lambda x: american_converter_to_decimal(x))
    # Save initial amount for graphing purposes
    initial_bankroll = bankroll_amount

    if model_optimism == 'opt':
        df['home_prob'] = df['home_opt']
        df['home_edge'] = df['home_opt'] - df['home_implied_prob']
        df['home_edge_relative'] = (df['home_opt'] - df['home_implied_prob']) / df['home_implied_prob']
        df['away_prob'] = df['away_opt']
        df['away_edge'] = df['away_opt'] - df['away_implied_prob']
        df['away_edge_relative'] = (df['away_opt'] - df['away_implied_prob']) / df['away_implied_prob']
    elif model_optimism == 'avg':
        df['home_prob'] = df['home_avg']
        df['home_edge'] = df['home_avg'] - df['home_implied_prob']
        df['home_edge_relative'] = (df['home_avg'] - df['home_implied_prob']) / df['home_implied_prob']
        df['away_prob'] = df['away_avg']
        df['away_edge'] = df['away_avg'] - df['away_implied_prob']
        df['away_edge_relative'] = (df['away_avg'] - df['away_implied_prob']) / df['away_implied_prob']
    elif model_optimism == 'pes':
        df['home_prob'] = df['home_pes']
        df['home_edge'] = df['home_pes'] - df['home_implied_prob']
        df['home_edge_relative'] = (df['home_pes'] - df['home_implied_prob']) / df['home_implied_prob']
        df['away_prob'] = df['away_pes']
        df['away_edge'] = df['away_pes'] - df['away_implied_prob']
        df['away_edge_relative'] = (df['away_pes'] - df['away_implied_prob']) / df['away_implied_prob']

    earliest_date = min(df['home_startDate'])
    daybeforestart = (datetime.strptime(earliest_date, '%Y-%m-%d') - timedelta(1)).strftime('%Y-%m-%d')
    capital_list = []
    capital_list.append([daybeforestart, initial_bankroll])

    # Combine all betting strategies into same daily loop
    # Another way is to combined kelly and percentage into same loop
    for dt in sorted(df['home_startDate'].unique()):
        daily = df[df['home_startDate'] == dt]
        daily_return = 0
        for idx, data in daily.iterrows():
            if bet_strategy == 'kelly':
                kelly_home = kelly_criterion(model_prob_win=data['home_prob'], market_odds=data['hH2h'],
                                             multiplier=kelly_multiplier, max=max_percentage)
                kelly_away = kelly_criterion(model_prob_win=data['away_prob'], market_odds=data['vH2h'],
                                             multiplier=kelly_multiplier, max=max_percentage)
                if kelly_home > 0 and kelly_away <= 0:
                    wager_amount = bankroll_amount * kelly_home
                    home_money_outcome = np.where(data['home_result'] == 1,
                                                  data['hH2h_decimal'] * wager_amount,
                                                  -1 * wager_amount)
                    daily_return += home_money_outcome
                if kelly_away > 0 and kelly_home <= 0:
                    wager_amount = bankroll_amount * kelly_away
                    away_money_outcome = np.where(data['home_result'] == 0,
                                                  data['vH2h_decimal'] * wager_amount,
                                                  -1 * wager_amount)
                    daily_return += away_money_outcome
            if bet_strategy == 'percentage':
                # Wager the percentage given, but make sure it's not more than the max allowed
                wager_amount = np.where(percentage_stake > max_percentage, max_percentage, percentage_stake)
                if alpha_type == 'relative':
                    home_alpha = data['home_edge_relative']
                    away_alpha = data['away_edge_relative']
                elif alpha_type == 'absolute':
                    home_alpha = data['home_edge']
                    away_alpha = data['away_edge']
                if home_alpha >= alpha_threshold and away_alpha < alpha_threshold:
                    home_money_outcome = np.where(data['home_result'] == 1,
                                                  data['hH2h_decimal'] * wager_amount,
                                                  -1 * wager_amount)
                    daily_return += home_money_outcome
                if away_alpha >= alpha_threshold and home_alpha < alpha_threshold:
                    away_money_outcome = np.where(data['home_result'] == 0,
                                                  data['vH2h_decimal'] * wager_amount,
                                                  -1 * wager_amount)
                    daily_return += away_money_outcome

        bankroll_amount -= daily_return
        capital_list.append([dt, bankroll_amount])

    result = pd.DataFrame(capital_list, columns=['date', 'capital'])

    # # 1. loop through each day --> get individual date block from df
    # # 2. find all positive kelly and make those bets but cap it
    # # find positive kelly bets, but cap the bet
    # if bet_strategy == 'kelly':
    #     for dt in sorted(df['home_startDate'].unique()):
    #         daily = df[df['home_startDate'] == dt]
    #         daily_return = 0
    #         print('the daily df is')
    #         print(daily)
    #         for idx, data in daily.iterrows():
    #             kelly_home = kelly_criterion(model_prob_win=data['home_prob'], market_odds=data['hH2h'],
    #                                          multiplier=kelly_multiplier, max=max_percentage)
    #             kelly_away = kelly_criterion(model_prob_win=data['away_prob'], market_odds=data['vH2h'],
    #                                          multiplier=kelly_multiplier, max=max_percentage)
    #             print('The data is')
    #             print(data)
    #             print('Kelly home is', kelly_home)
    #             print('Kelly away is', kelly_away)
    #             print(data['home_result'])
    #             if kelly_home > 0 and kelly_away <= 0:
    #                 wager_amount = bankroll_amount * kelly_home
    #                 home_money_outcome = np.where(data['home_result'] == 1,
    #                                    data['hH2h_decimal'] * wager_amount,
    #                                    -1 * wager_amount)
    #                 daily_return += home_money_outcome
    #                 print('Home money delta is', home_money_outcome)
    #             if kelly_away > 0 and kelly_home <= 0:
    #                 wager_amount = bankroll_amount * kelly_away
    #                 away_money_outcome = np.where(data['home_result'] == 0,
    #                                               data['vH2h_decimal'] * wager_amount,
    #                                               -1 * wager_amount)
    #                 daily_return += away_money_outcome
    #                 print('Away money delta is', away_money_outcome)
    #             print('\n\n\n')
    #
    #         bankroll_amount -= daily_return
    #         capital_list.append([dt, bankroll_amount])
    #
    #     result = pd.DataFrame(capital_list, columns=['date', 'capital'])
    #
    # if bet_strategy == 'percentage':
    #     pass
    #     # 1. loop through each day --> get individual date block from df
    #     # 2. set market alpha threshold and bet set percentage (1%) on all matches that meet criteria

    plt.plot(result['date'], result['capital'])
    plt.xlabel('Date')
    plt.ylabel('Capital')
    if bet_strategy == 'kelly':
        plt.title(f'Return on {initial_bankroll} with Kelly strategy with Kelly ratio {kelly_multiplier}.'
                  f' \nMax bet {max_percentage}.')
    if bet_strategy == 'percentage':
        plt.title(f'Return on {initial_bankroll} with percentage strategy with percentage {percentage_stake}.'
                  f' \nMax bet {max_percentage} and alpha type {alpha_type} and alpha threshold {alpha_threshold}.')
                  # f'\nBetting strategy is {bet_strategy}.')
    plt.show()

    return result


betting_deployment(df=combined[combined['home_startDate'] <= '2018-11-10'],
                   model_optimism='avg',
                   bet_strategy='kelly',
                   kelly_multiplier=0.5,
                   percentage_stake=0.01,
                   bankroll_amount=1,
                   alpha_type='relative',  # 'absolute' or 'relative'
                   alpha_threshold=0.075,
                   max_percentage=0.01,
                   )

# betting_deployment(df=combined[combined['home_startDate'] <= '2020-11-10'],
#                    model_optimism='avg',
#                    bet_strategy='kelly',
#                    kelly_multiplier=0.5,
#                    percentage_stake=0.01,
#                    bankroll_amount=1,
#                    alpha_type='relative',  # 'absolute' or 'relative'
#                    alpha_threshold=0.075,
#                    max_percentage=0.05,
#                    )

betting_deployment(df=combined[combined['home_startDate'] <= '2018-11-10'],
                   model_optimism='avg',
                   bet_strategy='percentage',
                   kelly_multiplier=0.5,
                   percentage_stake=0.01,
                   bankroll_amount=1,
                   alpha_type='relative',  # 'absolute' or 'relative'
                   alpha_threshold=0.075,
                   max_percentage=0.01,
                   )

betting_deployment(df=combined[combined['home_startDate'] <= '2019-10-01'],
                   model_optimism='avg',
                   bet_strategy='percentage',
                   kelly_multiplier=0.5,
                   percentage_stake=0.01,
                   bankroll_amount=1,
                   alpha_type='absolute',  # 'absolute' or 'relative'
                   alpha_threshold=0.05,
                   max_percentage=0.01,
                   )

betting_deployment(df=combined[combined['home_startDate'] <= '2019-10-01'],
                   model_optimism='pes',
                   bet_strategy='percentage',
                   kelly_multiplier=0.5,
                   percentage_stake=0.01,
                   bankroll_amount=1,
                   alpha_type='absolute',  # 'absolute' or 'relative'
                   alpha_threshold=0.05,
                   max_percentage=0.01,
                   )