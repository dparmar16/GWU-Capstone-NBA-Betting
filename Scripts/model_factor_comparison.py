import pandas as pd
import os
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adadelta # SGD, RMSprop, Adam
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, mean_squared_error

# Import re-usable functions from utility folder
toolbox_path = '../Utility_Functions'
sys.path.append(toolbox_path)
from functions import logistic_model_process, classical_model_process, mlp_model_process

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 12)

# Load train and test files
os.chdir('../Data')

df_train1 = pd.read_csv('Processed/df_train1.csv')
df_test1 = pd.read_csv('Processed/df_test1.csv')

df_train2 = pd.read_csv('Processed/df_train2.csv')
df_test2 = pd.read_csv('Processed/df_test2.csv')

# df_train3 = pd.read_csv('Processed/df_train3.csv')
# df_test3 = pd.read_csv('Processed/df_test3.csv')

# Get features created in pre-processing
# No need to remove correlated features because MLP can handle this
# features = ['home_team_efg_shifted', 'home_team_oreb_rate_shifted', 'home_team_ft_rate_shifted', 'home_team_to_rate_shifted', 'home_team_ha_efg_shifted', 'home_team_ha_oreb_rate_shifted',
#        'home_team_ha_ft_rate_shifted', 'home_team_ha_to_rate_shifted', 'home_streak_entering', 'home_streak_entering_ha', 'home_days_rest', 'home_avg_point_differential_shifted',
#        'home_avg_point_differential_ha_shifted', 'home_elo_pre', 'home_elo_prob', 'home_win_percentage_last10_shifted', 'home_win_percentage_ha_last10_shifted', 'home_b2b_flag',
#        'home_avg_point_differential_last10_shifted', 'home_avg_point_differential_last10_ha_shifted', 'home_team_efg_ma_shifted', 'home_team_oreb_rate_ma_shifted', 'home_team_ft_rate_ma_shifted',
#        'home_team_to_rate_ma_shifted', 'home_avg_2k_rating', 'home_weighted_avg_2k_rating', 'home_best_player_2k_rating',
#             'away_team_efg_shifted', 'away_team_oreb_rate_shifted', 'away_team_ft_rate_shifted', 'away_team_to_rate_shifted', 'away_team_ha_efg_shifted',
#        'away_team_ha_oreb_rate_shifted', 'away_team_ha_ft_rate_shifted', 'away_team_ha_to_rate_shifted', 'away_streak_entering', 'away_streak_entering_ha', 'away_days_rest',
#        'away_avg_point_differential_shifted', 'away_avg_point_differential_ha_shifted', 'away_elo_pre', 'away_win_percentage_last10_shifted', 'away_win_percentage_ha_last10_shifted',
#        'away_b2b_flag', 'away_avg_point_differential_last10_shifted', 'away_avg_point_differential_last10_ha_shifted', 'away_team_efg_ma_shifted', 'away_team_oreb_rate_ma_shifted',
#        'away_team_ft_rate_ma_shifted', 'away_team_to_rate_ma_shifted', 'away_avg_2k_rating', 'away_weighted_avg_2k_rating', 'away_best_player_2k_rating' # Keep comma in next line to easily remove spread if needed
#             ,'hSpreadPoints', 'home_spread_last10', 'away_spread_last10', # Keep point spread in model as a feature as it is known prior to game and is valuable information
#             'playoff_flag'
#             ]


# Test time series approach
# FBprophet was way too slow so auto-arima was used
arma = pd.read_csv('Processed/points_arma_predictions.csv')
arma = arma[['id', 'teamId', 'points_arma']]
base = pd.read_csv('Processed/base_file_for_model.csv')
base = base[base['home_season_games_played'] >= 20]
base = base[base['away_season_games_played'] >= 20]


base = pd.merge(left=base, right=arma, left_on=['home_id', 'home_teamId'], right_on=['id', 'teamId'])
base.rename(columns={'points_arma': 'home_points_arma'}, inplace=True)
base = pd.merge(left=base, right=arma, left_on=['home_id', 'away_teamId'], right_on=['id', 'teamId'])
base.rename(columns={'points_arma': 'away_points_arma'}, inplace=True)

print(base[['home_points', 'away_points', 'home_points_arma', 'away_points_arma', 'home_result']].head(10))

base['home_prediction_arma'] = np.where(base['home_points_arma'] >= base['away_points_arma'], 1, 0)

print('ARMA on 20+ games played')
print('Accuracy is: ', accuracy_score(y_true=base['home_result'], y_pred=base['home_prediction_arma']))
print('F1 score is: ', f1_score(y_true=base['home_result'], y_pred=base['home_prediction_arma']))
print('Confusion matrix is: ', confusion_matrix(y_true=base['home_result'], y_pred=base['home_prediction_arma']))

print('Home RMSE is: ', mean_squared_error(y_true=base['home_points'], y_pred=base['home_points_arma'], squared=False)) # Approximate 12
print('Away RMSE is: ', mean_squared_error(y_true=base['away_points'], y_pred=base['away_points_arma'], squared=False)) # Approximate 12

base = base[(base['home_season_games_played'] >= 50) & (base['home_season_games_played'] <= 82)]
base = base[(base['away_season_games_played'] >= 50) & (base['away_season_games_played'] <= 82)]

print('ARMA on 50-82 games played')
print('Accuracy is: ', accuracy_score(y_true=base['home_result'], y_pred=base['home_prediction_arma']))
print('F1 score is: ', f1_score(y_true=base['home_result'], y_pred=base['home_prediction_arma']))
print('Confusion matrix is: ', confusion_matrix(y_true=base['home_result'], y_pred=base['home_prediction_arma']))

print('Home RMSE is: ', mean_squared_error(y_true=base['home_points'], y_pred=base['home_points_arma'], squared=False)) # Approximate 12
print('Away RMSE is: ', mean_squared_error(y_true=base['away_points'], y_pred=base['away_points_arma'], squared=False)) # Approximate 12


features_closingline = ['hSpreadPoints'
                        ]

features_spread = ['home_spread_last10', 'away_spread_last10' #,'hSpreadPoints'
            ]

features_elo = [ 'home_elo_pre', 'home_elo_prob','away_elo_pre'
            ]

features_four_factors = ['home_team_efg_shifted', 'home_team_oreb_rate_shifted', 'home_team_ft_rate_shifted', 'home_team_to_rate_shifted',
            'away_team_efg_shifted', 'away_team_oreb_rate_shifted', 'away_team_ft_rate_shifted', 'away_team_to_rate_shifted'
            ]

features_four_factors_ma = ['home_team_efg_ma_shifted', 'home_team_oreb_rate_ma_shifted', 'home_team_ft_rate_ma_shifted', 'home_team_to_rate_ma_shifted',
            'away_team_efg_ma_shifted', 'away_team_oreb_rate_ma_shifted', 'away_team_ft_rate_ma_shifted', 'away_team_to_rate_ma_shifted'
            ]

features_2k = ['home_avg_2k_rating', 'home_weighted_avg_2k_rating', 'home_best_player_2k_rating',
               'away_avg_2k_rating', 'away_weighted_avg_2k_rating', 'away_best_player_2k_rating'
            ]

features_pointdiff = ['home_days_rest','home_avg_point_differential_shifted','home_win_percentage_last10_shifted',
                      'home_b2b_flag','home_avg_point_differential_last10_shifted', 'home_avg_point_differential_last10_ha_shifted',
                      'away_days_rest', 'away_avg_point_differential_shifted', 'away_win_percentage_last10_shifted',
                      'away_b2b_flag', 'away_avg_point_differential_last10_shifted', 'away_avg_point_differential_last10_ha_shifted',
                      'playoff_flag'
            ]

#print('Logistic Regression Results readout 1')
logistic_output1 = logistic_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_closingline, target='home_result', threshold=0.95)
#print('Logistic Regression Results readout 2')
logistic_output1 = logistic_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_spread, target='home_result', threshold=0.95)
#print('Logistic Regression Results readout 3')
logistic_output1 = logistic_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_elo, target='home_result', threshold=0.95)
#print('Logistic Regression Results readout 4')
logistic_output1 = logistic_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_four_factors, target='home_result', threshold=0.95)
#print('Logistic Regression Results readout 5')
logistic_output1 = logistic_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_four_factors_ma, target='home_result', threshold=0.95)
#print('Logistic Regression Results readout 6')
logistic_output1 = logistic_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_2k, target='home_result', threshold=0.95)
#print('Logistic Regression Results readout 7')
logistic_output1 = logistic_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_pointdiff, target='home_result', threshold=0.95)


classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_closingline, target='home_result', type='logit', threshold=0.95)

classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_spread, target='home_result', type='logit', threshold=0.95)

classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_elo, target='home_result', type='logit', threshold=0.95)

classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_four_factors, target='home_result', type='logit', threshold=0.95)

classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_four_factors_ma, target='home_result', type='logit', threshold=0.95)

classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_2k, target='home_result', type='logit', threshold=0.95)

classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_pointdiff, target='home_result', type='logit', threshold=0.95)


classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_closingline, target='home_result', type='naive-bayes', threshold=0.95)

classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_spread, target='home_result', type='naive-bayes', threshold=0.95)

classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_elo, target='home_result', type='naive-bayes', threshold=0.95)

classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_four_factors, target='home_result', type='naive-bayes', threshold=0.95)

classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_four_factors_ma, target='home_result', type='naive-bayes', threshold=0.95)

classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_2k, target='home_result', type='naive-bayes', threshold=0.95)

classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_pointdiff, target='home_result', type='naive-bayes', threshold=0.95)


features_list = [
[features_closingline, ['features_closingline']],
[features_spread, ['features_spread']],
[features_elo, ['features_elo']],
[features_four_factors, ['features_four_factors']],
[features_four_factors_ma, ['features_four_factors_ma']],
[features_2k, ['features_2k']],
[features_pointdiff, ['features_pointdiff']]
]

# Test neural network as well
for feature_set in features_list:
    features = feature_set[0]
    feature_set_name = feature_set[1]
    print(f'Running MLP for feature set: {feature_set_name}')
    mod = Sequential()
    mod.add(Dense(len(features), input_dim=len(features), activation='linear'))
    mod.add(Dense(100, activation='linear'))
    mod.add(Dense(50, activation='linear'))
    mod.add(Dense(1, activation='sigmoid'))
    opt = Adadelta(learning_rate=0.001, rho=0.7, epsilon=1e-08)
    mod.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    mlp_output1 = mlp_model_process(df_train=df_train1,
                      df_test=df_test1,
                      features=features,
                      target='home_result',
                      model=mod,
                      epochs=2000,
                      early_stopping=True,
                      decorrelate=False,
                      title=f'Model with {feature_set_name}')
