import pandas as pd
import numpy as np
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adadelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, brier_score_loss, log_loss



# Import re-usable functions from utility folder
toolbox_path = '../Utility_Functions'
sys.path.append(toolbox_path)
from functions import logistic_model_process, classical_model_process, mlp_model_process, odds_to_implied_prob

# Set formatting preferences for debugging in Pycharm editor
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 12)

# Load train-test splits
os.chdir('../Data')
# Split 1 is earlier seasons and later seasons
# Split 2 is first half/second half of all seasons
df_train1 = pd.read_csv('Processed/df_train1.csv')
df_test1 = pd.read_csv('Processed/df_test1.csv')
df_train2 = pd.read_csv('Processed/df_train2.csv')
df_test2 = pd.read_csv('Processed/df_test2.csv')

# Get features to use in models
# We are excluding closing home spread as it is highly predictive and will be used in its own model
features_minus_spread = ['home_team_efg_shifted','home_team_oreb_rate_shifted', 'home_team_ft_rate_shifted', 'home_team_to_rate_shifted',
            'home_team_ha_efg_shifted', 'home_team_ha_oreb_rate_shifted', 'home_team_ha_ft_rate_shifted', 'home_team_ha_to_rate_shifted',
            'home_streak_entering', 'home_streak_entering_ha', 'home_days_rest', 'home_avg_point_differential_shifted', 'home_avg_point_differential_ha_shifted',
            'home_elo_pre', 'home_elo_prob',
            'home_win_percentage_last10_shifted','home_win_percentage_ha_last10_shifted',
            'home_b2b_flag',
            'home_avg_point_differential_last10_shifted','home_avg_point_differential_last10_ha_shifted',
            'home_team_efg_ma_shifted','home_team_oreb_rate_ma_shifted','home_team_ft_rate_ma_shifted','home_team_to_rate_ma_shifted',
            'home_avg_2k_rating', 'home_weighted_avg_2k_rating', 'home_best_player_2k_rating',
            'home_team_coef_unweighted', 'home_team_coef_weighted',
            'home_spread_last10',
            'away_team_efg_shifted', 'away_team_oreb_rate_shifted', 'away_team_ft_rate_shifted', 'away_team_to_rate_shifted',
            'away_team_ha_efg_shifted', 'away_team_ha_oreb_rate_shifted', 'away_team_ha_ft_rate_shifted', 'away_team_ha_to_rate_shifted',
            'away_streak_entering', 'away_streak_entering_ha', 'away_days_rest', 'away_avg_point_differential_shifted', 'away_avg_point_differential_ha_shifted',
            'away_elo_pre',
            'away_win_percentage_last10_shifted','away_win_percentage_ha_last10_shifted',
            'away_b2b_flag',
            'away_avg_point_differential_last10_shifted','away_avg_point_differential_last10_ha_shifted',
            'away_team_efg_ma_shifted','away_team_oreb_rate_ma_shifted','away_team_ft_rate_ma_shifted','away_team_to_rate_ma_shifted',
            'away_avg_2k_rating', 'away_weighted_avg_2k_rating', 'away_best_player_2k_rating',
            'away_team_coef_unweighted', 'away_team_coef_weighted',
            'away_spread_last10',
            #'hSpreadPoints',
            'playoff_flag']

# Get first two models for ensemble
# Logistic with all features but closing line
# Logistic with closing line alone
logistic_output1 = logistic_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_minus_spread, target='home_result', threshold=0.8)
logistic_output2 = logistic_model_process(df_train=df_train1, df_test=df_test1,
                       features=['hSpreadPoints'], target='home_result', threshold=0.8)

# Create MLP model with all features minus spread (same as logistic) and save output
mod = Sequential()
mod.add(Dense(len(features_minus_spread), input_dim=len(features_minus_spread), activation='linear'))
mod.add(Dense(100, activation='linear'))
mod.add(Dense(50, activation='linear'))
mod.add(Dense(1, activation='sigmoid'))
opt = Adadelta(learning_rate=0.001, rho=0.7, epsilon=1e-08)
mod.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

mlp_output1 = mlp_model_process(df_train=df_train1,
                  df_test=df_test1,
                  features=features_minus_spread,
                  target='home_result',
                  model=mod,
                  epochs=1000,
                  early_stopping=True,
                  decorrelate=False,
                  title='Model with Train/Test 1')
mlp_output1.to_csv('Processed/mlp_df_for_ensemble.csv', index=False, index_label=False)
# mlp_output1 = pd.read_csv('Processed/mlp_df_for_ensemble.csv') # Avoid re-running model in future

# Also do logit MLE (probabilistic model)
logitmle_output1 = classical_model_process(df_train=df_train1, df_test=df_test1,
                       features=features_minus_spread, target='home_result', type='logit', threshold=0.8)

# Approach
# Combine these four models into one df
# Put into 1) logistic 2) random forest
# See if ensemble metrics such as f1, log loss, and brier score loss are higher
# print(logistic_output1.head(5))
# print(logistic_output2.head(5))
# print(mlp_output1.head(5))
# print(logitmle_output1.head(5))

# Get a single df for ensemble
# home_id, home_startDate, home_result, logistic pred home 1, logistic pred home 2, mlp pred home, logit pred home
ensemble_input = pd.merge(logistic_output1[['home_id', 'home_startDate', 'home_result',
                                            'hH2h', 'vH2h', 'logistic_pred_home']],
                          logistic_output2.rename({'logistic_pred_home':'logistic2_pred_home'}, axis=1)[['home_id','logistic2_pred_home']],
                          on='home_id',
                          how='inner')

ensemble_input = pd.merge(ensemble_input,
                          logitmle_output1[['home_id','logit_mle_pred_home']],
                          on='home_id',
                          how='inner')

ensemble_input = pd.merge(ensemble_input,
                          mlp_output1[['home_id','mlp_pred_home']],
                          on='home_id',
                          how='inner')
ensemble_input['home_implied_prob'] = ensemble_input['hH2h'].map(lambda x: odds_to_implied_prob(x))
ensemble_input['away_implied_prob'] = ensemble_input['vH2h'].map(lambda x: odds_to_implied_prob(x))
ensemble_input.to_csv('Processed/ensemble_model_df.csv', index=False, index_label=False)

features = ['logistic_pred_home',  'logistic2_pred_home',  'logit_mle_pred_home',  'mlp_pred_home']
target = ['home_result']

X_train, X_test, y_train, y_test = train_test_split(
    ensemble_input[features],
    ensemble_input[target],
    train_size=0.7)

logistic_ensemble = LogisticRegression(fit_intercept=True)
logistic_ensemble.fit(X_train, np.ravel(y_train))
y_pred = logistic_ensemble.predict(X_test)
y_pred_proba = logistic_ensemble.predict_proba(X_test)
y_pred_proba_for_metrics = [float(x) for x in y_pred_proba[:, 1]]
print(f'Logistic regression output for target {target} and features {features}.')
print('F1 score is', f1_score(y_pred, y_test))
print('Accuracy score is', accuracy_score(y_pred, y_test))
print('Log-loss score is', log_loss(y_true=y_test, y_pred=y_pred_proba_for_metrics))
print('Brier score loss is', brier_score_loss(y_true=y_test, y_prob=y_pred_proba_for_metrics))
print('Confusion matrix is\n', confusion_matrix(y_pred, y_test))

rf_ensemble = RandomForestClassifier()
rf_ensemble.fit(X_train, np.ravel(y_train))
y_pred = logistic_ensemble.predict(X_test)
y_pred_proba = logistic_ensemble.predict_proba(X_test)
y_pred_proba_for_metrics = [float(x) for x in y_pred_proba[:, 1]]
print(f'Random Forest output for target {target} and features {features}.')
print('F1 score is', f1_score(y_pred, y_test))
print('Accuracy score is', accuracy_score(y_pred, y_test))
print('Log-loss score is', log_loss(y_true=y_test, y_pred=y_pred_proba_for_metrics))
print('Brier score loss is', brier_score_loss(y_true=y_test, y_prob=y_pred_proba_for_metrics))
print('Confusion matrix is\n', confusion_matrix(y_pred, y_test))

# Cross validate
# Use the same metrics as above
# https://scikit-learn.org/stable/modules/model_evaluation.html
logistic_ensemble = LogisticRegression(fit_intercept=True)
scores_f1 = cross_val_score(logistic_ensemble, ensemble_input[features], np.ravel(ensemble_input[target]),
                            cv=5, scoring='f1')
scores_accuracy = cross_val_score(logistic_ensemble, ensemble_input[features], np.ravel(ensemble_input[target]),
                            cv=5, scoring='accuracy')
scores_brier = cross_val_score(logistic_ensemble, ensemble_input[features], np.ravel(ensemble_input[target]),
                            cv=5, scoring='neg_brier_score')
scores_logloss = cross_val_score(logistic_ensemble, ensemble_input[features], np.ravel(ensemble_input[target]),
                            cv=5, scoring='neg_log_loss')

print('Logistic Ensemble')
print('F1 cross validation')
print(scores_f1)
print('Accuracy cross validation')
print(scores_f1)
print('Brier cross validation')
print(scores_brier)
print('Log Loss cross validation')
print(scores_logloss)

# Do the same with Random Forest
rf_ensemble = RandomForestClassifier()
scores_f1 = cross_val_score(rf_ensemble, ensemble_input[features], np.ravel(ensemble_input[target]),
                            cv=5, scoring='f1')
scores_accuracy = cross_val_score(rf_ensemble, ensemble_input[features], np.ravel(ensemble_input[target]),
                            cv=5, scoring='accuracy')
scores_brier = cross_val_score(rf_ensemble, ensemble_input[features], np.ravel(ensemble_input[target]),
                            cv=5, scoring='neg_brier_score')
scores_logloss = cross_val_score(rf_ensemble, ensemble_input[features], np.ravel(ensemble_input[target]),
                            cv=5, scoring='neg_log_loss')

print('Random Forest Ensemble')
print('F1 cross validation')
print(scores_f1)
print('Accuracy cross validation')
print(scores_f1)
print('Brier cross validation')
print(scores_brier)
print('Log Loss cross validation')
print(scores_logloss)