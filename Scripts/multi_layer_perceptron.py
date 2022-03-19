import pandas as pd
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adadelta # SGD, RMSprop, Adam
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score

# Import re-usable functions from utility folder
toolbox_path = '../Utility_Functions'
sys.path.append(toolbox_path)
from functions import mlp_model_process, correlation_reduction

# Load train and test files
os.chdir('../Data')

df_train1 = pd.read_csv('Processed/df_train1.csv')
df_test1 = pd.read_csv('Processed/df_test1.csv')

df_train2 = pd.read_csv('Processed/df_train2.csv')
df_test2 = pd.read_csv('Processed/df_test2.csv')

df_train3 = pd.read_csv('Processed/df_train3.csv')
df_test3 = pd.read_csv('Processed/df_test3.csv')

# Get features created in pre-processing
# No need to remove correlated features because MLP can handle this
features = ['home_team_efg_shifted', 'home_team_oreb_rate_shifted', 'home_team_ft_rate_shifted', 'home_team_to_rate_shifted', 'home_team_ha_efg_shifted', 'home_team_ha_oreb_rate_shifted',
       'home_team_ha_ft_rate_shifted', 'home_team_ha_to_rate_shifted', 'home_streak_entering', 'home_streak_entering_ha', 'home_days_rest', 'home_avg_point_differential_shifted',
       'home_avg_point_differential_ha_shifted', 'home_elo_pre', 'home_elo_prob', 'home_win_percentage_last10_shifted', 'home_win_percentage_ha_last10_shifted', 'home_b2b_flag',
       'home_avg_point_differential_last10_shifted', 'home_avg_point_differential_last10_ha_shifted', 'home_team_efg_ma_shifted', 'home_team_oreb_rate_ma_shifted', 'home_team_ft_rate_ma_shifted',
       'home_team_to_rate_ma_shifted', 'home_avg_2k_rating', 'home_weighted_avg_2k_rating',
            'away_team_efg_shifted', 'away_team_oreb_rate_shifted', 'away_team_ft_rate_shifted', 'away_team_to_rate_shifted', 'away_team_ha_efg_shifted',
       'away_team_ha_oreb_rate_shifted', 'away_team_ha_ft_rate_shifted', 'away_team_ha_to_rate_shifted', 'away_streak_entering', 'away_streak_entering_ha', 'away_days_rest',
       'away_avg_point_differential_shifted', 'away_avg_point_differential_ha_shifted', 'away_elo_pre', 'away_win_percentage_last10_shifted', 'away_win_percentage_ha_last10_shifted',
       'away_b2b_flag', 'away_avg_point_differential_last10_shifted', 'away_avg_point_differential_last10_ha_shifted', 'away_team_efg_ma_shifted', 'away_team_oreb_rate_ma_shifted',
       'away_team_ft_rate_ma_shifted', 'away_team_to_rate_ma_shifted', 'away_avg_2k_rating', 'away_weighted_avg_2k_rating' # Keep comma in next line to easily remove spread if needed
            ,'hSpreadPoints' # Keep point spread in model as a feature as it is known prior to game and is valuable information
            ]

# # Model 1
# # Create model using tensorflow and keras
# mod = Sequential()
# mod.add(Dense(len(features), input_dim=len(features), activation='linear')) # Linear does better than relu
#
# mod.add(Dense(1, activation='sigmoid'))
# opt = Adadelta(learning_rate=0.05, rho=0.95, epsilon=1e-07)
# #opt = SGD(learning_rate=0.01, momentum=0.1, nesterov=False)
# mod.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#
# mlp_model_process(df_train=df_train1,
#                   df_test=df_test1,
#                   features=features,
#                   target='home_result',
#                   model=mod,
#                   epochs=500,
#                   early_stopping=True,
#                   decorrelate=False,
#                   title='Model with Linear First Layer')
#
# # Model 2
# # Create model using tensorflow and keras
# mod = Sequential()
# mod.add(Dense(len(features), input_dim=len(features), activation='linear'))
# mod.add(Dense(100, activation='linear'))
# #mod.add(Dense(100, activation='relu'))
# #mod.add(Dense(50, activation='relu'))
# mod.add(Dense(1, activation='sigmoid'))
# opt = Adadelta(learning_rate=0.05, rho=0.95, epsilon=1e-07)
# #opt = SGD(learning_rate=0.01, momentum=0.1, nesterov=False)
# mod.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#
# mlp_model_process(df_train=df_train1,
#                   df_test=df_test1,
#                   features=features,
#                   target='home_result',
#                   model=mod,
#                   epochs=500,
#                   early_stopping=True,
#                   decorrelate=False,
#                   title='Model with Linear Two Layers')

# # Model 3
# # Create model using tensorflow and keras
# mod = Sequential()
# mod.add(Dense(len(features), input_dim=len(features), activation='linear'))
# mod.add(Dense(100, activation='linear'))
# mod.add(Dense(100, activation='linear'))
# mod.add(Dense(50, activation='linear'))
# mod.add(Dense(1, activation='sigmoid'))
# opt = Adadelta(learning_rate=0.02, rho=0.9, epsilon=1e-07)
# #opt = SGD(learning_rate=0.01, momentum=0.1, nesterov=False)
# mod.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#
# mlp_model_process(df_train=df_train1,
#                   df_test=df_test1,
#                   features=features,
#                   target='home_result',
#                   model=mod,
#                   epochs=500,
#                   early_stopping=True,
#                   decorrelate=False,
#                   title='Model3 with Linear Three Layers')
#
# # Model 4
# # Create model using tensorflow and keras
# mod = Sequential()
# mod.add(Dense(len(features), input_dim=len(features), activation='linear'))
# mod.add(Dense(100, activation='linear'))
# mod.add(Dense(100, activation='linear'))
# mod.add(Dense(50, activation='linear'))
# mod.add(Dense(1, activation='sigmoid'))
# opt = Adadelta(learning_rate=0.08, rho=0.95, epsilon=1e-07)
# #opt = SGD(learning_rate=0.01, momentum=0.1, nesterov=False)
# mod.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#
# mlp_model_process(df_train=df_train1,
#                   df_test=df_test1,
#                   features=features,
#                   target='home_result',
#                   model=mod,
#                   epochs=500,
#                   early_stopping=True,
#                   decorrelate=False,
#                   title='Model4 with Linear Three Layers')


# Model 5
# Create model using tensorflow and keras
mod = Sequential()
mod.add(Dense(len(features), input_dim=len(features), activation='linear'))
mod.add(Dense(100, activation='linear'))
#mod.add(Dense(100, activation='linear'))
mod.add(Dense(50, activation='linear'))
mod.add(Dense(1, activation='sigmoid'))
opt = Adadelta(learning_rate=0.001, rho=0.7, epsilon=1e-08)
#opt = SGD(learning_rate=0.01, momentum=0.1, nesterov=False)
mod.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

mlp_output1 = mlp_model_process(df_train=df_train1,
                  df_test=df_test1,
                  features=features,
                  target='home_result',
                  model=mod,
                  epochs=2000,
                  early_stopping=True,
                  decorrelate=False,
                  title='Model with Train/Test 1')

mlp_output1.to_csv('Processed/mlp_model_traintest1_output.csv', index_label=False, index=False)

# mlp_output2 = mlp_model_process(df_train=df_train2,
#                   df_test=df_test2,
#                   features=features,
#                   target='home_result',
#                   model=mod,
#                   epochs=2000,
#                   early_stopping=True,
#                   decorrelate=False,
#                   title='Model with Train/Test 2')
#
# mlp_output2.to_csv('Processed/mlp_model_traintest2_output.csv', index_label=False, index=False)
#
# mlp_output3 = mlp_model_process(df_train=df_train3,
#                   df_test=df_test3,
#                   features=features,
#                   target='home_result',
#                   model=mod,
#                   epochs=2000,
#                   early_stopping=True,
#                   decorrelate=False,
#                   title='Model with Train/Test 3')
#
# mlp_output3.to_csv('Processed/mlp_model_traintest3_output.csv', index_label=False, index=False)

# Model 6
# Create model using tensorflow and keras
# mod2 = Sequential()
# mod2.add(Dense(len(features), input_dim=len(features), activation='linear'))
# mod2.add(Dense(100, activation='linear'))
# #mod.add(Dense(100, activation='linear'))
# mod2.add(Dense(50, activation='relu'))
# mod2.add(Dense(1, activation='sigmoid'))
# opt2 = Adadelta(learning_rate=0.001, rho=0.7, epsilon=1e-08)
# #opt = SGD(learning_rate=0.01, momentum=0.1, nesterov=False)
# mod2.compile(loss='binary_crossentropy', optimizer=opt2, metrics=['accuracy'])
#
# mlp_model_process(df_train=df_train1,
#                   df_test=df_test1,
#                   features=features,
#                   target='home_result',
#                   model=mod2,
#                   epochs=2000,
#                   early_stopping=True,
#                   decorrelate=False,
#                   title='Model6 with Linear Three Layers')


# Try with feature reduction
# NOTE: This didn't help very much
# mod = Sequential()
# mod.add(Dense(len(features), input_dim=33, activation='linear'))
# mod.add(Dense(100, activation='relu'))
# #mod.add(Dense(100, activation='relu'))
# #mod.add(Dense(50, activation='relu'))
# mod.add(Dense(1, activation='sigmoid'))
# opt = Adadelta(learning_rate=0.01, rho=0.95, epsilon=1e-07)
# #opt = SGD(learning_rate=0.01, momentum=0.1, nesterov=False)
# mod.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#
# mlp_model_process(df_train=df_train1,
#                   df_test=df_test1,
#                   features=features,
#                   target='home_result',
#                   model=mod,
#                   epochs=500,
#                   early_stopping=True,
#                   decorrelate=True,
#                   threshold=0.8)


# SKLEARN - MLPClassifier
# NOTE: This does much worse than tensorflow configured model
# Version 1 without correlation reduction
# Take feature columns for our X train matrix
X_train1 = df_train1[features]
# Standardize (set to mean of 0 and standard deviation of 1) for all features
ss = StandardScaler()
ss.fit(X_train1)
X_train1 = ss.transform(X_train1)

# Get remaining features after dimension reduction and do same processing steps to X test
X_test1 = df_test1[features]
X_test1 = ss.transform(X_test1)

# Process y vector for train and test
# Encode the train vector and apply the same transformation to test for consistency
y_train1 = df_train1['home_result']
le = LabelEncoder()
le.fit(y_train1)
y_train1 = le.transform(y_train1)
y_test1 = df_test1['home_result']
y_test1 = le.transform(y_test1)

clf = MLPClassifier(random_state=408, max_iter=100)
clf.fit(X_train1, y_train1)
print(clf.score(X_test1, y_test1))
#y_pred_probs = clf.predict_proba(X_test1)[:, 1]
y_pred = clf.predict(X_test1)
#print(roc_auc_score(y_test1, y_pred_probs))
print(confusion_matrix(y_pred, y_test1))


# Version 2 with correlation reduction
# NOTE: didn't help
# Take feature columns for our X train matrix
# X_train1 = df_train1[features]
# # Remove correlated features to reduce multicollinearity in linear model
# X_train1, x_cols_kept = correlation_reduction(X_train1, threshold=0.8, verbose=False)
# # Spot check matrix to make sure there aren't any high thresholds remaining
# #print('Correlation check after feature reduction')
# #print(pd.DataFrame(X_train1).corr(method='pearson'))
# # Standardize (set to mean of 0 and standard deviation of 1) for all features
# ss = StandardScaler()
# ss.fit(X_train1)
# X_train1 = ss.transform(X_train1)
#
# # Get remaining features after dimension reduction and do same processing steps to X test
# X_test1 = df_test1[x_cols_kept]
# X_test1 = ss.transform(X_test1)
#
# # Process y vector for train and test
# # Encode the train vector and apply the same transformation to test for consistency
# y_train1 = df_train1['home_result']
# le = LabelEncoder()
# le.fit(y_train1)
# y_train1 = le.transform(y_train1)
# y_test1 = df_test1['home_result']
# y_test1 = le.transform(y_test1)
#
# clf = MLPClassifier(random_state=408, max_iter=100)
# clf.fit(X_train1, y_train1)
# print(clf.score(X_test1, y_test1))
# y_pred_probs = clf.predict_proba(X_test1)[:, 1]
# y_pred = clf.predict(X_test1)
# print(roc_auc_score(y_test1, y_pred_probs))
# print(confusion_matrix(y_pred, y_test1))

# Train: 0.690, Test: 0.663 --> 55 neurons to 1 neuron
# Restoring model weights from the end of the best epoch.
# Epoch 00136: early stopping
# Train: 0.687, Test: 0.675 --> 55 neurons to 100 neurons to 1 neuron
# Restoring model weights from the end of the best epoch.
# Epoch 00157: early stopping
# Train: 0.690, Test: 0.680  --> 55 neurons to 100 neurons to 100 neurons to 1 neuron
# /opt/anaconda3/envs/tf/lib/python3.7/site-packages/sklearn/neural_network/_multilayer_perceptron.py:696: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (100) reached and the optimization hasn't converged yet.
#   ConvergenceWarning,
# 0.6299840510366826
# [[400 293]
#  [403 785]]