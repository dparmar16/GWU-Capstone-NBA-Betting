import shap
import pandas as pd
import os
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from matplotlib import colors

# Import re-usable functions from utility folder
toolbox_path = '../Utility_Functions'
sys.path.append(toolbox_path)
from functions import correlation_reduction, logistic_model_process


# Set formatting preferences for debugging in Pycharm editor
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 12)

# Load file
os.chdir('../Data')
df = pd.read_csv('Processed/base_file_for_model.csv')

# Load train-test splits
# Split 1 is earlier seasons and later seasons
# Split 2 is first half/second half of all seasons
# Split 3 is randomly shuffled
df_train1 = pd.read_csv('Processed/df_train1.csv')
df_test1 = pd.read_csv('Processed/df_test1.csv')
df_train2 = pd.read_csv('Processed/df_train2.csv')
df_test2 = pd.read_csv('Processed/df_test2.csv')
df_train3 = pd.read_csv('Processed/df_train2.csv')
df_test3 = pd.read_csv('Processed/df_test3.csv')

# Get feature list
features = ['home_team_efg_shifted','home_team_oreb_rate_shifted', 'home_team_ft_rate_shifted', 'home_team_to_rate_shifted',
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

# Do exploratory data analysis on features that could be standardizated
# All of these look somewhat normally distributed so we can do standard scaling
# plt.hist(df['home_streak_entering'], bins=50)
# plt.show()
#
# plt.hist(df['home_days_rest'], bins=50)
# plt.show()
#
# plt.hist(df['home_avg_point_differential_shifted'], bins=50)
# plt.show()
#
# plt.hist(df['home_elo_pre'], bins=50)
# plt.show()

# Do PCA analysis
#Fitting the PCA algorithm with our Data
x_scaled = StandardScaler().fit_transform(df_train1[features])
pca = PCA().fit(x_scaled)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title(f'Explained Variance in Original Dataset with {len(features)} features')
plt.show()

# Take feature columns for our X train matrix
X_train1 = df_train1[features]
# Remove correlated features to reduce multicollinearity in linear model
X_train1, x_cols_kept = correlation_reduction(X_train1, threshold=0.8)
# Spot check matrix to make sure there aren't any high thresholds remaining
print('Correlation check after feature reduction')
print(pd.DataFrame(X_train1).corr(method='pearson'))
# Standardize (set to mean of 0 and standard deviation of 1) for all features
ss = StandardScaler()
ss.fit(X_train1)
X_train1 = ss.transform(X_train1)

#Fitting the PCA algorithm with our correlation reduced dataset
pca = PCA().fit(X_train1)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title(f'Explained Variance in Reduced Dataset with {len(x_cols_kept)} features')
plt.show()

# Get remaining features after dimension reduction and do same processing steps to X test
X_test1 = df_test1[x_cols_kept]
X_test1 = ss.transform(X_test1)

# Process y vector for train and test
# Encode the train vector and apply the same transformation to test for consistency
y_train1 = df_train1['home_result']
le = LabelEncoder()
le.fit(y_train1)
y_train1 = le.transform(y_train1)
y_test1 = df_test1['home_result']
y_test1 = le.transform(y_test1)


# Default parameters of l2 regularization and lbfgs solver work the best
# for type in [['l1', 'liblinear'],['l1', 'saga'], ['l2', 'lbfgs'],['l2', 'newton-cg'], ['elasticnet', 'saga'], ['none', 'lbfgs']]:
#    print('=' * 25)
#    print(f'Running model with penalty {type[0]} and solver {type[1]}')
#    basemod = LogisticRegression(penalty=type[0], solver=type[1], max_iter=200, l1_ratio=0.5)
#    basemod.fit(X_train1, y_train1)
#    y_pred = basemod.predict(X_test1)
#    print(f1_score(y_pred, y_test1))
#    print(confusion_matrix(y_pred, y_test1))
#    print('=' * 25)


# Now that the best model has been identified, run it and get test metrics such as f1 score
basemod = LogisticRegression()
basemod.fit(X_train1, y_train1)
y_pred = basemod.predict(X_test1)
print(f1_score(y_pred, y_test1))
print(confusion_matrix(y_pred, y_test1))

# Use shap to see how each feature affects model output
#https://shap.readthedocs.io/en/latest/index.html

explainer = shap.Explainer(basemod, X_train1, feature_names=x_cols_kept)
shap_values = explainer(X_test1)

shap.plots.beeswarm(shap_values)

shap.summary_plot(shap_values, X_test1,
                  plot_type='dot',
                  max_display=len(x_cols_kept),
                  plot_size=(30, 14))

shap.plots.bar(shap_values, max_display=len(x_cols_kept))

# Get feature importances
# https://github.com/slundberg/shap/issues/632
feature_importances_logistic = np.abs(shap_values.values).mean(0)

plt.bar(x_cols_kept,feature_importances_logistic)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

feature_importance_sorted = pd.DataFrame(list(zip(x_cols_kept,feature_importances_logistic)),columns=['col_name','feature_importance_vals'])
feature_importance_sorted.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
plt.bar(feature_importance_sorted['col_name'],feature_importance_sorted['feature_importance_vals'])
plt.xticks(rotation=90)
plt.ylabel('SHAP Mean Value')
plt.title('SHAP Feature Importance Sorted after Dimensionality Reduction')
plt.tight_layout()
plt.show()

# Run second and third models using function to streamline process
# Add first model as well for consistency of result reporting
print('Results readout 1')
logistic_output1 = logistic_model_process(df_train=df_train1, df_test=df_test1,
                       features=features, target='home_result', threshold=0.8)
logistic_output1.to_csv('Processed/logistic_model_traintest1_output.csv', index_label=False, index=False)

print('Results readout 2')
logistic_model_process(df_train=df_train2, df_test=df_test2,
                       features=features, target='home_result', threshold=0.8)
print('Results readout 3')
logistic_model_process(df_train=df_train3, df_test=df_test3,
                       features=features, target='home_result', threshold=0.8)

# Get to f1 score of over 75% on shuffled dataset when including closing line
# F1 score is 0.7506024096385542
# Accuracy score is 0.6854103343465046
# Log-loss score is 0.6024870801478707
# Confusion matrix is
#  [[279 148]
#  [266 623]]



# # Financial forecast
# modfinal = LogisticRegression(penalty='none', solver='lbfgs')
# modfinal.fit(X_train, y_train)
#
# # Look at prediction versus key inputs
# print('PREDICITON METRIC SANITY CHECK')
# print(modfinal.predict_proba(X_test))
# print(X_test[['home_team_efg_shifted',
# 'home_avg_point_differential_shifted',
#               'away_team_efg_shifted',
#               'away_avg_point_differential_shifted']].head(1))
#
# predicted_probs = [x[1] for x in modfinal.predict_proba(X_test)]
# result = y_test_all['home_result']
#
# # Translate odds to probability
# # https://help.smarkets.com/hc/en-gb/articles/214058369-How-to-calculate-implied-probability-in-betting
#
# print('ODDS VERSUS PERCENTAGE')
# print(y_test_all['hH2h'])
# home_prob = np.where(y_test_all['hH2h'] < 0, -1 * y_test_all['hH2h'] / (-1 * y_test_all['hH2h'] + 100), 100 / (y_test_all['hH2h'] + 100))
# print(home_prob)
# def odds_to_implied_prob(value):
#     try:
#         np.where(value < 0,
#                  -1 * value / (-1 * value + 100),  # Minus odds
#                  100 / (value + 100)) # Plus odds
#     except ZeroDivisionError:
#         print(value < 0)
#         print('The value causing an error is',value)
#
#     if value < 0:
#         prob = -1 * value / (-1 * value + 100) # Minus odds
#     else:
#         prob = 100 / (value + 100) # Plus odds
#
#     return prob
#
# mycmap = colors.ListedColormap(['green', 'red'])
# plt.scatter(home_prob, predicted_probs, c=result, cmap=mycmap, s=16)
# plt.xlabel('Market Implied Home Win Probability')
# plt.ylabel('Model Implied Home Win Probability')
# plt.title('Market vs Model on Moneyline (Using Spread as Feature)')
# textstr = 'Red is home win \nGreen is away win'
# plt.text(0.7, 0.2, textstr, fontsize=11,
#         verticalalignment='bottom')
# plt.show()
