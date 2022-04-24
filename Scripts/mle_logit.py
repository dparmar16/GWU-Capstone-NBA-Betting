import pandas as pd
import os
import numpy as np
import sys
import statsmodels.tools
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.discrete.discrete_model import Logit, Probit
from sklearn.naive_bayes import GaussianNB

# Import re-usable functions from utility folder
toolbox_path = '../Utility_Functions'
sys.path.append(toolbox_path)
from functions import correlation_reduction


# Set formatting preferences for debugging in Pycharm editor
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 12)

# Load file
os.chdir('../Data')
df_train1 = pd.read_csv('Processed/df_train1.csv')
df_test1 = pd.read_csv('Processed/df_test1.csv')

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


# Statsmodels - Logit
print(x_cols_kept)
X_train1 = statsmodels.tools.add_constant(X_train1, has_constant='add')
logit_mod = Logit(endog=y_train1, exog=X_train1)
logit_result = logit_mod.fit()
X_test1 = statsmodels.tools.add_constant(X_test1, has_constant='add')
logit_probs = logit_result.predict(X_test1)
logit_preds = np.where(logit_probs >= 0.5, 1, 0)

print(confusion_matrix(y_true=y_test1, y_pred=logit_preds))
print(accuracy_score(y_true=y_test1, y_pred=logit_preds))
print(f1_score(y_true=y_test1, y_pred=logit_preds))

# Statsmodels - Probit
probit_mod = Probit(endog=y_train1, exog=X_train1)
probit_result = probit_mod.fit()
probit_probs = probit_result.predict(X_test1)
probit_preds = np.where(probit_probs >= 0.5, 1, 0)

print(confusion_matrix(y_true=y_test1, y_pred=probit_preds))
print(accuracy_score(y_true=y_test1, y_pred=probit_preds))
print(f1_score(y_true=y_test1, y_pred=probit_preds))



# Naive Bayes
clf = GaussianNB()
clf.fit(X=X_train1, y=y_train1)
nb_preds = clf.predict(X=X_test1)
print(confusion_matrix(y_true=y_test1, y_pred=nb_preds))
print(accuracy_score(y_true=y_test1, y_pred=nb_preds))
print(f1_score(y_true=y_test1, y_pred=nb_preds))


# Every possible model hitting between 66-68 percent
# It's not the model, it's the features going into it

# Possible option is mle library - fit equation of logit
# Can't do MLE with mle library as that requires an error term to be Normal/Uniform