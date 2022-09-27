### Set up environment
import pandas as pd
import shap
import sys
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np


### Utility functions
# Import re-usable functions from utility folder
toolbox_path = '../Utility_Functions'
sys.path.append(toolbox_path)
from functions import remove_collinear_features

### Load in data
df_pre = pd.read_csv('../Data/Processed/base_file_for_model.csv')
df_pre["home_b2b_flag"] = df_pre["home_b2b_flag"].astype(np.float64)
df_pre["away_b2b_flag"] = df_pre["away_b2b_flag"].astype(np.float64)
df_pre["playoff_flag"] = df_pre["playoff_flag"].astype(np.float64)
df_pre.drop([
             'hSpreadPoints', 'hSpreadOdds', 'vSpreadOdds', 'hH2h', 'vH2h', 'home_plusMinus',
             'home_id', 'home_seasonYear', 'home_startDate', 'home_points',
       'away_points', 'home_teamId', 'away_teamId',
    'home_season_games_played', 'away_season_games_played', 'home_seasonYear'], axis=1, inplace=True)

### Dimensionality reduction
# Remove correlated columns
df = remove_collinear_features(df_model=df_pre,
                               target_var='home_result',
                               threshold=0.8,
                               verbose=True)

### Modeling
X = df.drop(['home_result'], axis=1)
y_pre = df['home_result']
le = LabelEncoder()
y = le.fit_transform(y_pre)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, shuffle=True,
                                                    random_state=42)

# Run a shap on models: logistic regression, random forest, xgboost
# RFC = RandomForestClassifier()
# XGB = GradientBoostingClassifier()
LR = LogisticRegression(solver='lbfgs', max_iter=2000)
LR.fit(X_train, y_train)
explainer = shap.Explainer(model=LR, masker=shap.maskers.Independent(data=X_train),
                           algorithm='linear', feature_names=X_train.columns.values, seed=408)
explainer.expected_value # 0.5499750231704764
shap_values = explainer(X_train)

# SHAP EXAMPLE
plt.figure(constrained_layout=True)
plt.title('Test Title')
shap.plots.bar(shap_values[0],
               max_display=10)
plt.show()
# Identify the game that is train[0]
# Show the values of the features that shap is using to drive prediction
# Show that they are above/below average which is why the are contributing
# Show that the game date is past the 2k date and that info is out of date
# Show that the prediction is wrong
LR.predict_proba([X_train.iloc[0, :].array])

shap_values[0].values.sum() + shap_values[0].base_values

# Force plot example
# https://medium.com/mlearning-ai/shap-force-plots-for-classification-d30be430e195
# https://shap-lrjball.readthedocs.io/en/latest/generated/shap.force_plot.html
plt.figure(constrained_layout=True)
shap.force_plot(explainer.expected_value,
                shap_values.values[0], #[0][2],
                np.array(X_train.iloc[0,:]), #.iloc[0, :].array,
                feature_names=X_train.columns.values,
                link='logit',
                matplotlib=True,
                text_rotation=15)
plt.show()


# SHAP EXAMPLE #2
plt.figure(constrained_layout=True)
shap.plots.bar(shap_values[1], max_display=10)
plt.show()
# Identify the game that is train[1]
# Show the values of the features that shap is using to drive prediction
# Show that they are above/below average which is why the are contributing
plt.figure(constrained_layout=True)
shap.force_plot(explainer.expected_value,
                shap_values.values[1], #[0][2],
                np.array(X_train.iloc[1,:]), #.iloc[0, :].array,
                feature_names=X_train.columns.values,
                matplotlib=True,
                link='logit',
                text_rotation=15)
plt.show()

# Dependence plot
shap.dependence_plot('home_days_rest', shap_values,
                     np.array(X_train.values),#.astype(np.float64),#np.array(X_train).astype(np.float64),
                     feature_names=X_train.columns.values)

# LATER ON
# show which of the reduced features are truly driving the prediction
# show with point spread
# show without point spread
# take out features that are not driving much action, keep going

# Get three individual predictions to show why it was made
# See how prediction matches the home implied swim probability

def model_proba(x):
    return LR.predict_proba(x)[:,1]

plt.tight_layout()
fig,ax = shap.partial_dependence_plot(
    "away_days_rest", model_proba, X_train, model_expected_value=True,
    feature_expected_value=True, show=False, ice=False
)
plt.show()

plt.tight_layout()
fig,ax = shap.partial_dependence_plot(
    "away_streak_entering", model_proba, X_train, model_expected_value=True,
    feature_expected_value=True, show=False, ice=False
)
plt.show()

plt.tight_layout()
fig,ax = shap.partial_dependence_plot(
    "home_elo_prob", model_proba, X_train, model_expected_value=True,
    feature_expected_value=True, show=False, ice=True
)
ax.set_title('Partial Dependence of Home Win Probability on Elo Win Probability',
          fontsize=12)
plt.show()

plt.tight_layout()
fig,ax = shap.partial_dependence_plot(
    "home_team_coef_weighted", model_proba, X_train, model_expected_value=True,
    feature_expected_value=True, show=False, ice=False
)
plt.show()

plt.tight_layout()
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
plt.show()


plt.figure(constrained_layout=True)
plt.title('Prediction Explanation: SAS at PHX\nFebruary 21, 2016',
          fontsize=16)
shap.plots.waterfall(shap_values[-1], max_display=10, show=False)
plt.show()

# Get game_id of game
df_info = pd.read_csv('../Data/Processed/base_file_for_model.csv')
df_pre.loc[860, 'home_id']

# Get information from game_id
games = pd.read_csv('../Data/Raw/games.csv')
games[games['id'] == 937]