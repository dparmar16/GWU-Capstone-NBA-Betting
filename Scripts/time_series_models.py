import pandas as pd
import os
import sys
from prophet import Prophet
from pmdarima.arima import auto_arima

toolbox_path = '../Utility_Functions'
sys.path.append(toolbox_path)
from functions import Cal_rolling_mean_var, ACF_PACF_Plot, ADF_Cal

# Load features dataframe
os.chdir('../Data')
df = pd.read_csv('Processed/gst_all_columns.csv')
df = df[['teamId', 'seasonYear', 'startDate', 'points']]

print(df.head(10))
print(df.columns)

# Get one team season
t1 = df[(df['seasonYear'] == 2015) & (df['teamId'] == 1)][['startDate', 'points']]
t1.rename(columns={'startDate':'ds', 'points':'y'}, inplace=True)

m = Prophet(yearly_seasonality=False,
            daily_seasonality=False,
            weekly_seasonality=True,
            changepoint_prior_scale=0.9)
m.fit(t1)

future = m.make_future_dataframe(periods=5)
print(future.tail())

forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

fig1 = m.plot(forecast)
fig1.show()

fig2 = m.plot_components(forecast)
fig2.show()

# Check for stationary
# Rolling mean and var
Cal_rolling_mean_var(t1['y'])
# ADF Test
ADF_Cal(t1['y'])
# Plot ACF and PACF
ACF_PACF_Plot(t1['y'], lags=15)

# Check for another team-season
t2 = df[(df['seasonYear'] == 2016) & (df['teamId'] == 2)][['startDate', 'points']]
t2.rename(columns={'startDate':'ds', 'points':'y'}, inplace=True)
Cal_rolling_mean_var(t2['y'])
ADF_Cal(t2['y'])
ACF_PACF_Plot(t2['y'], lags=15)

# Check for another team-season
t3 = df[(df['seasonYear'] == 2017) & (df['teamId'] == 4)][['startDate', 'points']]
t3.rename(columns={'startDate':'ds', 'points':'y'}, inplace=True)
Cal_rolling_mean_var(t3['y'])
ADF_Cal(t3['y'])
ACF_PACF_Plot(t3['y'], lags=15)




# Use ARMA library - Auto-ARIMA
# Use prophet library from FB
# Compare MSE of both approaches
# Fit the predicted points of each one
# Calculate f1 score of winner using points approach


# teamSeason_vals = df['teamSeason'].unique()
#
# for val in teamSeason_vals:
#     df_slim = df[df['teamSeason'] == val]
#     for i in range(0, df_slim.shape[0]):
#         if df_slim['season_games_played'] > 20:
#             arima = pm.auto_arima(df_slim['fga'][0:i], error_action='ignore', trace=True,
#                               suppress_warnings=True, maxiter=5,
#                               seasonal=False)
#         predict one
#         plot predicted vs actual

# for i in range(0, df.shape[0]):
#     game_date = df['startDate'][i]
#     team = df['teamSeason'][i]
#     if (df['season_games_played'][i] > 20) and (df['teamSeason'][i] == '2015-1'):
#         series = df[(df['teamSeason'] == team) & (df['startDate'] < game_date)]['team_efg']
#         arima = pm.auto_arima(series, error_action='ignore', trace=False,
#                           suppress_warnings=True, maxiter=5,
#                           seasonal=True, m=5)
#         #print(i, arima.predict(n_periods=1))
#         df['team_efg_pred'][i] = arima.predict(n_periods=1)