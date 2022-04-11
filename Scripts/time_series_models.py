import pandas as pd
import numpy as np
import os
import sys
from prophet import Prophet
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose

toolbox_path = '../Utility_Functions'
sys.path.append(toolbox_path)
from functions import Cal_rolling_mean_var, ACF_PACF_Plot, ADF_Cal

# Load features dataframe
os.chdir('../Data')
all_cols = pd.read_csv('Processed/gst_all_columns.csv')
df = all_cols[['id', 'teamId', 'seasonYear', 'startDate', 'points']]
df.sort_values(by=['startDate', 'teamId'], inplace=True)
df['season_games_played'] = df.groupby(['teamId', 'seasonYear']).cumcount() + 1
df['teamSeason'] = df['seasonYear'].astype(str) + '-' + df['teamId'].astype(str)

# print(df.head(10))
# print(df.columns)

# Get one team season
t1 = df[(df['seasonYear'] == 2015) & (df['teamId'] == 1)][['startDate', 'points']]
t1.rename(columns={'startDate':'ds', 'points':'y'}, inplace=True)
# print(t1)

# m = Prophet(yearly_seasonality=False,
#             daily_seasonality=False,
#             weekly_seasonality=True,
#             changepoint_prior_scale=0.9)
# m.fit(t1)
#
# future = m.make_future_dataframe(periods=5)
# # print(future.tail())
#
# forecast = m.predict(future)
# # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
#
# fig1 = m.plot(forecast)
# fig1.show()
#
# fig2 = m.plot_components(forecast)
# fig2.show()

# Yes this is stationary
# # Check for stationary
# # Rolling mean and var
# Cal_rolling_mean_var(t1['y'])
# # ADF Test
# ADF_Cal(t1['y'])
# # Plot ACF and PACF
# ACF_PACF_Plot(t1['y'], lags=15)

# Look for seasonality and trend in first team-season
t1_decompose = t1.copy()
t1_decompose.set_index('ds', inplace=True)
#t1_decompose.drop(columns=['ds'], axis=1)

decompose_result_mult = seasonal_decompose(t1_decompose, model="multiplicative", period=41)
#decompose_result_mult = seasonal_decompose(t1_decompose.asfreq('d'), model="multiplicative")

trend = decompose_result_mult.trend
seasonal = decompose_result_mult.seasonal
residual = decompose_result_mult.resid

# decompose_plot = decompose_result_mult.plot()
# decompose_plot.show()

decompose_result_mult = seasonal_decompose(t1_decompose, model="additive", period=41)
#decompose_result_mult = seasonal_decompose(t1_decompose.asfreq('d'), model="multiplicative")

trend = decompose_result_mult.trend
seasonal = decompose_result_mult.seasonal
residual = decompose_result_mult.resid

# decompose_plot = decompose_result_mult.plot()
# decompose_plot.show()

# All of these are stationary
# # Check for another team-season
# t2 = df[(df['seasonYear'] == 2016) & (df['teamId'] == 2)][['startDate', 'points']]
# t2.rename(columns={'startDate':'ds', 'points':'y'}, inplace=True)
# Cal_rolling_mean_var(t2['y'])
# ADF_Cal(t2['y'])
# ACF_PACF_Plot(t2['y'], lags=15)
#
# # Check for another team-season
# t3 = df[(df['seasonYear'] == 2017) & (df['teamId'] == 4)][['startDate', 'points']]
# t3.rename(columns={'startDate':'ds', 'points':'y'}, inplace=True)
# Cal_rolling_mean_var(t3['y'])
# ADF_Cal(t3['y'])
# ACF_PACF_Plot(t3['y'], lags=15)




# Use ARMA library - Auto-ARIMA
# Use prophet library from FB
# Compare MSE of both approaches
# Fit the predicted points of each one
# Calculate f1 score of winner using points approach
rows_count = df.shape[0]
df['points_arma'] = np.nan

for i in range(0, rows_count):
    print(f'Working on row {i} of {rows_count}')
    game_date = df['startDate'][i]
    team = df['teamSeason'][i]
    if (df['season_games_played'][i] >= 6): # and (df['teamSeason'][i] == '2015-1'):
        series_pre = df[(df['teamSeason'] == team) & (df['startDate'] < game_date)]
        series_arima = series_pre['points']
        arima = auto_arima(series_arima, error_action='ignore', trace=False,
                          suppress_warnings=True, maxiter=10,
                          # can add p and q if desired
                              seasonal=True)

        # series_prophet = series_pre.rename(columns={'startDate':'ds', 'points':'y'})[['ds', 'y']]
        # prophet_mod = Prophet(yearly_seasonality=False,
        #                       daily_seasonality=False,
        #                       weekly_seasonality=True,
        #                       changepoint_prior_scale=0.5)
        # prophet_mod.fit(series_prophet)
        #
        # future = prophet_mod.make_future_dataframe(periods=1)
        # print(future)
        # forecast = prophet_mod.predict(future)
        # print('forecast is')
        # print(forecast)


        #df['points_prophet'], df['points_prophet_lower'], df['points_prophet_upper'] = np.NaN, np.NaN, np.NaN


        df['points_arma'][i] = arima.predict(n_periods=1)
        #print(df.iloc[i, :])
        #print(df['points_arma'][i])
        # df['points_prophet'][i] = forecast['yhat']
        # df['points_prophet_lower'][i] = forecast['yhat_lower']
        # df['points_prophet_upper'][i] = forecast['yhat_upper']

        # print('arma prediction is')
        # print(arima.predict(n_periods=1))


print(df['points_arma'].value_counts())
df.to_csv('Processed/points_arma_predictions.csv', index_label=False)