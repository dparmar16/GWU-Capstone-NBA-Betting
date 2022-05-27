# Explanation of Code Files

As a pre-cursor, we had to obtain the data. the data consists of game data, odds data, and other metrics. The data was loaded into AWS s3 and downloaded via csv files to place into the /Data folder in this repository.
- The game data consists of game information, team box scores, player box scores, and player characteristics. This data comes from [Rapid API NBA-API](https://rapidapi.com/api-sports/api/api-nba).
- The odds data comes from [Sportsbook Reviews Online](https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm).
- Elo data comes from [FiveThirtyEight](https://github.com/fivethirtyeight/data/tree/master/nba-elo) and is updated with each game.
- NBA 2K player ratings come from the [NBA 2K video game](https://hoopshype.com/nba2k/). 

Next, we can move on to our scripts. All scripts are in the /Scripts directory except for utility functions, which are stored separately.

First, the **_pre-processing.py_** script runs and takes raw data (found in /Data/Raw) and saves processed files for our exploration and modelling (saved in /Data/Processed).

Second, run all the different individual model files to see how they are developed and scoped. This includes the **_logistic_regression.py_**, **_mle_logit.py_**, **_multi_layer_perceptron.py_**, and **_time_series_models.py_** files.
- NOTE: The time series file saves an output that is needed later.
- NOTE: The MLP script will take somewhat significant time (10+ minutes) unless there is a GPU or cloud computing involved.

Third, run **_model_factor_comparison.py_** which will use utility functions and output a comparison of all the model/feature combinations.
- NOTE: Again, the MLP models will take significant time unless there is a GPU or cloud computing involved.

Fourth, run **_ensemble_model.py_** to get ensemble information and save an output file. This will be used both in analysis and to test the profitability of the models.

Lastly, run **_betting_model_deployment.py_** which will show that this betting system loses money under various betting assumptions. Although this was not the intended result, it showed how efficient the sports betting markets are. 

You may notice there are two more files (**_exploratory_data_analysis.py_** and **_team_strength_kalman_filter.py_**). These were used for conceptual prototyping but ultimately don't have meaningful output.