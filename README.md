# GWU-Capstone-NBA-Betting

### Abstract

This project investigates modelling and forecasting NBA games through feature engineering and model selection. When various model types are tested, they perform extremely similarly and return highly correlated outputs, suggesting that feature choices are more important than model choices. The betting spreads are found to be the most effective feature, displaying the skill of oddsmakers and the wisdom of the betting crowds, and adding additional features beyond this is often not helpful. There is a ceiling of model performance when simply using box score statistics, and to improve there is a need to model both game-by-game player ability and opponent specific characteristics. 

### Context

This analysis has always mattered to super fans and fantasy sports players, but the value has accelerated as sports betting becomes more mainstream. Prior to 2018, sports betting in the United States was largely illegal but was still estimated as a $150 billion industry (Supreme Court Ruling Favors Sports Betting - The New York Times). A 2018 Supreme Court ruling paved the way for legalization of sports betting, and the industry is expected to grow a compounded 10 percent rate from 2021 to 2028 (Sports Betting Market Size & Share Report, 2021-2028). 

### Problem Statement

The goal of this project is to apply and test NBA prediction strategies using known feature sets (team Elo ratings, Four Factors team statistics, NBA 2K video game player ratings) and known models (Random Forest, SVM, Logistic Regression, Multi-Layer Perceptron NN) to find which strategy provides the best forecast of game results from the 2015 through 2020 NBA seasons. 

The models will be measured on f1 score, log loss (on their predicted probabilities), and Brier score (as this is a forecasting problem). Furthermore, the models and feature types will be compared on their effectiveness.

### Methodology

As the literature largely supports static models with game-specific information, we set out to codify this information into features. These features largely come from the literature or NBA experts.

Features include:
- Four Factors: The idea is that shooting the ball, taking care of the ball, offensive rebounding, and getting to the free throw line are activities that dictate the outcome of the game. These factors are Effective Field Goal Percentage, Offensive Rebounding Rate, Free Throw Rate, and Turnover Rate.
- Winning Trends: These are factors that directly measure the team’s history of winning and losing.
- Point Differential: This has been shown to be highly predictive, as consistently outscoring opponents matters more than winning close games (which can be due to luck).
- Rest: Teams with more rest (days since last game) tend to play better. “Back to backs” (games on two days in a row) are known to be extremely difficult to win.
- NBA 2K Ratings: Another approach is to find a value of quality of each player on the team, and turn that into a composite score for team strength. 
- Elo: This is a zero-sum rating system that was originally used for chess. Teams are given an initial rating, and the rating goes up or down based on game result and the quality of the opponent.
- Spread: Lastly, the spread (how many points one team is favored over another) is extremely relevant when it comes to game results. A team favored by 10 points will win most of the time, while a team favored by 1 point will win around half the time. We can use oddsmakers models as our own and gain from the wisdom of the crowds of bettors. For spreads, we use the “closing line” – the final value of the spread right before the game starts. 

Models tried include:
- Logistic Regression: This is a classical model, and it is mentioned often in the sports forecasting literature. This model is also interpretable, as we can see which features are significant and prominent. However, multi-collinearity is an issue here and thus necessitates dimensionality reduction.
- Logit: This model is associated with the logistic distribution, but it is different in that it is probabilistic. The parameters are estimated via maximum likelihood estimate (MLE), meaning how likely the observed data is given the parameters, over multiple iterations.
- Naïve Bayes: This classical model uses conditional probability and Bayes Rule to estimate parameters. 
- Multi-Layer Perceptron: Although the theory and use of MLPs goes back decades, neural networks have come to the mainstream more recently. Thus, we can use an MLP classifier along with our other models. 
- Auto-Regressive Moving Average (ARMA): Although time series models are not mentioned much in the literature, they are relevant as this is time ordered data. 

### Results

**F1 Score**

| Feature Set Name|  Logistic Regression|  Logit MLE | Naïve Bayes | MLP | ARMA |
| :---: | :---: | :---: | :---: | :---: | :---: |
| All Features or N/A | 0.733 | 0.733 | 0.713 | 0.720| 
| Spread Only | 0.737 | 0.737 | 0.737 | 0.746 |
| Spread Comprehensive| 0.737| 0.737| 0.729| 0.728 |
| Elo| 0.725| 0.725| 0.710| 0.721 |
| Spread Coefficient| 0.729| 0.729| 0.730| 0.742 |
| Spread Weighted Coefficient| 0.730| 0.730| 0.733| 0.717 |
| Four Factors| 0.723| 0.723| 0.717| 0.720 |
| Four Factors Moving Avg| 0.706| 0.706| 0.709| 0.710 |
| NBA 2k| 0.719| 0.719| 0.706| 0.710 |
| Point Differential and Rest| 0.721| 0.720| 0.691| 0.712 |
| No Features (ARMA only) | | | | | 0.604 |

**Brier Score**

| Feature Set Name|  Logistic Regression|  Logit MLE | Naïve Bayes | MLP |
| :---: | :---: | :---: | :---: | :---: |
| All Features or N/A | 0.212 | 0.212 | 0.275 | 0.213 | 
| Spread Only | 0.206 | 0.206 | 0.206 | 0.247 |
| Spread Comprehensive| 0.737| 0.737| 0.729| 0.728 |
| Elo| 0.213| 0.213| 0.228| 0.217| 
| Spread Coefficient| 0.211| 0.211| 0.211| 0.224| 
| Spread Weighted Coefficient| 0.210| 0.210| 0.210| 0.214| 
| Four Factors| 0.226| 0.226| 0.230| 0.231 |
| Four Factors Moving Avg| 0.232| 0.232| 0.235| 0.234 |
| NBA 2k| 0.223| 0.223| 0.232| 0.222 |
| Point Differential and Rest| 0.215| 0.215| 0.256| 0.218 |

### Conclusion 

One notable finding that the different static model types (logistic regression, logit, naïve bayes, and multi-layer perceptron) all come to approximately the same performance metrics. This suggests that feature engineering and pre-processing are just as important as model selection for sports forecasting. 

As the closing point spread captures a large amount of market information and wisdom of the crowd, it predicts the outcome of a game the best of all the features created. A model with just the closing line is better than a model with additional information about the teams themselves. This suggests that markets are highly efficient, and adding correlated features such as winning percentage, point differential, and shooting percentage can actually harm model performance through multi-collinearity and unnecessary complexity. 

Although this is time ordered data, doing an ARMA model on a target variable was not effective. There may be a way to do time series analysis on player performance fluctuations, but it’s possible that sports data has much more day-to-day variance and model complexity than daily temperatures or airline passengers (datasets commonly modelled with time series). 

Given the information captured with this dataset, there seems to be a ceiling of an F1-score of around 75% and floor on the Brier score loss of around 0.2. Ensembling models together helped slightly, but it doesn’t get the metrics past this ceiling. At this level, the models are not precise enough to use in betting, as missed classifications are very costly and there is an oddsmaker cut in priced into each bet. 

To take the next step in model performance, there is a need for data beyond simple box scores and game statistics. Examples of this would be a better understanding of team-specific matchups (how does Team A match up with Team B), player ability fluctuations within a season, and a method to identify how many minutes each player will play (and to put an outcome-specific weight to their effectiveness). It is likely that this advanced work is already happening, but it is proprietary, as one would not want to lose the financial edge in developing such insights. 
