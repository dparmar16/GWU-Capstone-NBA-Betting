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

### Results

### Repository Explanation

