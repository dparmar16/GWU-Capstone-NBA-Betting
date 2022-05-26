# Explanation of Plots 

- acf_pacf_team1_2015_ex.jpeg
  - This plot looks at the Auto-Correlation Function (ACF) and Partial Auto-Correlation Function (PACF) for points scored by one team over one season (in this example the 2015 Phoenix Suns). The ACF and PACF beyond a lag of 1 are not significant, indicating this series behaves like white noise. This indicates that past points scored are not the best method to predict future points scored. 
- corr_matrix_all_features.jpeg
  - This is a correlation matrix of all features generated in my pre-processing and feature engineering. We find that many of the features show significant correlation, creating multi-collinearity risk in certain models.
- corr_matrix_reduced_features.jpeg
  - This is a correlation matrix of the reduced feature set, meaning dropping certain features that were highly correlated with other features. We still see some correlation, but much less than before.
- model_agreement_logistic_mlp.jpeg
  - This plot shows the predicted probability of a home team victory from both a logistic regression model and multi-layer perceptron (MLP) model. We see high agreement between the two, with a correlation of over 0.9. 
- pca_all_features.jpeg
  - This plot shows the principal component analysis (PCA) of all features created for the predictive models. PCA shows how much of the variation can be explained by each feature. We see that around half the features explain over 90 percent of the variance. This means we can do dimensionality reduction (reduce or combine features) and still get decent model performance.
- pca_reduced_features.jpeg
  - This plot shows the PCA of the reduced feature set (removing highly correlated features). We see that each feature is more useful, and there is less of a tail-off at the end.
- rolling_mean_var_team1_2015_ex.jpeg
  - This plot looks at the rolling mean and variance of points scored by one team over one season (in this example the 2015 Phoenix Suns). We use this plot to show the data is stationary, and therefore we can fit an Auto-Regressive Moving Average (ARMA) model to this data.
- shap_reduced_feature_set.jpeg
  - [SHAP](https://github.com/slundberg/shap) stands for SHapley Additive exPlanations, and this library can be used to explain how a model makes predictions on individual observations given feature values. We can then aggregate the individual decisions to show how much each feature contributes to the model's decisions. This turns into feature importance, which is plotted here for the logistic regression model which predicts whether the home team will win the given game.