<p align="center" ><img align='center' alt="https://github.com/d4rk-lucif3r/LuciferML/blob/master/assets/img/LUCIFER-ML.gif"  src="https://github.com/d4rk-lucif3r/LuciferML/blob/master/assets/img/LUCIFER-ML.gif"></p>

# LuciferML a Semi-Automated Machine Learning Library by d4rk-lucif3r

[![Downloads](https://static.pepy.tech/personalized-badge/lucifer-ml?period=month&units=international_system&left_color=black&right_color=orange&left_text=Downloads%20per%20Month)](https://pepy.tech/project/lucifer-ml)
![ReadTheDocs](https://img.shields.io/readthedocs/luciferml?style=plastic)

## About

The LuciferML is a Semi-Automated Machine Learning Python Library that works with tabular data. It is designed to save time while doing data analysis. It will help you right from data preprocessing to Data Prediction.

### The LuciferML will help you with

1. Preprocessing Data:
    - Encoding
    - Splitting
    - Scaling
    - Dimensionality Reduction
    - Resampling
2. Trying many different machine learning models with hyperparameter tuning,

## Installation

    pip install lucifer-ml

## Available Preprocessing Techniques

1) Skewness Correction

    Takes Pandas Dataframe as input. Transforms each column in dataset except the columns given as an optional parameter.
    Returns Transformed Data.

    Example:

     1) All Columns:

             from luciferml.preprocessing import Preprocess as prep
             import pandas as pd
             dataset = pd.read_csv('/examples/Social_Network_Ads.csv')
             dataset = prep.skewcorrect(dataset)

     2) Except column/columns:

             from luciferml.preprocessing import Preprocess as prep
             import pandas as pd
             dataset = pd.read_csv('/examples/Social_Network_Ads.csv')
             dataset = prep.skewcorrect(dataset,except_columns=['Purchased'])

    More about Preprocessing [here](https://github.com/d4rk-lucif3r/LuciferML/blob/master/luciferml/supervised/README/Preprocessing.md)

## Available Modelling Techniques

1) Classification

    Available Models for Classification

        - 'lr'  : 'Logistic Regression',
        - 'sgd' : 'Stochastic Gradient Descent',
        - 'perc': 'Perceptron',
        - 'pass': 'Passive Aggressive Classifier',
        - 'ridg': 'Ridge Classifier', 
        - 'svm' : 'Support Vector Machine',
        - 'knn' : 'K-Nearest Neighbours',
        - 'dt'  : 'Decision Trees',
        - 'nb'  : 'Naive Bayes',
        - 'rfc' : 'Random Forest Classifier',
        - 'gbc' : 'Gradient Boosting Classifier',
        - 'ada' : 'AdaBoost Classifier',
        - 'bag' : 'Bagging Classifier',
        - 'extc': 'Extra Trees Classifier',
        - 'lgbm': 'LightGBM Classifier',
        - 'cat' : 'CatBoost Classifier',
        - 'xgb' : 'XGBoost Classifier',
        - 'ann' : 'Artificial Neural Network',
        - 'all' : 'Applies all above classifiers'

    Example:

        from luciferml.supervised.classification import Classification
        dataset = pd.read_csv('Social_Network_Ads.csv')
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        classifier = Classification(predictor = 'lr')
        classifier.fit(X, y)
        result = classifier.result()

    More About [Classification](https://github.com/d4rk-lucif3r/LuciferML/blob/master/luciferml/supervised/README/Classification.md)

2) Regression

       Available Models for Regression

        - 'lin' : 'Linear Regression',
        - 'sgd' : 'Stochastic Gradient Descent Regressor',
        - 'elas': 'Elastic Net Regressot',
        - 'krr' : 'Kernel Ridge Regressor',
        - 'br'  : 'Bayesian Ridge Regressor',
        - 'svr' : 'Support Vector Regressor',
        - 'knr' : 'K-Nearest Regressor',
        - 'dt'  : 'Decision Trees',
        - 'rfr' : 'Random Forest Regressor',
        - 'gbr' : 'Gradient Boost Regressor',
        - 'ada' : 'AdaBoost Regressor',
        - 'bag' : 'Bagging Regressor',
        - 'extr': 'Extra Trees Regressor',
        - 'lgbm': 'LightGBM Regressor',
        - 'xgb' : 'XGBoost Regressor',
        - 'cat' : 'Catboost Regressor',
        - 'ann' : 'Artificial Neural Network',
        - 'all' : 'Applies all above regressors'

    Example:

        from luciferml.supervised.regression import Regression
        dataset = pd.read_excel('examples\Folds5x2_pp.xlsx')
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        regressor = Regression(predictor = 'lin')
        regressor.fit(X, y)
        result = regressor.result()

    More about Regression [here](https://github.com/d4rk-lucif3r/LuciferML/blob/master/luciferml/supervised/README/Regression.md)

---

## [To-Do's](https://github.com/d4rk-lucif3r/LuciferML/issues/10)
