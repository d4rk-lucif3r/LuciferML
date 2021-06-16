# LuciferML a Semi-Automated Machine Learning Library by d4rk-lucif3r

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

     1) All Columns

         from luciferml.preprocessing import Preprocess as prep

         import pandas as pd

         dataset = pd.read_csv('/examples/Social_Network_Ads.csv')

         dataset = prep.skewcorrect(dataset)

     2) Except column/columns

         from luciferml.preprocessing import Preprocess as prep

         import pandas as pd

         dataset = pd.read_csv('/examples/Social_Network_Ads.csv')

         dataset = prep.skewcorrect(dataset,except_columns=['Purchased'])

    More about Preprocessing [here](https://github.com/d4rk-lucif3r/LuciferML/blob/master/LuciferML/Preprocessing.md)

## Available Modelling Techniques

1) Classification

    Available Models for Classification

        - 'lr' : 'Logistic Regression',
        - 'svm': 'Support Vector Machine',
        - 'knn': 'K-Nearest Neighbours',
        - 'dt' : 'Decision Trees',
        - 'nb' : 'Naive Bayes',
        - 'rfc': 'Random Forest Classifier',
        - 'xgb': 'XGBoost Classifier',
        - 'ann': 'Artificical Neural Network',

    Example:

        from luciferml.supervised import classification as cls
        dataset = pd.read_csv('Social_Network_Ads.csv')
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        cls.Classification(predictor = 'lr').predict(X, y)

    More About [Classification](https://github.com/d4rk-lucif3r/LuciferML/blob/master/LuciferML/Classification.md)

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
        - 'lgbm': 'LightGB Regressor',
        - 'xgb' : 'XGBoost Regressor',
        - 'cat' : 'Catboost Regressor',
        - 'ann' : 'Artificical Neural Network',

    Example:

        from luciferml.supervised import regression as reg
        dataset = pd.read_excel('examples\Folds5x2_pp.xlsx')
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        reg.Regression(predictor = 'lin').predict(X, y)

    More about Preprocessing [here](https://github.com/d4rk-lucif3r/LuciferML/blob/master/LuciferML/Regression.md)

## More To be Added Soon
