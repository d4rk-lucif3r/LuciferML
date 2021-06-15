# LuciferML a Semi-Automated Machine Learning Library by d4rk-lucif3r

## About

The LuciferML is a Semi-Automated Machine Learning Python Library that works with tabular data. It is designed to save time while doing data analysis. It will help you right from data preprocessing to Data Prediction.

### The LuciferML will help you with:

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

## Available Modelling Techniques: 

1) Classification 
    
    Available Predictors for Classification
    
        - lr - Logisitic Regression
        - svm - SupportVector Machine
        - knn - K-Nearest Neighbours
        - dt - Decision Trees
        - nb - GaussianNaive bayes
        - rfc- Random Forest Classifier
        - xgb- XGBoost Classifier
        - ann - Artificial Neural Network

    Example:
    
        from luciferml.supervised import classification as cls
        dataset = pd.read_csv('Social_Network_Ads.csv')
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        cls.Classification(predictor = 'lr').predict(X, y)

    More About [Classification](https://github.com/d4rk-lucif3r/LuciferML/blob/master/LuciferML/Classification.md)

    
## Note - LuciferML rightnow supports only Classification.
## More To be Added Soon
