intro = """
          
██╗░░░░░██╗░░░██╗░█████╗░██╗███████╗███████╗██████╗░░░░░░░███╗░░░███╗██╗░░░░░
██║░░░░░██║░░░██║██╔══██╗██║██╔════╝██╔════╝██╔══██╗░░░░░░████╗░████║██║░░░░░
██║░░░░░██║░░░██║██║░░╚═╝██║█████╗░░█████╗░░██████╔╝█████╗██╔████╔██║██║░░░░░
██║░░░░░██║░░░██║██║░░██╗██║██╔══╝░░██╔══╝░░██╔══██╗╚════╝██║╚██╔╝██║██║░░░░░
███████╗╚██████╔╝╚█████╔╝██║██║░░░░░███████╗██║░░██║░░░░░░██║░╚═╝░██║███████╗
╚══════╝░╚═════╝░░╚════╝░╚═╝╚═╝░░░░░╚══════╝╚═╝░░╚═╝░░░░░░╚═╝░░░░░╚═╝╚══════╝
"""

classifiers = {
    "lr": "Logistic Regression",
    "sgd": "Stochastic Gradient Descent",
    "perc": "Perceptron",
    "pass": "Passive Aggressive Classifier",
    "ridg": "Ridge Classifier",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbours",
    "dt": "Decision Trees",
    "nb": "Naive Bayes",
    "rfc": "Random Forest Classifier",
    "gbc": "Gradient Boosting Classifier",
    "ada": "AdaBoost Classifier",
    "bag": "Bagging Classifier",
    "extc": "Extra Trees Classifier",
    "lgbm": "LightGBM Classifier",
    "cat": "CatBoost Classifier",
    "xgb": "XGBoost Classifier",
    "ann": "Artificial Neural Network",
}

regressors = {
    "lin": "Linear Regression",
    "sgd": "Stochastic Gradient Descent Regressor",
    "krr": "Kernel Ridge Regressor",
    "elas": "Elastic Net Regressor",
    "br": "Bayesian Ridge Regressor",
    "svr": "Support Vector Regressor",
    "knr": "K-Neighbors Regressor",
    "dt": "Decision Trees Regressor",
    "rfr": "Random Forest Regressor",
    "gbr": "Gradient Boost Regressor",
    "ada": "AdaBoost Regressor",
    "bag": "Bagging Regressor",
    "extr": "Extra Trees Regressor",
    "lgbm": "LightGBM Regressor",
    "xgb": "XGBoost Regressor",
    "cat": "Catboost Regressor",
    "ann": "Multi-Layer Perceptron Regressor",
}

params_use_warning = (
    "Params will not work with predictor = 'all'. Settings params = {} "
)

unsupported_pred_warning = """Predictor not available. Please use the predictor which is supported by LuciferML.
Please check the documentation for more details.\n
"""
