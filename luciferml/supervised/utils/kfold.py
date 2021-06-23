from sklearn.model_selection import cross_val_score


def kfold(model, predictor,
          X_train, y_train, cv_folds,isReg
          ):
    """
    Takes predictor, input_units, epochs, batch_size, X_train, y_train, cv_folds, and accuracy_scores dictionary. 
    Performs K-Fold Cross validation and stores result in accuracy_scores dictionary and returns it.
    """
    if not isReg:
        name = {
            'lr' : 'Logistic Regression',
            'svm': 'Support Vector Machine',
            'knn': 'K-Nearest Neighbours',
            'dt' : 'Decision Trees',
            'nb' : 'Naive Bayes',
            'rfc': 'Random Forest Classifier',
            'xgb': 'XGBoost Classifier',
            'ann': 'Artificial Neural Network',
        }
        scoring = 'accuracy'
    if isReg:
        name = {
            'lin' : 'Linear Regression',
            'sgd' : 'Stochastic Gradient Descent Regressor',
            'krr' : 'Kernel Ridge Regressor',
            'elas': 'Elastic Net Regressot',
            'br'  : 'Bayesian Ridge Regressor',
            'svr' : 'Support Vector Regressor',
            'knr' : 'K-Neighbors Regressor',
            'dt'  : 'Decision Trees',
            'rfr' : 'Random Forest Regressor',
            'gbr' : 'Gradient Boost Regressor',
            'lgbm': 'LightGBM Regressor',
            'xgb' : 'XGBoost Regressor',
            'cat' : 'Catboost Regressor',
            'ann' : 'Artificial Neural Network',
        }
        scoring = 'r2'
    try:
        print('Applying K-Fold Cross Validation [*]')
        accuracies = cross_val_score(
            estimator=model, X=X_train, y=y_train, cv=cv_folds, scoring=scoring)
        if not isReg:
            print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
        if isReg:
            print("R2 Score: {:.2f} %".format(accuracies.mean()*100))
        model_name = name[predictor]
        accuracy = accuracies.mean()*100

        print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
        print('K-Fold Cross Validation [', u'\u2713', ']\n')
        return (model_name, accuracy)

    except Exception as error:
        print('K-Fold Cross Validation failed with error: ', error, '\n')
