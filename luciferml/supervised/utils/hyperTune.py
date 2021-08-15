from sklearn.model_selection import GridSearchCV
def hyperTune(classifier, parameters, X_train, y_train, cv_folds, tune_mode, isReg=False, all_mode=False):
    """
    Takes classifier, tune-parameters, Training Data and no. of folds as input and Performs GridSearch Crossvalidation.
    """
    try:
        scoring = 'accuracy'
        if isReg:
            scoring = 'r2'
        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=parameters,
            scoring=scoring,
            cv=cv_folds,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        best_accuracy = grid_search.best_score_
        best_parameters = grid_search.best_params_
        if not all_mode:
            print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
            print("Best Parameters:", best_parameters)
            print('Applying Grid Search Cross validation [', u'\u2713', ']\n')
        return best_parameters, best_accuracy
    except Exception as error:
        print('HyperParam Tuning Failed with Error: ', error, '\n')
