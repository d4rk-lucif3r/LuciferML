from luciferml.supervised.utils.configs import *
from sklearn.model_selection import cross_val_score


def kfold(model, predictor,
          X_train, y_train, cv_folds, isReg=False, all_mode=False
          ):
    """
    Takes predictor, input_units, epochs, batch_size, X_train, y_train, cv_folds, and accuracy_scores dictionary. 
    Performs K-Fold Cross validation and stores result in accuracy_scores dictionary and returns it.
    """
    if not isReg:
        name = classifiers
        scoring = 'accuracy'
    if isReg:
        name = regressors
        scoring = 'r2'
    try:
        if not all_mode:
            print('Applying K-Fold Cross Validation [*]')
        accuracies = cross_val_score(
            estimator=model, X=X_train, y=y_train, cv=cv_folds, scoring=scoring)
        if not all_mode:
            if not isReg:
                print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
            if isReg:
                print("R2 Score: {:.2f} %".format(accuracies.mean()*100))
        model_name = name[predictor]
        accuracy = accuracies.mean()*100
        if not all_mode:
            print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
            print('K-Fold Cross Validation [', u'\u2713', ']\n')
        return (model_name, accuracy)

    except Exception as error:
        print('K-Fold Cross Validation failed with error: ', error, '\n')
