from sklearn.model_selection import cross_val_score
import scipy
from luciferml.supervised.utils.configs import *


def pred_check(predictor, type):
    if type == "regression":
        avlbl_predictors = list(regressors.keys())
    elif type == "classification":
        avlbl_predictors = list(classifiers.keys())
    if predictor in avlbl_predictors:
        return True
    else:
        return False


def sparse_check(features, labels):
    features = features
    labels = labels
    """
        Takes features and labels as input and checks if any of those is sparse csr_matrix.
        """
    try:
        print("Checking for Sparse Matrix [*]\n")
        if scipy.sparse.issparse(features[()]):
            print("Converting Sparse Features to array []\n")
            features = features[()].toarray()
            print("Conversion of Sparse Features to array Done [", u"\u2713", "]\n")

        elif scipy.sparse.issparse(labels[()]):
            print("Converting Sparse Labels to array []\n")
            labels = labels[()].toarray()
            print("Conversion of Sparse Labels to array Done [", u"\u2713", "]\n")

        else:
            print("No Sparse Matrix Found")

    except Exception as error:
        # print('Sparse matrix Check failed with KeyError: ', error)
        pass
    print("Checking for Sparse Matrix Done [", u"\u2713", "]\n")
    return (features, labels)


def kfold(model, predictor, X_train, y_train, cv_folds, isReg=False, all_mode=False):
    """
    Takes predictor, input_units, epochs, batch_size, X_train, y_train, cv_folds, and accuracy_scores dictionary.
    Performs K-Fold Cross validation and stores result in accuracy_scores dictionary and returns it.
    """
    if not isReg:
        name = classifiers
        scoring = "accuracy"
    if isReg:
        name = regressors
        scoring = "r2"
    try:
        if not all_mode:
            print("Applying K-Fold Cross Validation [*]")
        accuracies = cross_val_score(
            estimator=model, X=X_train, y=y_train, cv=cv_folds, scoring=scoring
        )
        if not all_mode:
            if not isReg:
                print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
            if isReg:
                print("R2 Score: {:.2f} %".format(accuracies.mean() * 100))
        model_name = name[predictor]
        accuracy = accuracies.mean() * 100
        if not all_mode:
            print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
            print("K-Fold Cross Validation [", u"\u2713", "]\n")
        return (model_name, accuracy)

    except Exception as error:
        print("K-Fold Cross Validation failed with error: ", error, "\n")
