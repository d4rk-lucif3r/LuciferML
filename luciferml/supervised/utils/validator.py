from sklearn.model_selection import cross_val_score
import scipy
from luciferml.supervised.utils.configs import *
from colorama import Fore


def pred_check(predictor, pred_type):
    if pred_type == "regression":
        avlbl_predictors = list(regressors_ver.keys())
    elif pred_type == "classification":
        avlbl_predictors = list(classifiers_ver.keys())
    if type(predictor) == str:
        if predictor in avlbl_predictors:
            return True, predictor
        else:
            return False, predictor
    elif type(predictor) == list:
        for i in predictor:
            if i not in avlbl_predictors:
                return False, i
        return True, None


def sparse_check(features, labels):
    features = features
    labels = labels
    """
        Takes features and labels as input and checks if any of those is sparse csr_matrix.
        """
    try:
        if scipy.sparse.issparse(features[()]):
            features = features[()].toarray()
        elif scipy.sparse.issparse(labels[()]):
            labels = labels[()].toarray()
    except Exception as error:
        pass
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
        accuracies = cross_val_score(
            estimator=model, X=X_train, y=y_train, cv=cv_folds, scoring=scoring
        )
        if not all_mode:
            if not isReg:
                print("        KFold Accuracy: {:.2f} %".format(accuracies.mean() * 100))
            if isReg:
                print("        R2 Score: {:.2f} %".format(accuracies.mean() * 100))
        model_name = name[predictor]
        accuracy = accuracies.mean() * 100
        if not all_mode:
            print(
                "        Standard Deviation: {:.2f} %".format(accuracies.std() * 100),
                "\n",
            )
        return (model_name, accuracy)

    except Exception as error:
        print(Fore.RED + "K-Fold Cross Validation failed with error: ", error, "\n")
