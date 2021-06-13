from distutils.log import error
import tensorflow as tf
from sklearn.model_selection import cross_val_score


def kfold(classifier, predictor,
          X_train, y_train, cv_folds,
          ):
    """
    Takes predictor, input_units, epochs, batch_size, X_train, y_train, cv_folds, and accuracy_scores dictionary. 
    Performs K-Fold Cross validation and stores result in accuracy_scores dictionary and returns it.
    """
    name = {
        'lr': 'Logistic Regression',
        'svm': 'Support Vector Machine',
        'knn': 'K-Nearesr Neighbours',
        'dt': 'Decision Trees',
        'nb': 'Naive Bayes',
        'rfc': 'Random Forest CLassifier',
        'xgb': 'XGBoost Classifier',
        'ann': 'Artificical Neural Network',
    }
    try:
        print('Applying K-Fold Cross validation [*]')
        accuracies = cross_val_score(
            estimator=classifier, X=X_train, y=y_train, cv=cv_folds, scoring='accuracy')
        print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

        classifier_name = name[predictor]
        accuracy = accuracies.mean()*100

        print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
        print('K-Fold Cross validation [', u'\u2713', ']\n')
        return (classifier_name, accuracy)

    except Exception as error:
        print('K-Fold Cross Validation failed with error: ', error, '\n')
