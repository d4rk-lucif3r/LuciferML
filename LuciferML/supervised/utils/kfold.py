from distutils.log import error
import tensorflow as tf
from sklearn.model_selection import cross_val_score
from luciferml.supervised.utils.classificationPredictor import build_ann_model


def kfold(classifier, predictor,
          input_units, epochs, batch_size,
          X_train, y_train, cv_folds, accuracy_scores,
          hidden_layers,
          input_activation, output_activation,
          output_units, optimizer, loss, metrics
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
        if predictor == 'ann':
            classifier = tf.keras.wrappers.scikit_learn.KerasClassifier(
                build_fn=build_ann_model, verbose=1, input_units=input_units,
                epochs=epochs, batch_size=batch_size,
                hidden_layers=hidden_layers,
                input_activation=input_activation, output_activation=output_activation,
                output_units=output_units, optimizer=optimizer, loss=loss, metrics=metrics)
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
