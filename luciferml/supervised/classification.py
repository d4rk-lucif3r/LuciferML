

import time
from typing import Tuple


from sklearn.metrics import accuracy_score

import pandas as pd

from luciferml.supervised.utils.encoder import encoder
from luciferml.supervised.utils.predPreprocess import pred_preprocess
from luciferml.supervised.utils.dimensionalityReduction import dimensionalityReduction
from luciferml.supervised.utils.classificationPredictor import classificationPredictor
from luciferml.supervised.utils.confusionMatrix import confusionMatrix
from luciferml.supervised.utils.kfold import kfold
from luciferml.supervised.utils.hyperTune import hyperTune
from luciferml.supervised.utils.sparseCheck import sparseCheck


class Classification:

    def __init__(self,
                 predictor='lr',
                 params={},
                 tune=False,
                 test_size=.2,
                 cv_folds=10,
                 random_state=42,
                 pca_kernel='linear',
                 n_components_lda=1,
                 lda='n', pca='n',
                 n_components_pca=2,
                 hidden_layers=4,
                 output_units=1,
                 input_units=6,
                 input_activation='relu',
                 output_activation='sigmoid',
                 optimizer='adam',
                 metrics=['accuracy',],
                 loss='binary_crossentropy',
                 validation_split=.20,
                 epochs=100,
                 batch_size=32,
                 tune_mode=1,
                 smote= 'n',
                 k_neighbors=1
                 ):
        """
        Encode Categorical Data then Applies SMOTE , Splits the features and labels in training and validation sets with test_size = .2 , scales self.X_train, X_val using StandardScaler.
        Fits every model on training set and predicts results find and plots Confusion Matrix,
        finds accuracy of model applies K-Fold Cross Validation
        and stores accuracy in variable name accuracy and model name in self.classifier name and returns both as a tuple.
        Applies GridSearch Cross Validation and gives best self.params out from param list.

        self.Parameters:
            features : array
                        features array

            lables : array
                        labels array

            predictor : str
                        Predicting model to be used
                        Default 'lr'
                            Predictor Strings:
                                lr - Logisitic Regression

                                svm -SupportVector Machine

                                knn - K-Nearest Neighbours

                                dt - Decision Trees

                                nb - GaussianNaive bayes

                                rfc- Random Forest self.Classifier

                                xgb- XGBoost self.Classifier

                                ann - Artificial Neural Network
            params : dict
                        contains parameters for model
            tune : boolean
                    when True Applies GridSearch CrossValidation
                    Default is False

            test_size: float or int, default=.2
                        If float, should be between 0.0 and 1.0 and represent
                        the proportion of the dataset to include in
                        the test split.
                        If int, represents the absolute number of test samples.

            cv_folds : int
                    No. of cross validation folds. Default = 10
            pca : str
                if 'y' will apply PCA on Train and Validation set. Default = 'n'
            lda : str
                if 'y' will apply LDA on Train and Validation set. Default = 'n'
            pca_kernel : str
                    Kernel to be use in PCA. Default = 'linear'
            n_components_lda : int
                    No. of components for LDA. Default = 1
            n_components_pca : int
                    No. of components for PCA. Default = 2
            self.hidden_layers : int
                    No. of default layers of ann. Default = 4
            inputs_units : int
                    No. of units in input layer. Default = 6
            output_units : int
                    No. of units in output layer. Default = 6
            self.input_activation : str
                    Activation function for Hidden layers. Default = 'relu'
            self.output_activation : str
                    Activation function for Output layers. Default = 'sigmoid'
            optimizer: str
                    Optimizer for ann. Default = 'adam'
            loss : str
                    loss method for ann. Default = 'binary_crossentropy'
            validation_split : float or int
                    Percentage of validation set splitting in ann. Default = .20
            epochs : int
                    No. of epochs for ann. Default = 100
            batch_size :
                    Batch Size for ANN. Default = 32
            tune_mode : int
                    HyperParam tune modes. Default = 1
                        Available Modes:
                            1 : Basic Tune
                            2 : Intermediate Tune
                            3 : Extreme Tune (Can Take Much Time)
            smote : str,
                Whether to apply SMOTE. Default = 'y'
            k_neighbors : int
                No. of neighbours for SMOTE. Default = 1

        Example:

            from luciferml.supervised import classification as cls

            dataset = pd.read_csv('Social_Network_Ads.csv')

            X = dataset.iloc[:, :-1]

            y = dataset.iloc[:, -1]
            
            cls.Classification(predictor = 'lr').predict(X, y)

        """

        self.predictor = predictor
        self.params = params
        self.tune = tune
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.pca_kernel = pca_kernel
        self.n_components_lda = n_components_lda
        self.lda = lda
        self.pca = pca
        self.n_components_pca = n_components_pca
        self.hidden_layers = hidden_layers
        self.output_units = output_units
        self.input_units = input_units
        self.input_activation = input_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = loss
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.tune_mode = tune_mode
        self.rerun = False
        self.smote = smote
        self.k_neighbors = k_neighbors

        self.accuracy_scores = {}

    def predict(self, features, labels) -> Tuple:
        self.features = features
        self.labels = labels

        # Time Function ---------------------------------------------------------------------

        start = time.time()
        print("Started Predictor \n")
        if not self.rerun:
            # CHECKUP ---------------------------------------------------------------------
            if not isinstance(self.features, pd.DataFrame) and not isinstance(self.labels, pd.Series):
                print('TypeError: This Function take features as Pandas Dataframe and labels as Pandas Series. Please check your implementation.\n')
                end = time.time()
                print(end - start)
                return

            # Encoding ---------------------------------------------------------------------

            self.features, self.labels = encoder(self.features, self.labels)

            # Sparse Check -------------------------------------------------------------
            self.features, self.labels = sparseCheck(self.features, self.labels)

            # Preprocessing ---------------------------------------------------------------------
            self.X_train, X_val, self.y_train, y_val = pred_preprocess(
                self.features, self.labels, self.test_size, self.random_state, self.smote, self.k_neighbors)

            # Dimensionality Reduction---------------------------------------------------------------------
            self.X_train, X_val = dimensionalityReduction(
                self.lda, self.pca, self.X_train, X_val, self.y_train,
                self.n_components_lda, self.n_components_pca, self.pca_kernel, start)

        # Models ---------------------------------------------------------------------
        if self.predictor == 'ann':
            self.parameters, self.classifier, self.classifier_wrap = classificationPredictor(
                self.predictor, self.params, self.X_train, X_val, self.y_train, y_val, self.epochs, self.hidden_layers,
                self.input_activation, self.output_activation, self.loss,
                self.batch_size, self.metrics, self.validation_split, self.optimizer, self.output_units, self.input_units, self.tune_mode
            )
        else:
            self.parameters, self.classifier = classificationPredictor(
                self.predictor, self.params, self.X_train, X_val, self.y_train, y_val, self.epochs, self.hidden_layers,
                self.input_activation, self.output_activation, self.loss,
                self.batch_size, self.metrics, self.validation_split, self.optimizer, self.output_units, self.input_units, self.tune_mode
            )

        try:

            if not self.predictor == 'ann':
                self.classifier.fit(self.X_train, self.y_train)
        except Exception as error:
            print('Model Train Failed with error: ', error, '\n')

        print('Model Training Done [', u'\u2713', ']\n')
        print('Predicting Data [*]\n')
        try:
            y_pred = self.classifier.predict(X_val)
            print('Data Prediction Done [', u'\u2713', ']\n')
        except Exception as error:
            print('Prediction Failed with error: ', error,  '\n')

        # Confusion Matrix --------------------------------------------------------------
        if self.predictor == 'ann':
            y_pred = (y_pred > 0.5).astype("int32")
        confusionMatrix(y_pred, y_val)

        # Accuracy ---------------------------------------------------------------------
        print('''Evaluating Model Performance [*]''')
        try:
            accuracy = accuracy_score(y_val, y_pred)
            print('Validation Accuracy is :', accuracy)
            print('Evaluating Model Performance [', u'\u2713', ']\n')
        except Exception as error:
            print('Model Evaluation Failed with error: ', error, '\n')

        # K-Fold ---------------------------------------------------------------------
        if self.predictor == 'ann':
            self.classifier_name, accuracy = kfold(
                self.classifier_wrap,
                self.predictor, self.X_train, self.y_train, self.cv_folds


            )
        else:
            self.classifier_name, accuracy = kfold(
                self.classifier,
                self.predictor,
                self.X_train, self.y_train, self.cv_folds
            )

        # GridSearch ---------------------------------------------------------------------
        if not self.predictor == 'nb' and self.tune:
            self.__tuner()

        print('Complete [', u'\u2713', ']\n')
        end = time.time()
        print('Time Elapsed : ', end - start, 'seconds \n')
        return (self.classifier_name, accuracy)

    def __tuner(self):
        if self.predictor == 'ann':
            self.best_params = hyperTune(
                self.classifier_wrap, self.parameters, self.X_train, self.y_train, self.cv_folds, self.tune_mode)
        else:
            self.best_params = hyperTune(
            self.classifier, self.parameters, self.X_train, self.y_train, self.cv_folds, self.tune_mode)
        # if self.tune_mode == 3:
        #     self.params = self.best_params
        #     self.tune = False
        #     self.rerun = True
        #     self.predict(self.features, self.labels)
        #     print("Re-ran Predictor on these params :", self.params)
        #     print(
        #         'Re-running Classifier with Best Params Done[', u'\u2713', ']\n')
