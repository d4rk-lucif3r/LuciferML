import copy
from math import e
import os
import time
import warnings

import numpy as np
import pandas as pd
from IPython.display import display
from joblib import dump, load
from luciferml.supervised.utils.best import Best
from luciferml.supervised.utils.classificationPredictor import classificationPredictor
from luciferml.supervised.utils.configs import *
from luciferml.supervised.utils.confusionMatrix import confusionMatrix
from luciferml.supervised.utils.dimensionalityReduction import dimensionalityReduction
from luciferml.supervised.utils.encoder import encoder
from luciferml.supervised.utils.hyperTune import hyperTune
from luciferml.supervised.utils.intro import intro
from luciferml.supervised.utils.kfold import kfold
from luciferml.supervised.utils.predPreprocess import pred_preprocess
from luciferml.supervised.utils.sparseCheck import sparseCheck
from sklearn.metrics import accuracy_score


class Classification:
    def __init__(
        self,
        predictor="lr",
        params={},
        tune=False,
        test_size=0.2,
        cv_folds=10,
        random_state=42,
        pca_kernel="linear",
        n_components_lda=1,
        lda="n",
        pca="n",
        n_components_pca=2,
        hidden_layers=4,
        output_units=1,
        input_units=6,
        input_activation="relu",
        output_activation="sigmoid",
        optimizer="adam",
        metrics=[
            "accuracy",
        ],
        loss="binary_crossentropy",
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        tune_mode=1,
        smote="n",
        k_neighbors=1,
        dropout_rate=0,
        verbose=False,
        exclude_models=[],
        path=None,
    ):
        """
        Encode Categorical Data then Applies SMOTE , Splits the features and labels in training and validation sets with test_size = .2 , scales self.X_train, self.X_val using StandardScaler.
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
                                sgd - Stochastic Gradient Descent Classifier
                                perc - Perceptron
                                pass - Passive Aggressive Classifier
                                ridg - Ridge Classifier
                                svm -SupportVector Machine
                                knn - K-Nearest Neighbours
                                dt - Decision Trees
                                nb - GaussianNaive bayes
                                rfc- Random Forest self.Classifier
                                gbc - Gradient Boosting Classifier
                                ada - AdaBoost Classifier
                                bag - Bagging Classifier
                                extc - Extra Trees Classifier
                                lgbm - LightGBM Classifier
                                cat - CatBoost Classifier
                                xgb- XGBoost self.Classifier
                                ann - Artificial Neural Network
                                all - Applies all above classifiers
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
            hidden_layers : int
                    No. of default layers of ann. Default = 4
            inputs_units : int
                    No. of units in input layer. Default = 6
            output_units : int
                    No. of units in output layer. Default = 6
            input_activation : str
                    Activation function for Hidden layers. Default = 'relu'
            output_activation : str
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
            dropout_rate : int or float
                    rate for dropout layer. Default = 0
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
            verbose : boolean
                Verbosity of models. Default = False
            exclude_models : list
                List of models to be excluded when using predictor = 'all' . Default = []
            path : list
                List containing path to saved model and scaler. Default = None
                Example: [model.pkl, scaler.pkl]

        Returns:
            Dict Containing Name of Classifiers, Its K-Fold Cross Validated Accuracy and Prediction set
            Dataframe containing all the models and their accuracies when predictor is 'all'

        Example:

            from luciferml.supervised.classification import Classification

            dataset = pd.read_csv('Social_Network_Ads.csv')

            X = dataset.iloc[:, :-1]

            y = dataset.iloc[:, -1]

            classifier = Classification(predictor = 'lr')

            classifier.fit(X, y)

            result = classifier.result()

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
        self.dropout_rate = dropout_rate
        self.tune_mode = tune_mode
        self.rerun = False
        self.smote = smote
        self.k_neighbors = k_neighbors
        self.verbose = verbose
        self.exclude_models = exclude_models

        self.accuracy_scores = {}
        self.reg_result = {}
        self.accuracy = 0
        self.y_pred = []
        self.kfold_accuracy = 0
        self.classifier_name = ""
        self.sc = 0

        self.kfoldacc = []
        self.acc = []
        self.bestacc = []
        self.bestparams = []
        self.best_classifier_path = ""
        self.scaler_path = ""
        self.classifier_model = []
        self.result_df = pd.DataFrame(index=None)
        self.classifiers = copy.deepcopy(classifiers)
        for i in self.exclude_models:
            self.classifiers.pop(i)
        self.result_df["Name"] = list(self.classifiers.values())
        self.best_classifier = "First Run the Predictor in All mode"

        self.pred_mode = ""

        if path != None:
            try:
                self.classifier, self.sc = self.__load(path)
            except Exception as e:
                print(e)
                print("Model not found")

    def fit(self, features, labels):
        """[Takes Features and Labels and Encodes Categorical Data then Applies SMOTE , Splits the features and labels in training and validation sets with test_size = .2
        scales X_train, self.X_val using StandardScaler.
        Fits every model on training set and predicts results,
        finds accuracy of model applies K-Fold Cross Validation
        and stores its accuracies in a dictionary containing Model name as Key and accuracies as values and returns it
        Applies GridSearch Cross Validation and gives best params out from param list.]

        Args:
            features ([Pandas DataFrame]): [DataFrame containing Features]
            labels ([Pandas DataFrame]): [DataFrame containing Labels]
        """
        self.features = features
        self.labels = labels
        # Time Function ---------------------------------------------------------------------

        self.start = time.time()
        intro()
        print("Started LuciferML \n")
        if not self.rerun:
            # CHECKUP ---------------------------------------------------------------------
            if not isinstance(self.features, pd.DataFrame) and not isinstance(
                self.labels, pd.Series
            ):
                print(
                    "TypeError: This Function take features as Pandas Dataframe and labels as Pandas Series. Please check your implementation.\n"
                )
                self.end = time.time()
                print(self.end - self.start)
                return

            # Encoding ---------------------------------------------------------------------

            self.features, self.labels = encoder(self.features, self.labels)

            # Sparse Check -------------------------------------------------------------
            self.features, self.labels = sparseCheck(self.features, self.labels)

            # Preprocessing ---------------------------------------------------------------------
            (
                self.X_train,
                self.X_val,
                self.y_train,
                self.y_val,
                self.sc,
            ) = pred_preprocess(
                self.features,
                self.labels,
                self.test_size,
                self.random_state,
                self.smote,
                self.k_neighbors,
            )

            # Dimensionality Reduction---------------------------------------------------------------------
            self.X_train, self.X_val = dimensionalityReduction(
                self.lda,
                self.pca,
                self.X_train,
                self.X_val,
                self.y_train,
                self.n_components_lda,
                self.n_components_pca,
                self.pca_kernel,
                self.start,
            )

        # Models ---------------------------------------------------------------------
        if self.predictor == "all":
            self.pred_mode = "all"
            self.__fitall()
            return

        elif self.predictor == "ann":
            (
                self.parameters,
                self.classifier,
                self.classifier_wrap,
            ) = classificationPredictor(
                self.predictor,
                self.params,
                self.X_train,
                self.X_val,
                self.y_train,
                self.y_val,
                self.epochs,
                self.hidden_layers,
                self.input_activation,
                self.output_activation,
                self.loss,
                self.batch_size,
                self.metrics,
                self.validation_split,
                self.optimizer,
                self.output_units,
                self.input_units,
                self.tune_mode,
                self.dropout_rate,
                verbose=self.verbose,
            )

        else:
            self.parameters, self.classifier = classificationPredictor(
                self.predictor,
                self.params,
                self.X_train,
                self.X_val,
                self.y_train,
                self.y_val,
                self.epochs,
                self.hidden_layers,
                self.input_activation,
                self.output_activation,
                self.loss,
                self.batch_size,
                self.metrics,
                self.validation_split,
                self.optimizer,
                self.output_units,
                self.input_units,
                self.tune_mode,
                verbose=self.verbose,
            )

        try:

            if not self.predictor == "ann":
                self.classifier.fit(self.X_train, self.y_train)
        except Exception as error:
            print("Model Train Failed with error: ", error, "\n")

        print("Model Training Done [", u"\u2713", "]\n")
        print("Predicting Data [*]\n")
        try:
            self.y_pred = self.classifier.predict(self.X_val)
            print("Data Prediction Done [", u"\u2713", "]\n")
            if self.predictor == "ann":
                self.y_pred = (self.y_pred > 0.5).astype("int32")
        except Exception as error:
            print("Prediction Failed with error: ", error, "\n")

        # Confusion Matrix --------------------------------------------------------------
        confusionMatrix(self.y_pred, self.y_val)

        # Accuracy ---------------------------------------------------------------------
        print("""Evaluating Model Performance [*]""")
        try:
            self.accuracy = accuracy_score(self.y_val, self.y_pred)
            print("Validation Accuracy is :", self.accuracy)
            print("Evaluating Model Performance [", u"\u2713", "]\n")
        except Exception as error:
            print("Model Evaluation Failed with error: ", error, "\n")

        # K-Fold ---------------------------------------------------------------------
        if self.predictor == "ann":
            self.classifier_name, self.kfold_accuracy = kfold(
                self.classifier_wrap,
                self.predictor,
                self.X_train,
                self.y_train,
                self.cv_folds,
            )
        else:
            self.classifier_name, self.kfold_accuracy = kfold(
                self.classifier,
                self.predictor,
                self.X_train,
                self.y_train,
                self.cv_folds,
            )

        # GridSearch ---------------------------------------------------------------------
        if not self.predictor == "nb" and self.tune:
            self.__tuner()

        print("Complete [", u"\u2713", "]\n")
        self.end = time.time()
        print("Time Elapsed : ", self.end - self.start, "seconds \n")

    def __fitall(self):
        print("Training All Classifiers [*]\n")
        if self.params != {}:
            warnings.warn(params_use_warning, UserWarning)
            self.params = {}
        for _, self.predictor in enumerate(self.classifiers):
            if not self.predictor in self.exclude_models:
                if not self.predictor == "ann":
                    self.parameters, self.classifier = classificationPredictor(
                        self.predictor,
                        self.params,
                        self.X_train,
                        self.X_val,
                        self.y_train,
                        self.y_val,
                        self.epochs,
                        self.hidden_layers,
                        self.input_activation,
                        self.output_activation,
                        self.loss,
                        self.batch_size,
                        self.metrics,
                        self.validation_split,
                        self.optimizer,
                        self.output_units,
                        self.input_units,
                        self.tune_mode,
                        all_mode=True,
                        verbose=self.verbose,
                    )
                elif self.predictor == "ann":
                    (
                        self.parameters,
                        self.classifier,
                        self.classifier_wrap,
                    ) = classificationPredictor(
                        self.predictor,
                        self.params,
                        self.X_train,
                        self.X_val,
                        self.y_train,
                        self.y_val,
                        self.epochs,
                        self.hidden_layers,
                        self.input_activation,
                        self.output_activation,
                        self.loss,
                        self.batch_size,
                        self.metrics,
                        self.validation_split,
                        self.optimizer,
                        self.output_units,
                        self.input_units,
                        self.tune_mode,
                        self.dropout_rate,
                        all_mode=True,
                        verbose=self.verbose,
                    )
                try:

                    if not self.predictor == "ann":
                        self.classifier.fit(self.X_train, self.y_train)
                except Exception as error:
                    print(
                        classifiers[self.predictor],
                        "Model Train Failed with error: ",
                        error,
                        "\n",
                    )
                try:
                    self.y_pred = self.classifier.predict(self.X_val)
                except Exception as error:
                    print(
                        classifiers[self.predictor],
                        "Data Prediction Failed with error: ",
                        error,
                        "\n",
                    )
                if self.predictor == "ann":
                    self.y_pred = (self.y_pred > 0.5).astype("int32")

                # Accuracy ---------------------------------------------------------------------
                try:
                    self.accuracy = accuracy_score(self.y_val, self.y_pred)
                    self.acc.append(self.accuracy * 100)
                except Exception as error:
                    print(
                        classifiers[self.predictor],
                        "Evaluation Failed with error: ",
                        error,
                        "\n",
                    )

                # K-Fold ---------------------------------------------------------------------
                if self.predictor == "ann":
                    self.classifier_name, self.kfold_accuracy = kfold(
                        self.classifier_wrap,
                        self.predictor,
                        self.X_train,
                        self.y_train,
                        self.cv_folds,
                        all_mode=True,
                    )
                else:
                    self.classifier_name, self.kfold_accuracy = kfold(
                        self.classifier,
                        self.predictor,
                        self.X_train,
                        self.y_train,
                        self.cv_folds,
                        all_mode=True,
                    )
                self.kfoldacc.append(self.kfold_accuracy)
                self.classifier_model.append(self.classifier)
                # GridSearch ---------------------------------------------------------------------
                if not self.predictor == "nb" and self.tune:
                    self.__tuner(all_mode=True)
                if self.predictor == "nb":
                    self.best_params = ""
                    self.best_accuracy = self.kfold_accuracy
        self.result_df["Accuracy"] = self.acc
        self.result_df["KFold Accuracy"] = self.kfoldacc
        self.result_df["Model"] = self.classifier_model
        if self.tune:
            self.result_df["Best Parameters"] = self.bestparams
            self.result_df["Best Accuracy"] = self.bestacc

        self.best_classifier = Best(
            self.result_df.loc[self.result_df["KFold Accuracy"].idxmax()], self.tune
        )
        self.best_classifier_path, self.scaler_path = self.save(
            best=True, model=self.best_classifier.model, scaler=self.sc
        )
        print("Training All Classifiers Done [", u"\u2713", "]\n")
        print(
            "Saved Best Model to {} and its scaler to {}".format(
                self.best_classifier_path, self.scaler_path
            ),
            "\n",
        )
        display(self.result_df.iloc[:, :-1])
        print("Complete [", u"\u2713", "]\n")
        self.end = time.time()
        print("Time Elapsed : ", self.end - self.start, "seconds \n")
        return

    def __tuner(self, all_mode=False):
        if not all_mode:
            print(
                "Applying Grid Search Cross validation on Mode {} [*]".format(
                    self.tune_mode
                )
            )
        if self.predictor == "ann":
            self.best_params, self.best_accuracy = hyperTune(
                self.classifier_wrap,
                self.parameters,
                self.X_train,
                self.y_train,
                self.cv_folds,
                self.tune_mode,
                all_mode=all_mode,
            )
        else:
            self.best_params, self.best_accuracy = hyperTune(
                self.classifier,
                self.parameters,
                self.X_train,
                self.y_train,
                self.cv_folds,
                self.tune_mode,
                all_mode=all_mode,
            )

        self.bestparams.append(self.best_params)
        self.bestacc.append(self.best_accuracy * 100)

    def result(self):
        """[Makes a dictionary containing Classifier Name, K-Fold CV Accuracy, RMSE, Prediction set.]

        Returns:
            [dict]: [Dictionary containing :
                        - "Classifier" - Classifier Name
                        - "Accuracy" - KFold CV Accuracy
                        - "YPred" - Array for Prediction set
                        ]
            [dataframe] : [Dataset containing accuracy and best_params
                            for all predictors only when predictor = 'all' is used
                            ]
        """
        if not self.pred_mode == "all":
            self.reg_result["Classifier"] = self.classifier_name
            self.reg_result["Accuracy"] = self.kfold_accuracy
            self.reg_result["YPred"] = self.y_pred

            return self.reg_result
        if self.pred_mode == "all":
            return self.result_df

    def predict(self, X_test):
        """[Takes test set and returns predictions for that test set]

        Args:
            X_test ([Array]): [Array Containing Test Set]

        Returns:
            [Array]: [Predicted set for given test set]
        """
        if not self.pred_mode == "all":
            X_test = np.array(X_test)
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)

            y_test = self.classifier.predict(self.sc.transform(X_test))

            return y_test
        if self.pred_mode == "all":
            raise TypeError("Predict is only applicable on single predictor")

    def save(self, path=None, **kwargs):
        """
        Saves the model and its scaler to a file provided with a path.
        If no path is provided will create a directory named
        lucifer_ml_info/models/ and lucifer_ml_info/scaler/ in current working directory
        Args:
            path ([list]): [List containing path to save the model and scaler.]
                Example: path = ["model.pkl", "scaler.pkl"]

        Returns:
            Path to the saved model and its scaler.
        """
        if not type(path) == list and path != None:
            raise TypeError("Path must be a list")

        if not self.predictor == "all":
            dir_path_model = path[0] if path else "lucifer_ml_info/models/classifier/"
            dir_path_scaler = path[1] if path else "lucifer_ml_info/scalers/classifier/"
            if kwargs.get("best"):
                dir_path_model = "lucifer_ml_info/best/classifier/models/"
                dir_path_scaler = "lucifer_ml_info/best/classifier/scalers/"
            os.makedirs(dir_path_model, exist_ok=True)
            os.makedirs(dir_path_scaler, exist_ok=True)
            timestamp = str(int(time.time()))
            path_model = (
                dir_path_model
                + classifiers[self.predictor].replace(" ", "_")
                + "_"
                + timestamp
                + ".pkl"
            )
            path_scaler = (
                dir_path_scaler
                + classifiers[self.predictor].replace(" ", "_")
                + "_"
                + "Scaler"
                + "_"
                + timestamp
                + ".pkl"
            )
            if (
                not kwargs.get("model")
                and not kwargs.get("best")
                and not kwargs.get("scaler")
            ):
                dump(self.classifier, open(path_model, "wb"))
                dump(self.sc, open(path_scaler, "wb"))
            else:
                dump(kwargs.get("model"), open(path_model, "wb"))
                dump(kwargs.get("scaler"), open(path_scaler, "wb"))
            if not kwargs.get("best"):
                print(
                    "Model Saved at {} and Scaler at {}".format(path_model, path_scaler)
                )
            return path_model, path_scaler
        else:
            raise Exception(
                "[Error] This method is only applicable on single predictor"
            )

    def __load(self, path=None):
        """
        Loads model and scaler from the specified path
        Args:
            path ([list]): [List containing path to load the model and scaler.]
                Example: path = ["model.pkl", "scaler.pkl"]

        Returns:
            [Model] : [Loaded model]
            [Scaler] : [Loaded scaler]
        """

        model_path = path[0] if path[0] else None
        scaler_path = path[1] if path[1] else None
        if not ".pkl" in model_path and not model_path == None:
            raise TypeError(
                "[Error] Model Filetype not supported. Please use .pkl type "
            )
        if not ".pkl" in scaler_path and not scaler_path == None:
            raise TypeError(
                "[Error] Scaler Filetype not supported. Please use .pkl type "
            )
        if model_path != None and scaler_path != None:
            model = load(open(model_path, "rb"))
            scaler = load(open(scaler_path, "rb"))
            print(
                "[Info] Model and Scaler Loaded from {} and {}".format(
                    model_path, scaler_path
                )
            )
            return model, scaler
        elif model_path != None and scaler_path == None:
            model = load(open(model_path, "rb"))
            print("[Info] Model Loaded from {}".format(model_path))
            return model
        elif model_path == None and scaler_path != None:
            scaler = load(open(scaler_path, "rb"))
            print("[Info] Scaler Loaded from {}".format(scaler_path))
            return scaler
        else:
            raise ValueError("No path specified.Please provide actual path\n")
