# Classification

Encode Categorical Data then Applies SMOTE , Splits the features and labels in training and validation sets with test_size = .2 , scales self.X_train, self.X_val using StandardScaler.
Fits every model on training set and predicts results find and plots Confusion Matrix,
finds accuracy of model applies K-Fold Cross Validation
and stores accuracy in variable name accuracy and model name in self.classifier name and returns both as a tuple.
Applies GridSearch Cross Validation and gives best self.params out from param list.

        Parameters:
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

Returns:

            Dict Containing Name of Classifiers, Its K-Fold Cross Validated Accuracy and Prediction set

            Dataframe containing all the models and their accuracies when predictor is 'all'

Example :

        from luciferml.supervised.classification import Classification
        dataset = pd.read_csv('Social_Network_Ads.csv')
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        classifier = Classification(predictor = 'lr')
        classifier.fit(X, y)
        result = classifier.result()
