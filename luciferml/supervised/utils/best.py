class Best:
    """
    Best is used to utilise the best model when predictor = 'all' is used.

    """

    def __init__(self, best_model, tune, isReg=False):
        self.__best_model = best_model
        self.model = self.__best_model["Model"]
        self.name = self.__best_model["Name"]
        if isReg:
            self.r2_score = self.__best_model["R2 Score"]
            self.mae = self.__best_model["Mean Absolute Error"]
            self.rmse = self.__best_model["Root Mean Squared Error"]
        if not isReg:
            self.accuracy = self.__best_model["Accuracy"]
        self.kfold_acc = self.__best_model["KFold Accuracy"]
        if tune == True:
            self.best_params = self.__best_model["Best Parameters"]
        else:
            self.best_params = "Run with tune = True to get best parameters"
        self.isReg = isReg

    def predict(self, pred):
        """Predicts the output of the best model"""
        prediction = self.__best_model["Model"].predict(pred)
        return prediction
