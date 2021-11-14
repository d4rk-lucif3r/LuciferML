from joblib import dump
import os
import time
class Best():
    """
    Best is used to utilise the best model when predictor = 'all' is used.
    
    """

    def __init__(self,best_model,tune, isReg = False):
        self.__best_model = best_model
        self.model = self.__best_model['Model']
        self.name = self.__best_model['Name']
        if isReg:
            self.r2_score = self.__best_model['R2 Score']
            self.mae = self.__best_model['Mean Absolute Error']
            self.rmse = self.__best_model['Root Mean Squared Error']
        if not isReg:
            self.accuracy = self.__best_model['Accuracy']    
        self.kfold_acc = self.__best_model['KFold Accuracy']            
        if tune == True:
            self.best_params = self.__best_model['Best Parameters']
        else:
            self.best_params = 'Run with tune = True to get best parameters'
        self.isReg = isReg
    def predict(self,pred):
        """Predicts the output of the best model"""
        prediction = self.__best_model['Model'].predict(pred)
        return prediction
    def save_model(self,path=None):
        '''
        Saves the best model to a file provided with a path. 
        If no path is provided will create a directory named
        lucifer_ml_info/best_models/ in current working directory
        Returns the path to the saved model.
        '''
        if path == None:
            if self.isReg:
                
                dir_path = 'lucifer_ml_info/best_models/regression/'
            if not self.isReg:
                dir_path = 'lucifer_ml_info/best_models/classification/'
            os.makedirs(dir_path,exist_ok=True)
            path = dir_path+self.name.replace(' ', '_')+'_'+str(int(time.time()))+'.joblib'
        file_path = dump(self.model,path)
        return file_path

