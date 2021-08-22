from scipy.special import boxcox1p
import pandas as pd
import time
import seaborn as sns
from scipy.stats import norm, skew, probplot
import matplotlib.pyplot as plt
from luciferml.supervised.utils.intro import *

class Preprocess:
    def __plotter(dataset, column_name, text, color):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        sns.distplot(dataset[column_name], fit=norm, color=color,
                     label="Skewness: %.2f" % (dataset[column_name].skew()))
        plt.title(column_name.capitalize() +
                  " Distplot for {} {} Skewness Transformation".format(column_name, text), color="black")
        plt.legend()
        plt.subplot(1, 2, 2)

        probplot(dataset[column_name], plot=plt)
        plt.show()

    def __skewcheck(dataset, except_columns):
        numeric_feats = dataset.dtypes[dataset.dtypes !=
                    "object"].index
        if not len(except_columns) == 0:
            if len(except_columns) > len(numeric_feats):
                numeric_feats = set(except_columns)-set(numeric_feats)
            else:
                numeric_feats = set(numeric_feats) - set(except_columns)
        skewed_feats = dataset[numeric_feats].apply(
            lambda x: skew(x.dropna())).sort_values(ascending=False)
        print("\nSkewness in numerical features: \n")
        skewness = pd.DataFrame( skewed_feats, columns=[ 'Skewness'])
        print(skewness.head(10))
        skew_dict = dict(skewness['Skewness'])
        skewed_features = skewness.index
        return (skewed_features, skew_dict)

    def skewcorrect(dataset, except_columns=[]) -> pd.DataFrame:
        """
            Plots distplot and probability plot for non-normalized data and after normalizing the provided data.
            Normalizes data using boxcox normalization
        
        Parameters:
            dataset : pd.DataFrame
                Dataset on which skewness correction has to be done.
            except_columns : list
                Columns for which skewness correction need not to be done.Default = []
                
        :returns: Scaled Dataset
        :rtype: pd.DataFrame        
        
        Example:

         1) All Columns

                     from luciferml.preprocessing import Preprocess as prep

                     import pandas as pd

                     dataset = pd.read_csv('/examples/Social_Network_Ads.csv')

                     dataset = prep.skewcorrect(dataset)

         2) Except column/columns

                     from luciferml.preprocessing import Preprocess as prep

                     import pandas as pd

                     dataset = pd.read_csv('/examples/Social_Network_Ads.csv')

                     dataset = prep.skewcorrect(dataset,except_columns=['Purchased'])
        

        """
        try:
            start = time.time()
            intro()
            print("Started Preprocessor \n")
            if not isinstance(dataset, pd.DataFrame) :
                print('TypeError: This Function expects  Pandas Dataframe but {}'.format(type(dataset)),' is given \n')
                end = time.time()
                print('Elapsed Time: ',end - start, 'seconds\n')
                return

            (skewed_features, skew_dict) =  Preprocess.__skewcheck(dataset, except_columns)
            for column_name in skewed_features:
                lam = 0    
                (mu, sigma) = norm.fit(dataset[column_name])
                print('Skewness Before Transformation for {}: '.format(
                    column_name), dataset[column_name].skew(), '\n')
                print("Mean before Transformation for {} : {}, Standard Deviation before Transformation for {} : {}".format(
                    column_name.capitalize(), mu, column_name.capitalize(), sigma),'\n')
                Preprocess.__plotter(dataset, column_name,
                                    'Before', "lightcoral")
                try:
                    if skew_dict[column_name] > .75:
                        lam = .15
                    dataset[column_name] = boxcox1p(dataset[column_name], lam)
                    print('Skewness After Transformation for {}: '.format(
                        column_name), dataset[column_name].skew(), '\n')
                    (mu, sigma) = norm.fit(dataset[column_name])
                    print("Mean before Transformation for {} : {}, Standard Deviation before Transformation for {} : {}".format(
                        column_name.capitalize(), mu, column_name.capitalize(), sigma),'\n')
                    Preprocess.__plotter(dataset, column_name, 'After', 'orange')
                except Exception as error:
                    print('\nPlease check your dataset\'s column :', column_name,
                    'Raised Error: ', error,'\n')
                    pass
            end = time.time()
            print('Elapsed Time: ', end - start, 'seconds\n')
            return dataset

        except Exception as error:
            print('\033[41m','Skewness Correction Failed with error : ', error, '\n')

