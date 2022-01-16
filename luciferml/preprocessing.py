import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Fore
from IPython.display import display
from scipy.special import boxcox1p
from scipy.stats import norm, probplot, skew

from luciferml.supervised.utils.configs import intro


class Preprocess:
    
    def __init__(self, dataset, columns, except_columns = []):
        self.__dataset = dataset
        self.__columns = columns
        self.__except_columns = except_columns
    
    def __plotter(self, name, text, color):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        sns.distplot(
            self.__dataset[self.column_name],
            fit=norm,
            color=color,
            label="Skewness: %.2f" % (self.__dataset[name].skew()),
        )
        plt.title(
            name.capitalize()
            + " Distplot for {} {} Skewness Transformation".format(name, text),
            color="black",
        )
        plt.legend()
        plt.subplot(1, 2, 2)

        probplot(self.__dataset[name], plot=plt)
        plt.show()

    def __skewcheck(self):
        numeric_feats = self.__dataset.dtypes[self.__dataset.dtypes !=
                                            "object"].index
        if not len(self.__except_columns) == 0:
            if len(self.__except_columns) > len(numeric_feats):
                numeric_feats = set(self.__except_columns) - set(numeric_feats)
            else:
                numeric_feats = set(numeric_feats) - set(self.__except_columns)
        skewed_feats = (
            self.__dataset[numeric_feats]
            .apply(lambda x: skew(x.dropna()))
            .sort_values(ascending=False)
        )
        print(Fore.GREEN + "\nSkewness in numerical features: \n")
        skewness = pd.DataFrame(skewed_feats, columns=["Skewness"])
        display(skewness)
        skew_dict = dict(skewness["Skewness"])
        skewed_features = skewness.index
        return (skewed_features, skew_dict)

    def skewcorrect(self) -> pd.DataFrame:
        """
            Plots distplot and probability plot for non-normalized data and after normalizing the provided data.
            Normalizes data using boxcox normalization

        :returns: Scaled Dataset
        :rtype: pd.DataFrame

        Example:

         1) All Columns

                     from luciferml.preprocessing import Preprocess as pp

                     import pandas as pd

                     dataset = pd.read_csv('/examples/Social_Network_Ads.csv')
                     prep = pp(dataset, dataset.columns)
                     dataset = prep.skewcorrect(dataset)

         2) Except column/columns

                     from luciferml.preprocessing import Preprocess as pp

                     import pandas as pd

                     dataset = pd.read_csv('/examples/Social_Network_Ads.csv')
                     prep = pp(dataset, dataset.columns, except_columns=['Purchased'])
                     dataset = prep.skewcorrect()


        """
        try:
            start = time.time()
            print(Fore.MAGENTA + intro, "\n")
            print(Fore.GREEN + "Started LuciferML [", "\u2713", "]\n")
            if not isinstance(self.__dataset, pd.DataFrame):
                print(
                    Fore.RED + "TypeError: This Function expects  Pandas Dataframe but {}".format(
                        type(self.__dataset)
                    ),
                    " is given \n",
                )
                end = time.time()
                print(Fore.GREEN + "Elapsed Time: ", end - start, "seconds\n")
                return

            (skewed_features, skew_dict) = self.__skewcheck()
            for column_name in skewed_features:
                lam = 0
                (mu, sigma) = norm.fit(self.__dataset[column_name])
                print(
                    Fore.CYAN +
                    "Skewness Before Transformation for {}: ".format(
                        column_name),
                    self.__dataset[column_name].skew(),
                    "\n",
                )
                print(
                    Fore.CYAN + "Mean before Transformation for {} : {}, Standard Deviation before Transformation for {} : {}".format(
                        column_name.capitalize(), mu, column_name.capitalize(), sigma
                    ),
                    "\n",
                )
                self.__plotter(
                    self.__dataset, column_name, "Before", "lightcoral")
                try:
                    if skew_dict[column_name] > 0.75:
                        lam = 0.15
                    self.__dataset[column_name] = boxcox1p(
                        self.__dataset[column_name], lam)
                    print(
                        Fore.GREEN +
                        "Skewness After Transformation for {}: ".format(
                            column_name),
                        self.__dataset[column_name].skew(),
                        "\n",
                    )
                    (mu, sigma) = norm.fit(self.__dataset[column_name])
                    print(
                        Fore.GREEN + "Mean before Transformation for {} : {}, Standard Deviation before Transformation for {} : {}".format(
                            column_name.capitalize(),
                            mu,
                            column_name.capitalize(),
                            sigma,
                        ),
                        "\n",
                    )
                    self.__plotter(
                        self.__dataset, column_name, "After", "orange")
                except Exception as error:
                    print(
                        Fore.RED + "\nPlease check your dataset's column :",
                        column_name,
                        "Raised Error: ",
                        error,
                        "\n",
                    )
                    pass
            end = time.time()
            print(Fore.GREEN + "Elapsed Time: ", end - start, "seconds\n")
            return self.__dataset

        except Exception as error:
            print(Fore.RED + "Skewness Correction Failed with error : ", error, "\n")

    def detect_outliers(self):
        """
        This function takes dataset and columns as input and finds Q1, Q3 and IQR for that list of column
        Detects the outlier and it index and stores them in a list.
        Then it creates as counter object with that list and stores it
        in Multiple Outliers list if the value of outlier is greater than 1.5
        
        Ex:
            1) For printing no. of outliers.
                print("number of outliers detected --> ",
                len(dataset.loc[detect_outliers(dataset, dataset.columns[:-1])]))
            2) Printing rows and columns collecting the outliers
                dataset.loc[detect_outliers(dataset.columns[:-1])]
            3) Dropping those detected outliers
                dataset = dataset.drop(detect_outliers(dataset.columns[:-1]),axis = 0).reset_index(drop = True)
        """
        outlier_indices = []
        for column in self.__columns:
            Q1 = np.percentile(self.__dataset[column], 25)
            Q3 = np.percentile(self.__dataset[column], 75)
            IQR = Q3 - Q1
            outlier_step = IQR * 1.5
            outlier_list_col = self.__dataset[(self.__dataset[column] < Q1 - outlier_step)
                                            | (self.__dataset[column] > Q3 + outlier_step)].index
            outlier_indices.extend(outlier_list_col)
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list(i for i, v in outlier_indices.items() if v > 1.5)
        return multiple_outliers

    def preprocess(self):
        
        display(self.__datasetdescribe().T.style.bar(
            subset=['mean'],
            color='#606ff2').background_gradient(
            subset=['std'], cmap='PuBu').background_gradient(subset=['50%'], cmap='PuBu'))
        