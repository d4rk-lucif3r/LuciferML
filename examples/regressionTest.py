# %%
import os

os.chdir("../")
#%%
from luciferml.supervised.regression import Regression

# from luciferml.preprocessing import Preprocess as prep
import pandas as pd


#%%

dataset = pd.read_excel("examples\Folds5x2_pp.xlsx")
dataset.head()
#%%

# dataset = prep.skewcorrect(dataset,except_columns=['Purchased'])

#%%
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
#%%
regressor = Regression(
    predictor="all",
    cv_folds=2,
    epochs=5,
    tune=True,
)
df = regressor.fit(X, y)
#%%
regression = regressor.result()
print(regression)

# %%
prediction = regressor.predict(
    [
        14.96,
        41.76,
        1024.07,
        73.17,
    ]
)

# %%
