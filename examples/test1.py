# %%
import os
os.chdir('../')
#%%
from luciferml.supervised.regression import Regression
from luciferml.supervised.classification import Classification
import pandas as pd

# %%
# dataset = pd.read_excel('examples\Folds5x2_pp.xlsx') #regression
dataset = pd.read_csv('examples\Social_Network_Ads.csv') #classification
dataset.head()
# %%

# dataset = prep.skewcorrect(dataset,except_columns=['Purchased'])

# %%
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
# %%
# model = Regression(predictor='lin',
#                  params={},
#                  cv_folds=5)
model = Classification()
#%%
model.fit(X, y)

# %%
result = model.result()
# %%
result
# %%
