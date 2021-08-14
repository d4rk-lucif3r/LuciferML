#%%
import os
os.chdir('../')
#%%
from luciferml.supervised.classification import Classification
import pandas as pd
# %%

# %%
dataset = pd.read_csv('examples\Social_Network_Ads.csv')
dataset.head()

# %%
# %%
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# %%
predictors = ['cat']

for predictor in predictors:
    classifier = Classification(
        predictor=predictor, params={}, cv_folds=2, epochs=10, tune=True)
    classifier.fit(X, y)
    regression = classifier.result()
# %%
