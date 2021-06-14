# %%
import os
os.chdir('../')
#%%
from luciferml.supervised import classification as cls
from luciferml.preprocessing import Preprocess as prep
import pandas as pd


#%%

dataset = pd.read_csv('examples/Social_Network_Ads.csv')
dataset.head()
#%%

dataset = prep.skewcorrect(dataset,except_columns=['Purchased'])

#%%
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
#%%
name, accuracy = cls.Classification(predictor='lr', cv_folds=5, epochs = 5, 
).predict(X, y)

# cls.Classification(predictor='svm', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='knn', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='dt', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='nb', cv_folds = 5, pca='y').predict(X, y)
# cls.Classification(predictor='rfc', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='xgb', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='ann', epochs = 5, cv_folds = 5, pca='y').predict(X, y)


# %%
