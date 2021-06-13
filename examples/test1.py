# %%
import os
# os.chdir('../')
#%%
from luciferml.supervised import classification as cls
import pandas as pd


#%%

dataset = pd.read_csv('examples/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


name, accuracy = cls.Classification(predictor='ann', cv_folds=5, epochs = 5, tune =True).predict(X, y)
# cls.Classification(predictor='svm', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='knn', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='dt', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='nb', cv_folds = 5, pca='y').predict(X, y)
# cls.Classification(predictor='rfc', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='xgb', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='ann', epochs = 5, cv_folds = 5, pca='y').predict(X, y)


# %%
