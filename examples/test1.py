# %%
from sklearn.metrics import mean_squared_error
import os
os.chdir('../')
#%%
from luciferml.supervised import regression as reg
# from luciferml.preprocessing import Preprocess as prep
import pandas as pd


#%%

dataset = pd.read_excel('examples\Folds5x2_pp.xlsx')
dataset.head()
#%%

# dataset = prep.skewcorrect(dataset,except_columns=['Purchased'])

#%%
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
#%%
name, accuracy = reg.Regression(predictor='ann', 
                                params={}, 
                                cv_folds=5, batch_size= 20, input_units= 5, epochs= 200,
                               
).predict(X, y)

# cls.Classification(predictor='svm', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='knn', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='dt', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='nb', cv_folds = 5, pca='y').predict(X, y)
# cls.Classification(predictor='rfc', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='xgb', cv_folds=5, pca='y').predict(X, y)
# cls.Classification(predictor='ann', epochs = 5, cv_folds = 5, pca='y').predict(X, y)


# %%

