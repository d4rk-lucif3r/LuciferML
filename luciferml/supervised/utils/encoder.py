from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


def encoder(features, labels):
    """
    Takes features and labels as arguments and encodes features using onehot encoding and labels with label encoding. 
    Returns Encoded Features and Labels.
    """
    try:
        print('Checking if labels or features are categorical! [*]\n')
        cat_features = [
            i for i in features.columns if features.dtypes[i] == 'object']
        if len(cat_features) >= 1:
            index = []
            for i in range(0, len(cat_features)):
                index.append(features.columns.get_loc(cat_features[i]))
            print('Features are Categorical\n')
            ct = ColumnTransformer(
                transformers=[('encoder', OneHotEncoder(), index)], remainder='passthrough')
            print('Encoding Features [*]\n')
            features = np.array(ct.fit_transform(features))
            print('Encoding Features Done [', u'\u2713', ']\n')
        if labels.dtype == 'O':
            le = LabelEncoder()
            print('Labels are Categorical [*] \n')
            print('Encoding Labels \n')
            labels = le.fit_transform(labels)
            print('Encoding Labels Done [', u'\u2713', ']\n')
        else:
            print(
                'Features and labels are not categorical [', u'\u2713', ']\n')
        print('Checking for Categorical Variables Done [', u'\u2713', ']\n')
        return (features, labels)
    except Exception as error:
        print('Encoding Failed with error :', error)
