from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def pred_preprocess(features, labels, test_size, random_state, smote, k_neighbors):

    try:
        if smote == 'y':
            print('Applying SMOTE [*]\n')

            sm = SMOTE(k_neighbors=k_neighbors)
            features, labels = sm.fit_resample(
                features, labels)
            print('SMOTE Done [', u'\u2713', ']\n')

        # Splitting ---------------------------------------------------------------------
        print('Splitting Data into Train and Validation Sets [*]\n')

        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=test_size, random_state=random_state)
        print('Splitting Done [', u'\u2713', ']\n')

        # Scaling ---------------------------------------------------------------------
        print('Scaling Training and Test Sets [*]\n')

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_val = sc.transform(X_val)
        print('Scaling Done [', u'\u2713', ']\n')

        return (X_train, X_val, y_train, y_val, sc)

    except Exception as error:
        print('Preprocessing Failed with error: ', error, '\n')
