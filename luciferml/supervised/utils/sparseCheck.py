import scipy


def sparseCheck(features, labels):
    features = features
    labels = labels
    """
    Takes features and labels as input and checks if any of those is sparse csr_matrix.
    """
    try:
        print('Checking for Sparse Matrix [*]\n')
        if scipy.sparse.issparse(features[()]):
            print('Converting Sparse Features to array []\n')
            features = features[()].toarray()
            print(
                'Conversion of Sparse Features to array Done [', u'\u2713', ']\n')

        elif scipy.sparse.issparse(labels[()]):
            print('Converting Sparse Labels to array []\n')
            labels = labels[()].toarray()
            print(
                'Conversion of Sparse Labels to array Done [', u'\u2713', ']\n')

        else:
            print('No Sparse Matrix Found')

    except Exception as error:
        # print('Sparse matrix Check failed with KeyError: ', error)
        pass
    return (features, labels)
