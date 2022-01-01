import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from luciferml.supervised.utils.configs import *
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from colorama import Fore


class PreProcesser:
    def data_preprocess(
        self, features, labels, test_size, random_state, smote, k_neighbors
    ):
        try:
            if smote == "y":
                sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
                features, labels = sm.fit_resample(features, labels)
            # Splitting ---------------------------------------------------------------------
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels, test_size=test_size, random_state=random_state
            )
            # Scaling ---------------------------------------------------------------------
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_val = sc.transform(X_val)
            return (X_train, X_val, y_train, y_val, sc)
        except Exception as error:
            print(Fore.RED + "Preprocessing Failed with error: ", error, "\n")

    def confusion_matrix(self, y_pred, y_val):
        """
        Takes Predicted data and Validation data as input and prepares and plots Confusion Matrix.
        """
        try:
            cm = confusion_matrix(y_val, y_pred)
            ax = plt.subplot()
            sns.heatmap(cm, annot=True, fmt="g", ax=ax)
            ax.set_xlabel("Predicted labels")
            ax.set_ylabel("True labels")
            ax.set_title("Confusion Matrix")
            ax.xaxis.set_ticklabels(np.unique(y_val))
            ax.yaxis.set_ticklabels(np.unique(y_val))
            plt.show()
        except Exception as error:
            print(
                Fore.RED + "Building Confusion Matrix Failed with error :", error, "\n"
            )

    def dimensionality_reduction(
        self,
        lda,
        pca,
        X_train,
        X_val,
        y_train,
        n_components_lda,
        n_components_pca,
        pca_kernel,
        start,
    ):
        """
        Performs Dimensionality Reduction on Training and Validation independent variables.
        """
        try:
            if lda == "y":
                lda = LDA(n_components=n_components_lda)
                X_train = lda.fit_transform(X_train, y_train)
                X_val = lda.transform(X_val)
            if pca == "y" and not lda == "y":
                if not pca_kernel == "linear":
                    try:

                        kpca = KernelPCA(
                            n_components=n_components_pca, kernel=pca_kernel
                        )
                        X_train = kpca.fit_transform(X_train)
                        X_val = kpca.transform(X_val)
                    except MemoryError as error:
                        print(error)
                        end = time.time()
                        print("Time Elapsed :", end - start)
                        return

                elif pca_kernel == "linear":
                    pca = PCA(n_components=n_components_pca)
                    X_train = pca.fit_transform(X_train)
                    X_val = pca.transform(X_val)
                else:
                    print("Un-identified PCA Kernel\n")
                    return
            return (X_train, X_val)
        except Exception as error:
            print(
                Fore.RED + "Dimensionality Reduction Failed with error :", error, "\n"
            )
            return (X_train, X_val)

    def encoder(self, features, labels):
        """
        Takes features and labels as arguments and encodes features using onehot encoding and labels with label encoding.
        Returns Encoded Features and Labels.
        """
        try:
            cat_features = [
                i for i in features.columns if features.dtypes[i] == "object"
            ]
            if len(cat_features) >= 1:
                index = []
                for i in range(0, len(cat_features)):
                    index.append(features.columns.get_loc(cat_features[i]))
                ct = ColumnTransformer(
                    transformers=[("encoder", OneHotEncoder(), index)],
                    remainder="passthrough",
                )
                print("Encoding Features [*]\n")
                features = np.array(ct.fit_transform(features))
            if labels.dtype == "O":
                le = LabelEncoder()
                labels = le.fit_transform(labels)
            return (features, labels)
        except Exception as error:
            print(Fore.RED + "Encoding Failed with error :", error)

    def permutational_feature_imp(self, features, X_test, y_test, model):
        perm_importance = permutation_importance(model, X_test, y_test)
        sorted_idx = perm_importance.importances_mean.argsort()
        plt.barh(
            features.columns[sorted_idx], perm_importance.importances_mean[sorted_idx]
        )
        plt.xlabel("Feature Importance")

    def shap_feature_imp(self, features, X_train, model, *args, **kwargs):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(
            shap_values,
            X_train,
            feature_names=features.columns,
            plot_type="bar",
            *args,
            **kwargs
        )
        shap.summary_plot(
            shap_values, X_train, feature_names=features.columns, *args, **kwargs
        )
        for i in range(len(features.columns)):
            shap.dependence_plot(
                i, shap_values, X_train, feature_names=features.columns, *args, **kwargs
            )
