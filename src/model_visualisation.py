import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def plot_svm_decision_boundary(df):
    X = df.select_dtypes(include=['float64','int64'])
    y = df['Eye_Strain_Risk_Level']

    pca = PCA(n_components=2)
    X_plot = pca.fit_transform(X)

    svm = SVC(kernel="linear")
    svm.fit(X_plot, y)

    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, s=12)
    plt.title("SVM – Decision Boundary")
    plt.show()


def plot_all_models(df):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    X = df.select_dtypes(include=['float64','int64'])
    y = df['Eye_Strain_Risk_Level']

    pca = PCA(n_components=2)
    X_plot = pca.fit_transform(X)

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Linear SVM": SVC(kernel="linear")
    }

    for name, model in models.items():
        model.fit(X_plot, y)

        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 500),
            np.linspace(y_min, y_max, 500)
        )

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, s=12)
        plt.title(f"{name} – Decision Regions")
        plt.show()
