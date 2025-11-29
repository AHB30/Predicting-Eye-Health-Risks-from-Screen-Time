import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.combine import SMOTETomek

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def train_all_models(df):
    X = df.drop(columns=[
        'Health_Impacts', 'Eye_Strain_Present',
        'Eye_Strain_Risk_Level', 'ScreenTime_Class'
    ])
    X = pd.get_dummies(X, drop_first=True)

    y = df['Eye_Strain_Risk_Level']

    num_cols = X.select_dtypes(include=['int64','float64']).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    sampler = SMOTETomek(random_state=42)
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=8, min_samples_leaf=3,
            criterion='entropy', class_weight='balanced', random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=8,
            min_samples_leaf=3, class_weight='balanced', random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=5000, solver='lbfgs', class_weight='balanced', C=2.0
        ),
        "Linear SVM": SVC(kernel="linear", C=1.0, class_weight="balanced")
    }

    best_acc = 0
    best_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print("\n", "="*60)
        print(name)
        print("Accuracy:", acc)
        print(classification_report(y_test, preds))

        if acc > best_acc:
            best_acc = acc
            best_name = name

    print("\nBest Model:", best_name, "| Accuracy =", best_acc)
    return best_name
