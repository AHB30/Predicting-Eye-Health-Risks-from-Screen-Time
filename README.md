# Predicting Eye Health Risks from Screen Time

This project delivers a modular machine-learning pipeline designed to predict **eye-strain risk levels** in children based on **screen time habits**, **behavioural patterns**, and **lifestyle indicators**. The system adheres to industry-grade engineering practices, ensuring reproducibility, maintainability, and clarity across all modules.

---

## 🚀 Key Features
- **Automated Data Preparation:** Cleans, validates, and formats the raw dataset for downstream analysis.
- **Strategic Feature Engineering:** Extracts behavioural and environmental predictors that drive accurate modelling.
- **Imbalance Handling:** Employs **SMOTETomek** to balance class distributions and improve model generalisation.
- **Multiple Classifiers:** Includes Decision Tree, Random Forest, Logistic Regression, and SVM.
- **PCA Visualisation:** Generates visual representations of the classifier decision regions.
- **Modular Architecture:** Every stage of the pipeline is compartmentalised for clarity and easy enhancement.
- **Reproducible Structure:** Aligned with standard ML engineering layouts.

---

## 📁 Project Structure
```
src/
├── data_preparation.py
├── feature_engineering.py
├── model_training.py
├── model_visualisation.py
└── utils.py

data/
└── raw/
    └── Indian_Kids_Screen_Time.csv

main.py
```

---

## ▶ Running the Project
```bash
python main.py
```

---

## 📦 Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset
The primary dataset used in this pipeline is:
```
data/raw/Indian_Kids_Screen_Time.csv
```
Ensure the dataset path is correct before running `main.py`.

---

## 🧩 Future Enhancements
- Integration of additional lifestyle features.
- Hyperparameter tuning using GridSearchCV or Optuna.
- Deployment-ready API wrapper.

---

## 📜 Licence
This project is released under the MIT Licence.
