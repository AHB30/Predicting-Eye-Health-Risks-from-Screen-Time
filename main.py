from src.data_preparation import load_dataset, create_labels
from src.feature_engineering import engineer_features
from src.model_training import train_all_models
from src.model_visualisation import plot_svm_decision_boundary, plot_all_models

def main():
    df = load_dataset()
    df = create_labels(df)
    df = engineer_features(df)
    
    best_model = train_all_models(df)

    plot_svm_decision_boundary(df)
    plot_all_models(df)

    print("\n✔ Pipeline completed successfully.")
    print(f"✔ Best performing model: {best_model}")

if __name__ == "__main__":
    main()
