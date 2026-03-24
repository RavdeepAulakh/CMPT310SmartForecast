from src.data_preprocessing import load_data, split_data
from src.eda import plot_score_distribution, plot_correlations, scatter_relationships
from src.baseline_model import train_baseline
from src.model_experiments import random_forest_model, svr_model
import pandas as pd


def main():
    df = load_data("data/exams.csv")

    # -------------------------
    # EDA (Milestone 2)
    # -------------------------
    plot_score_distribution(df)
    plot_correlations(df)
    scatter_relationships(df)

    # -------------------------
    # Train/Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = split_data(df)

    # -------------------------
    # Models (Milestone 3)
    # -------------------------
    results = []

    # Baseline
    baseline_result = train_baseline(X_train, X_test, y_train, y_test)
    results.append(baseline_result)

    # Random Forest
    rf_result = random_forest_model(X_train, X_test, y_train, y_test)
    results.append(rf_result)

    # SVR
    svr_result = svr_model(X_train, X_test, y_train, y_test)
    results.append(svr_result)

    # -------------------------
    # Comparison Output
    # -------------------------
    print("\n=== Model Comparison Table ===")
    df_results = pd.DataFrame(results)
    print(df_results)


if __name__ == "__main__":
    main()