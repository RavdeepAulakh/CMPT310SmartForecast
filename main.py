from src.data_preprocessing import load_data, split_data
from src.eda import plot_score_distribution, plot_correlations, scatter_relationships
from src.baseline_model import train_baseline
from src.model_experiments import random_forest_model, svr_model
from src.final_model import select_best_model, retrain_best, final_evaluation, plot_predictions, plot_residuals

import pandas as pd


def main():
    df = load_data("data/exams.csv")

    # -------------------------
    # EDA
    # -------------------------
    plot_score_distribution(df)
    plot_correlations(df)
    scatter_relationships(df)

    # -------------------------
    # Split
    # -------------------------
    X_train, X_test, y_train, y_test = split_data(df)

    # -------------------------
    # Model Training
    # -------------------------
    results = []

    results.append(train_baseline(X_train, X_test, y_train, y_test))
    results.append(random_forest_model(X_train, X_test, y_train, y_test))
    results.append(svr_model(X_train, X_test, y_train, y_test))

    # -------------------------
    # Comparison Table
    # -------------------------
    print("\n=== Model Comparison Table ===")
    df_results = pd.DataFrame(results)
    print(df_results)

    # -------------------------
    # Final Model Selection
    # -------------------------
    best_model_name = select_best_model(results)

    final_model = retrain_best(best_model_name, X_train, y_train)

    y_pred = final_evaluation(final_model, X_test, y_test)

    # -------------------------
    # Final Visualizations
    # -------------------------
    plot_predictions(y_test, y_pred)
    plot_residuals(y_test, y_pred)


if __name__ == "__main__":
    main()
