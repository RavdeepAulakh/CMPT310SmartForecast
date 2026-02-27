from src.data_preprocessing import load_data, split_data
from src.eda import plot_score_distribution, plot_correlations, scatter_relationships
from src.baseline_model import train_baseline


def main():
    df = load_data("data/exams.csv")

    # EDA
    plot_score_distribution(df)
    plot_correlations(df)
    scatter_relationships(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Train baseline regression model
    train_baseline(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()