import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def select_best_model(results):
    # Select model with highest R²
    best = max(results, key=lambda x: x["r2"])
    print("\nBest Model Selected:", best["model"])
    return best["model"]


def retrain_best(model_name, X_train, y_train):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_name == "Linear Regression":
        model = LinearRegression()

    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    elif model_name == "SVR":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(C=1, epsilon=0.1, kernel="rbf"))
        ])

    model.fit(X_train, y_train)
    return model


def final_evaluation(model, X_test, y_test):

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Final Model Performance ===")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R²:", r2)

    return y_pred


# -------------------------
# FINAL VISUALIZATIONS
# -------------------------

def plot_predictions(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Scores")
    plt.ylabel("Predicted Scores")
    plt.title("Actual vs Predicted Scores")

    # ideal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.show()


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred

    plt.hist(residuals, bins=20)
    plt.title("Residual Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()
