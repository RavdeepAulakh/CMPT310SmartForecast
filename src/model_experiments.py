import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R²:", r2)

    return {"model": name, "mse": mse, "rmse": rmse, "r2": r2}


# -------------------------
# RANDOM FOREST
# -------------------------
def random_forest_model(X_train, X_test, y_train, y_test):

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    }

    rf = RandomForestRegressor(random_state=42)

    grid = GridSearchCV(rf, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("\nBest Random Forest Params:", grid.best_params_)

    return evaluate_model("Random Forest", best_model, X_test, y_test)


# -------------------------
# SUPPORT VECTOR REGRESSION
# -------------------------
def svr_model(X_train, X_test, y_train, y_test):

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])

    param_grid = {
        "svr__C": [0.1, 1, 10],
        "svr__epsilon": [0.1, 0.5, 1],
        "svr__kernel": ["linear", "rbf"]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("\nBest SVR Params:", grid.best_params_)

    return evaluate_model("SVR", best_model, X_test, y_test)