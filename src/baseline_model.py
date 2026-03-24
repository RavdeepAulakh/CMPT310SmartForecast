from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def train_baseline(X_train, X_test, y_train, y_test):

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nLinear Regression Results:")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R²:", r2)

    return {"model": "Linear Regression", "mse": mse, "rmse": rmse, "r2": r2}