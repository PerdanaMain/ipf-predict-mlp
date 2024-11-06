from model import *
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib untuk plotting
from sklearn.neural_network import MLPRegressor  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore


def create_dataset(data, window_size=3):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def index():
    print("========== Get Data From Database ==========")
    data = get_values(1)
    print("============================================")

    data = np.array([item[2] for item in data])  # Mengambil kolom nilai saja
    X, y = create_dataset(data, window_size=3)

    print("============== Training Started ============")

    # Membagi data menjadi training dan testing
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Membuat dan melatih model MLPRegressor
    mod = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    mod.fit(X_train, y_train)

    # Melakukan prediksi pada data testing
    y_pred = mod.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Predicted values: {y_pred}")

    # Plot hasil prediksi vs nilai aktual
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(y_test)), y_test, label="Actual", color="b")
    plt.plot(range(len(y_test)), y_pred, label="Predicted", color="r", linestyle="--")
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    index()
