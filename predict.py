import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from model import get_values


def plot_data(
    title, actual, predicted=None, actual_label="Actual", predicted_label="Predicted"
):
    """Helper function to plot actual vs predicted data."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label=actual_label, color="b")
    if predicted is not None:
        plt.plot(predicted, label=predicted_label, color="r", linestyle="--")
    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def index():
    print("========== Get Data From Database ==========")
    data = get_values(1)
    data = np.array([item[2] for item in data])

    # Plot original data
    plot_data("Original Data Plot", data)

    # Split data into training and test sets
    split_index = len(data) // 2
    train = data[:split_index]
    train_plus_1 = data[1 : split_index + 1]
    test = data[split_index:]

    # Train MLPRegressor model
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    model = mlp.fit(train.reshape(-1, 1), train_plus_1)

    # Predict training data
    pred_train = model.predict(train.reshape(-1, 1))
    plot_data("Train Data: Actual vs Predicted", train, pred_train)

    # Predict test data
    pred_test = model.predict(test.reshape(-1, 1))
    plot_data("Test Data: Actual vs Predicted", test, pred_test)

    # Predict entire data for overall comparison
    recurrent = model.predict(data.reshape(-1, 1))
    plot_data("Overall: Actual vs Predicted", data, recurrent)


if __name__ == "__main__":
    index()
