import numpy as np
import matplotlib.pyplot as plt


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def bias_variance(y, y_pred):
    bias = np.mean((y - np.mean(y_pred)) ** 2)
    variance = np.var(y_pred)

    return bias, variance


def print_metrics(
    model_name,
    mse_train,
    mse_val,
    bias_train,
    bias_val,
    variance_train,
    variance_val,
    degree,
    lambda_values=None,
):
    print(f"\nMetrics for {model_name} Model")
    print("=" * 40)

    for i, d in enumerate(degree):
        if lambda_values is None:  # OLS case, no lambda values
            print(f"Degree {d}:")
            print(f"  Train MSE: {mse_train[i]:.4f}, Val MSE: {mse_val[i]:.4f}")
            print(f"  Train Bias: {bias_train[i]:.4f}, Val Bias: {bias_val[i]:.4f}")
            print(
                f"  Train Variance: {variance_train[i]:.4f}, Val Variance: {variance_val[i]:.4f}"
            )
        else:  # Ridge/Lasso case, iterate over lambda values
            for j, lamb in enumerate(lambda_values):
                print(f"Degree {d}, Lambda {lamb:.4f}:")
                print(
                    f"  Train MSE: {mse_train[i, j]:.4f}, Val MSE: {mse_val[i, j]:.4f}"
                )
                print(
                    f"  Train Bias: {bias_train[i, j]:.4f}, Val Bias: {bias_val[i, j]:.4f}"
                )
                print(
                    f"  Train Variance: {variance_train[i, j]:.4f}, Val Variance: {variance_val[i, j]:.4f}"
                )
        print()


def plot_metrics(title, degree, train, test, lambda_values=None):
    plt.figure(figsize=(10, 6))
    if lambda_values is None:  # OLS case
        plt.plot(degree, train, label=f"Train", marker="o")
        plt.plot(degree, test, label=f"Test", marker="o")
    else:  # Ridge/Lasso case, use 2D heatmap
        for i, lamb in enumerate(lambda_values):
            plt.plot(degree, train[:, i], label=f"Train (λ={lamb:.4f})", marker="o")
            plt.plot(degree, test[:, i], label=f"Test (λ={lamb:.4f})", marker="o")

    plt.title(title)
    plt.xlabel("Polynomial Degree")
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.show()
