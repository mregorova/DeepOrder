import os
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame

# ### Defining function to plot correlation between loss and epochs


def plot_loss(loss_values: list, val_loss_values: list) -> None:
    history_dict = {"loss": loss_values, "val_loss": val_loss_values}

    plt.plot(loss_values, "b--", label="training loss")  # Training data loss
    plt.plot(val_loss_values, "r", label="training loss val")  # Validation data loss
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.title("Loss Curve", fontsize=22)
    save_figures(plt, "loss_function_iofrol")
    plt.close()


# ###  Defining function to plot the comparision between 'Actual' and 'Predicted' value.


def actual_vs_prediction(y_test: Any, y_test_pred: Any) -> None:
    outcome = pd.DataFrame({"Actual": y_test, "Predicted": y_test_pred.flatten()})
    df_sorted = outcome.head(40).sort_values(by="Actual")

    df_sorted.plot(kind="bar", figsize=(12, 7))
    plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
    plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    plt.xlabel("Test Cases", fontsize=22)
    plt.ylabel("Priority Values", fontsize=22)
    plt.title("Comparision between 'Actual' and 'Predicted' values", fontsize=22)
    save_figures(plt, "actual_vs_prediction_iofrol")
    plt.close()

    plt.plot(df_sorted["Actual"].tolist(), label="Actual")
    plt.plot(df_sorted["Predicted"].tolist(), label="prediction")
    plt.xlabel("Test cases", fontsize=22)
    plt.ylabel("Priority Values", fontsize=22)
    plt.title("Comparision between 'Actual' and 'Predicted' values", fontsize=22)
    plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
    plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    plt.legend()
    save_figures(plt, "actual_vs_prediction_2_iofrol")
    plt.close()


# ###  Defining function to test the model


def save_figures(fig: Any, filename: str) -> None:
    FIGURE_DIR = os.path.abspath(os.getcwd())
    fig.savefig(os.path.join(FIGURE_DIR + "/src/Iofrol", filename + ".pdf"), bbox_inches="tight")


# ###  Defining function to plot the regression line for the model


def regression_line(y_test: Any, y_test_pred: Any) -> None:
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_test_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=4)
    ax.set_xlabel("Calculated by DeepOrder algorithm", fontsize=22)
    ax.set_ylabel("Predicted by Neural Network", fontsize=22)
    plt.title("Neural Network Regression Line", fontsize=22)
    plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
    plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    save_figures(plt, "regression_line_iofrol")
    plt.close()


def napfd_plot(Cycle: np.ndarray, napfd: list) -> None:
    plt.plot(Cycle[1:], napfd, color="red", marker="o")
    plt.title("napfd over cycles paintcontrol", fontsize=14)
    plt.xlabel("Cycles", fontsize=24)
    plt.ylabel("napfd", fontsize=24)
    plt.grid(True)
    plt.gcf().set_size_inches(25, 10)
    save_figures(plt, "napfd_per_cycle_paintcontrol")
    plt.close()


def visualization(final_df: DataFrame, df: DataFrame) -> None:
    # Plotting Prioritized test cases class distribution
    sns.FacetGrid(final_df, hue="Verdict", height=6).map(sns.kdeplot, "ranking")
    plt.legend(title="Verdict", loc="upper right", labels=["Passed", "Failed"])
    save_figures(plt, "Prioritized_test_cases_class_distribution_paintcontrol")
    plt.close()
    # Plotting Non-Prioritized test cases class distribution
    sns.FacetGrid(df, hue="Verdict", height=6).map(sns.kdeplot, "Id")
    plt.legend(title="Verdict", loc="upper right", labels=["Passed", "Failed"])
    save_figures(plt, "Non_Prioritized_test_cases_class_distribution_paintcontrol")
    plt.close()

    # For ['Verdict'] based binary visualization
    plt.plot(df["Verdict"].tolist(), label="Actual")
    plt.plot(final_df["Verdict"].tolist(), label="Prediction")
    plt.xlabel("Test cases")
    plt.ylabel("Time")
    plt.title("Comparision between 'Actual' and 'Predicted' values")
    plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
    plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    plt.legend(fontsize=16, loc="upper right")
    plt.gcf().set_size_inches(20, 10)
    save_figures(plt, "binary_visualization_verdict_paintcontrol")
    plt.close()

    # For ['Duration'] based visualization
    plt.plot(df["Duration"].tolist(), label="Actual")
    plt.plot(final_df["Duration"].tolist(), label="Prediction")
    plt.xlabel("Test cases", fontsize=22)
    plt.ylabel("Time", fontsize=22)
    plt.title("Comparision between 'Actual' and 'Predicted' values w.r.t Execution time", fontsize=22)
    plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
    plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    plt.legend(fontsize=16)
    plt.gcf().set_size_inches(20, 10)
    save_figures(plt, "Test_execution_time_overall_paintcontrol")
    plt.close()


def fault_detection(sum_chunk: list, sum_chunk_non: list) -> None:
    # Visualizing frequency of fault detection cases per 10 intervals
    length = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Year =  range(0, length)
    newList = [x / max(sum_chunk) for x in sum_chunk]
    newList2 = [x / max(sum_chunk_non) for x in sum_chunk_non]

    plt.plot(length, sum_chunk, color="green", marker="o", label="Prioritized")
    plt.plot(length, sum_chunk_non, color="red", marker="o", label="Actual")

    plt.title("Frequency of Failed test cases in paintcontrol", fontsize=14)
    plt.xlabel("Fault test cases", fontsize=18)
    plt.ylabel("Verdict Frequency", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=16)

    plt.gcf().set_size_inches(15, 8)
    save_figures(plt, "frequency_of_fault_detection_per_interval_paintcontrol")
    plt.close()


def average_time(mean_chunk: list, mean_chunk_non: list) -> None:
    # Visualizing average time of all test cases per 10 intervals
    length = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Year =  range(0, length)
    newList = [x / max(mean_chunk) for x in mean_chunk]
    newList2 = [x / max(mean_chunk_non) for x in mean_chunk_non]
    plt.plot(length, mean_chunk, color="green", marker="o", label="Prioritized")
    plt.plot(length, mean_chunk_non, color="red", marker="o", label="Non-Prioritized")
    plt.legend()
    plt.title("time over test cases paintcontrol", fontsize=14)
    plt.xlabel("test cases", fontsize=18)
    plt.ylabel("time (average per interval))", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=16)
    save_figures(plt, "average_time_of_all_tests_per_interval_paintcontrol")
    plt.close()


def summed_time(add_chunk: list, add_chunk_non: list) -> None:
    # Visualizing summed time of all test cases per 10 intervals
    length = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Year =  range(0, length)
    newList = [x / max(add_chunk) for x in add_chunk]
    newList2 = [x / max(add_chunk_non) for x in add_chunk_non]
    plt.plot(length, add_chunk, color="green", marker="o", label="Prioritized")
    plt.plot(length, add_chunk_non, color="red", marker="o", label="Non-Prioritized")

    plt.title("time over test cases paintcontrol", fontsize=14)
    plt.xlabel("test cases", fontsize=18)
    plt.ylabel("time (sum per interval))", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=16)

    plt.gcf().set_size_inches(15, 8)
    save_figures(plt, "total_time_of_all_tests_per_interval_paintcontrol")
    plt.close()
