import argparse
import os
import warnings
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Any, Sized, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.utils.data import DataLoader, Subset, TensorDataset

from metrics import (
    AT_calc,
    AT_non_calc,
    apfd_calc,
    chunks,
    mse_and_r2,
    napfd_calc,
    soft_acc,
    sum_chunks,
    sum_chunks_non,
    time_calc,
)
from plots_funcs import (
    actual_vs_prediction,
    average_time,
    fault_detection,
    napfd_plot,
    plot_loss,
    regression_line,
    summed_time,
    visualization,
)
from train import train
from utils import data_to_tensor, predict_on_dataset, save_output, split_dataset


class MLPModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.Tanh())

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> torch.Tensor:
        return self.model(x)


def main(input_file: str, lr: float, num_epochs: int) -> None:
    model = MLPModel(input_size=7, output_size=1, hidden_sizes=[7, 10, 20, 15])

    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.MSELoss()
    metrics = [soft_acc]

    # warnings.simplefilter(action="ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True
    model = model.to(device)  # Move the model to the GPU

    start_time = datetime.now()

    # Reading the dataset
    # df = pd.read_csv("Datasets/iofrol_updated.csv")
    df = pd.read_csv(input_file, sep=",")
    values = pd.DataFrame()
    values["Id"] = range(0, 1)
    values["env"] = "Iofrol"

    X = df[["Duration", "E1", "E2", "E3", "LastRunFeature", "DIST", "CHANGE_IN_STATUS"]]  # Defining feature variable
    Y = df["PRIORITY_VALUE"]  # Defining label variable
    mse_list: list = []  # mean square error list
    r2_list: list = []
    loss_values: list = []
    val_loss_values: list = []

    X_train, X_test, y_train, y_test = split_dataset(X, Y)

    X_train, X_test, y_train, y_test = map(data_to_tensor, [X_train, X_test, y_train, y_test])

    (
        X_train,
        y_train,
        loss_values,
        model,
        criterion,
        loss,
        epoch,
        mse_list,
        r2_list,
        y_test_pred,
        val_loss,
        val_loss_values,
    ) = train(
        X_train,
        y_train,
        loss_values,
        lr,
        num_epochs,
        X_test,
        y_test,
        val_loss_values,
        mse_list,
        r2_list,
        model,
        optimizer,
        criterion,
        metrics,
    )

    # ---------------------#
    values, mse_list, r2_list, preparation_time = mse_and_r2(values, mse_list, r2_list, start_time)

    plot_loss(loss_values, val_loss_values)  # displaying performance based on loss

    actual_vs_prediction(y_test, y_test_pred)

    regression_line(y_test, y_test_pred)

    data_scaler = MinMaxScaler()

    start_prio_time = datetime.now()

    my_model = data_scaler.fit_transform(
        df[["DurationFeature", "E1", "E2", "E3", "LastRunFeature", "DIST", "CHANGE_IN_STATUS"]]
    )
    my_model = torch.tensor(my_model, dtype=torch.float32)
    # my_model is a model variable
    predictions = predict_on_dataset(model, my_model)
    predictions_numpy = predictions.detach().cpu().numpy()

    df.loc[:, "CalcPrio"] = predictions_numpy

    final_df = df.sort_values(by=["CalcPrio"], ascending=False)
    final_df["ranking"] = range(1, 1 + len(df))
    available_time = sum(final_df["Duration"])
    prioritization_time = datetime.now() - start_prio_time

    scheduled_time = 0
    detection_ranks = []
    undetected_failures = 0
    rank_counter = 1

    for index, row in final_df.iterrows():
        if scheduled_time + row["Duration"] <= available_time:
            if row["Verdict"] == 1:
                # print (rank_counter)

                detection_ranks.append(rank_counter)
            scheduled_time += row["Duration"]
            rank_counter += 1
        else:
            undetected_failures += row["Id"]

    detected_failures = len(detection_ranks)

    apfd = apfd_calc(df, values, undetected_failures, detected_failures, detection_ranks)
    napfd: list = []
    napfd_calc(df, data_scaler, model, values, napfd)
    Cycle = df["Cycle"].unique()

    napfd_plot(Cycle, napfd)
    save_output(df, napfd)

    # ## Calculating Time related metrics
    visualization(final_df, df)

    mean_chunk: list = []
    mean_chunk_non: list = []
    add_chunk: list = []
    add_chunk_non: list = []
    chunks(final_df, df, mean_chunk, mean_chunk_non, add_chunk, add_chunk_non)

    sum_chunk: list = []
    sum_chunks(sum_chunk, final_df)

    sum_chunk_non: list = []
    sum_chunks_non(sum_chunk_non, df)

    fault_detection(sum_chunk, sum_chunk_non)
    average_time(mean_chunk, mean_chunk_non)
    summed_time(add_chunk, add_chunk_non)

    # ----------------------------------------------------------------------------

    FT, LT, AT = AT_calc(final_df)

    FT_non, LT_non, AT_non = AT_non_calc(df, final_df)

    time_calc(FT, LT, AT, FT_non, LT_non, AT_non, start_time, preparation_time, prioritization_time, values)

    values.to_csv("values" + ".csv", sep=";", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the CSV file and perform data preprocessing.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")

    args = parser.parse_args()

    main(args.input_file, args.lr, args.num_epochs)
