from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import torch.nn as nn
from loguru import logger
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor

from utils import predict_on_dataset

# from model import undetected_failures, detected_failures, values, detection_ranks, df, data_scaler, model, mean, final_df, sum_chunk, sum_chunk_non
# from plots_funcs import save_figures

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


def soft_acc(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return torch.mean(torch.eq(torch.round(y_true), torch.round(y_pred)))


def mse_and_r2(
    values: DataFrame, mse_list: list, r2_list: list, start_time: datetime
) -> Tuple[DataFrame, list, list, timedelta]:
    values["mse_list"] = np.mean(mse_list)
    logger.info("Average Mean Squared Error: %.6f" % np.mean(mse_list))
    if len(mse_list) > 1:
        logger.info("Standard Deviation of mse: %.6f" % stdev(mse_list))

    values["r2_list"] = np.mean(r2_list)

    logger.info("Average r2 Score: %.6f" % np.mean(r2_list))
    if len(r2_list) > 1:
        logger.info("Standard Deviation of r2 Score: %.6f" % stdev(r2_list))
    preparation_time = datetime.now() - start_time

    return values, mse_list, r2_list, preparation_time


def apfd_calc(
    df: DataFrame, values: DataFrame, undetected_failures: int, detected_failures: int, detection_ranks: list
) -> float:
    _, n_faults = df["Verdict"].value_counts()
    n = df.shape[0]

    values["Number of Failed Tests"] = n_faults
    values["Total Test Cases"] = n

    p = float(0)
    if undetected_failures > 0:
        p = detected_failures / n_faults
    else:
        p = 1
    logger.info(f"recall: {p}")

    if p == 1:
        apfd = p - sum(detection_ranks) / (n_faults * n) + p / (2 * n)
        logger.info(f"apfd: {apfd}")
        logger.info("napfd equals apfd")
    else:
        napfd = p - sum(detection_ranks) / (n_faults * n) + p / (2 * n)
        logger.info(f"napfd: {napfd}")
    values["apfd"] = apfd

    return apfd


def napfd_calc(
    df: DataFrame,
    data_scaler: sklearn.preprocessing._data.MinMaxScaler,
    model: nn.Sequential,
    values: DataFrame,
    napfd: list,
) -> list:
    # ## Calculating napfd per cycle

    missing = []
    last_cycle = df["Cycle"].iloc[-1]
    logger.info(f"{last_cycle} will be last cycle")
    i = 1
    prev_cycle = 0
    for index in range(1, last_cycle):
        if i <= last_cycle and i in df["Cycle"]:
            df_temp = df.loc[df["Cycle"] == (i)]
            i = i + 1
            if df_temp.empty:
                missing.append(i - 1)
                continue

            my_model = data_scaler.fit_transform(
                df_temp[["DurationFeature", "E1", "E2", "E3", "LastRunFeature", "DIST", "CHANGE_IN_STATUS"]]
            )
            my_model = torch.tensor(my_model, dtype=torch.float32)
            # my_model is a model variable
            predictions = predict_on_dataset(model, my_model)
            predictions_numpy = predictions.detach().cpu().numpy()

            df_temp.loc[:, "CalcPrio"] = predictions_numpy

            final_df_temp = df_temp.sort_values(by=["CalcPrio"], ascending=False)
            available_time = sum(final_df_temp["Duration"])

            counts = final_df_temp["Verdict"].value_counts().to_dict()

            n_faults = counts.get(1, 0)

            n = final_df_temp.shape[0]

            scheduled_time = 0
            detection_ranks = []

            undetected_failures = 0
            rank_counter = 1

            for index, row in final_df_temp.iterrows():
                if scheduled_time + row["Duration"] <= available_time:
                    if row["Verdict"] == 1:
                        detection_ranks.append(rank_counter)
                    scheduled_time += row["Duration"]
                    rank_counter += 1
                else:
                    undetected_failures = undetected_failures + 1
            detected_failures = len(detection_ranks)
            if n_faults > 0:
                p = float(0)
                if undetected_failures > 0:
                    p = detected_failures / n_faults
                else:
                    p = 1
                if n_faults == 1 and n == 1:
                    napfd.append(1)
                elif p == 1:
                    apfd = p - sum(detection_ranks) / (n_faults * n) + p / (2 * n)
                    # logger.info ("apfd: ",apfd)
                    napfd.append(apfd)
                else:
                    napfd = p - sum(detection_ranks) / (n_faults * n) + p / (2 * n)
                    napfd.append(napfd)
                    # logger.info ("napfd: ",napfd)
            else:
                napfd.append(0)

            # logger.info ("-----------------------------------------------------------------------------------")

    values["Avg napfd per cycle"] = mean(napfd)

    return napfd


def chunks(
    final_df: DataFrame, df: DataFrame, mean_chunk: list, mean_chunk_non: list, add_chunk: list, add_chunk_non: list
) -> Tuple[list, list, list, list]:
    # For pririotized
    chunks = np.array_split(final_df["Duration"], 10)

    for chunk in chunks:
        mean_chunk.append(mean(chunk))

    # For non-pririotized
    chunks_non = np.array_split(final_df["Duration"], 10)

    for chunk_non in chunks_non:
        mean_chunk_non.append(mean(chunk_non))

    for chunk in chunks:
        add_chunk.append(sum(chunk))

    for chunk_non in chunks_non:
        add_chunk_non.append(sum(chunk_non))

    return mean_chunk, mean_chunk_non, add_chunk, add_chunk_non


# For pririotized
# sum_chunk = []
def sum_chunks(sum_chunk: list, final_df: DataFrame) -> list:
    sum_chunks = np.array_split(final_df["Verdict"], 10)

    chunk_series = []
    for i in range(10):
        chunk_series.append(pd.Series(sum_chunks[i]))
        counts = chunk_series[i].value_counts().to_dict()
        sum_chunk.append(counts.get(1, 0))

    return sum_chunk


# For non-pririotized
# sum_chunk_non = []
def sum_chunks_non(sum_chunk_non: list, df: DataFrame) -> list:
    sum_chunks_non = np.array_split(df["Verdict"], 10)

    chunk_series_non = []
    for i in range(10):
        chunk_series_non.append(pd.Series(sum_chunks_non[i]))
        counts = chunk_series_non[i].value_counts().to_dict()
        sum_chunk_non.append(counts.get(1, 0))

    return sum_chunk_non


def AT_calc(final_df: DataFrame) -> Tuple[timedelta, timedelta, timedelta]:
    # For pririotized test cases
    start = datetime.now()

    rank = final_df.ranking[final_df["Verdict"] == 1].tolist()
    rank_counter = 1
    first_rank = False
    for index, row in final_df.iterrows():
        if row["Verdict"] == 1 and first_rank == False:
            FT = datetime.now() - start

            logger.info(rank_counter)
            first_rank = True

        if row["Verdict"] == 1 and rank[-1] == rank_counter:
            LT = datetime.now() - start

            logger.info(rank_counter)
        rank_counter += 1

    AT = (FT + LT) / 2

    logger.info(f"AT {AT} seconds : FT {FT} seconds : LT {LT} seconds")
    return FT, LT, AT


def AT_non_calc(df: DataFrame, final_df: DataFrame) -> Tuple[timedelta, timedelta, timedelta]:
    # For non-pririotized test cases
    start = datetime.now()

    rank = df.Id[final_df["Verdict"] == 1].tolist()
    rank_counter = 1
    first_rank = False
    for index, row in df.iterrows():
        if row["Verdict"] == 1 and first_rank == False:
            FT_non = datetime.now() - start

            logger.info(rank_counter)
            first_rank = True

        if row["Verdict"] == 1 and rank[-1] == rank_counter:
            LT_non = datetime.now() - start

            logger.info(rank_counter)
        rank_counter += 1

    AT_non = (FT_non + LT_non) / 2

    logger.info(f"AT {AT_non} seconds : FT {FT_non} seconds : LT {LT_non} seconds")
    return FT_non, LT_non, AT_non


def time_calc(
    FT: timedelta,
    LT: timedelta,
    AT: timedelta,
    FT_non: timedelta,
    LT_non: timedelta,
    AT_non: timedelta,
    start_time: datetime,
    preparation_time: timedelta,
    prioritization_time: timedelta,
    values: DataFrame,
) -> None:
    time = pd.DataFrame()
    time["Id"] = range(0, 1)
    time["env"] = "Paintcontrol"
    time["FT"] = FT
    time["LT"] = LT
    time["AT"] = AT
    time["FT non-prio"] = FT_non
    time["LT non-prio"] = LT_non
    time["AT non-prio"] = AT_non
    time.to_csv("time" + ".csv", sep=";", index=False)

    total_time = datetime.now() - start_time

    logger.info(f"Preparation Time : {preparation_time}")
    logger.info(f"Pririotization Time : {prioritization_time}")
    logger.info(f"Total Algorithm Time : {total_time}")
    values["Preparation Time"] = preparation_time
    values["Pririotization Time"] = prioritization_time
    values["Total Algorithm Time"] = total_time
