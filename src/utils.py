from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler


def split_dataset(X: DataFrame, Y: pd.Series) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # splitting dataset, 80% training and 20% testing
    X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2)

    # Scaling both training and testing input data.
    data_scaler = MinMaxScaler()
    X_train = data_scaler.fit_transform(X_train)
    X_test = data_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def data_to_tensor(data: Union[Any, np.ndarray]) -> torch.Tensor:
    if isinstance(data, pd.Series):
        data = data.values  # Convert pandas Series to numpy array
    return torch.tensor(data, dtype=torch.float32)


def save_output(df: DataFrame, napfd: list) -> None:
    # ## Saving results to output.csv for visualizations

    output = pd.DataFrame()
    output["step"] = df["Cycle"][1:].unique()
    output["env"] = "paintcontrol"
    output["model"] = "DeepOrder"
    output["napfd"] = napfd
    output.to_csv("output.csv", mode="a", sep=";", header=False, index=False)


def predict_on_dataset(model: Any, dataset: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        predictions = model(dataset)
    return predictions
