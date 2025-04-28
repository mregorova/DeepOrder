from typing import Any, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from pandas import DataFrame
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor

from validation import validation


def train(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    loss_values: list,
    lr: float,
    num_epochs: int,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    val_loss_values: list,
    mse_list: list,
    r2_list: list,
    model: Any,
    optimizer: torch.optim.Adam,
    criterion: torch.nn.modules.loss.MSELoss,
    metrics: list,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    list,
    nn.Sequential,
    torch.nn.modules.loss.MSELoss,
    torch.Tensor,
    int,
    list,
    list,
    torch.Tensor,
    torch.Tensor,
    list,
]:
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(X_train)
        outputs = outputs.squeeze()

        loss = criterion(outputs, y_train).squeeze()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

        model, mse_list, r2_list, y_test_pred, val_loss, val_loss_values = validation(
            model, X_test, y_test, criterion, val_loss_values, epoch, loss, mse_list, r2_list
        )

        print(y_test_pred)

    return (
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
    )
