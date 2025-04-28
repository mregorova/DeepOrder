from typing import Any, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score


def validation(
    model: Any,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    criterion: torch.nn.modules.loss.MSELoss,
    val_loss_values: list,
    epoch: int,
    loss: torch.Tensor,
    mse_list: list,
    r2_list: list,
) -> Tuple[nn.Sequential, list, list, torch.Tensor, torch.Tensor, list]:
    # Validation
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        y_test_pred = y_test_pred.squeeze()
        val_loss = criterion(y_test_pred, y_test)
        val_loss_values.append(val_loss.item())

        print(f"Epoch: {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    mse_list.append(mean_squared_error(y_test, y_test_pred.numpy()))
    r2_list.append(r2_score(y_test, y_test_pred.numpy()))

    outcome = pd.DataFrame({"Actual": y_test, "Predicted": y_test_pred.flatten()})

    return model, mse_list, r2_list, y_test_pred, val_loss, val_loss_values
