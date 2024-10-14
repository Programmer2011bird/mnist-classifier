from torch import inference_mode
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


def train_step(optimizer: Optimizer, loss_fn, model: nn.Module, dataLoader: DataLoader):
    model.train()
    train_loss = 0

    for batch, (X, y) in enumerate(dataLoader):
        y_preds = model(X)

        loss = loss_fn(y_preds, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataLoader)

    return train_loss


def test_step(loss_fn, model: nn.Module, dataLoader: DataLoader):
    model.eval()
    loss = 0
    
    with inference_mode():
        for X, y in dataLoader:
            y_preds = model(X)

            loss += loss_fn(y_preds, y)

        loss /= len(dataLoader)

        return loss

