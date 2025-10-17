
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass
from typing import Tuple

class SNN(nn.Module):
    def __init__(self, in_dim=784, hidden=64, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"

def make_loader(X, y, batch_size, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig):
    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = None
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            total += xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        train_acc = correct / total
        train_loss = loss_sum / total
        val_acc = evaluate(model, val_loader, device)
        history.append((epoch, train_loss, train_acc, val_acc))
    return history

def evaluate(model: nn.Module, loader: DataLoader, device=None) -> float:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / total
