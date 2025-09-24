import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import get_device
from typing import Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_epoch(model: nn.Module, data_loader: DataLoader, optimizer: optim.Optimizer,) -> Tuple[float, float]:
    model.train()
    device = get_device()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        total_correct += (output.argmax(1) == target).sum().item()
        total_samples += data.size(0)
    return total_loss / total_samples, total_correct / total_samples


def evaluate(model: nn.Module, data_loader: DataLoader,) -> Tuple[float, float]:
    model.eval()
    device = get_device()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item() * data.size(0)
            total_correct += (output.argmax(1) == target).sum().item()
            total_samples += data.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train(model: nn.Module, data_loader: DataLoader, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, epochs: int = 10,) -> dict:
    print("Training...")
    model.to(get_device())
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'lr_history': []
    }
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, data_loader, optimizer)
        # Update metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['lr_history'].append(scheduler.get_last_lr()[0])
        scheduler.step()  
        print(f"Epoch {epoch + 1} / {epochs} | " + f"Train Loss: {train_loss:.4f} | " + f"Train Acc: {train_acc:.4f}")
    print("Training complete!")
    return metrics



def plot_metrics(metrics: dict, filename: str):
    epochs = range(len(metrics['train_loss']))
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot loss on primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, metrics['train_loss'], color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot accuracy on secondary y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, metrics['train_acc'], color=color, label='Train Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Legends and layout
    fig.tight_layout()
    plt.title("Training Loss and Accuracy")
    plt.savefig(filename)
    plt.close()

import matplotlib.pyplot as plt

def plot_combined_metrics(all_metrics: dict, loss_filename: str, acc_filename: str):
    plt.figure(figsize=(10, 5))
    for depth, metrics in all_metrics.items():
        plt.plot(metrics['train_loss'], label=f"PlainNet-{depth}")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_filename)
    plt.close()

    plt.figure(figsize=(10, 5))
    for depth, metrics in all_metrics.items():
        plt.plot(metrics['train_acc'], label=f"PlainNet-{depth}")
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(acc_filename)
    plt.close()

