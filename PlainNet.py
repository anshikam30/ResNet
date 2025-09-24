import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from train_utils import *


class PlainBlock(nn.Module):

    def __init__(self, in_chann, chann, stride):
        super(PlainBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, padding=1, stride=stride)
        self.bn1   = nn.BatchNorm2d(chann)
        
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(chann)
        self.dropout = nn.Dropout2d(0.2)  #

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = nn.functional.relu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = nn.functional.relu(y)
        return y
    
class BaseNet(nn.Module):
    
    def __init__(self, Block, n):
        super(BaseNet, self).__init__()
        self.Block = Block
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn0   = nn.BatchNorm2d(16)
        self.convs  = self._make_layers(n)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = nn.functional.relu(x)
        x = self.convs(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x

    def _make_layers(self, n):
        layers = []
        in_chann = 16
        chann = 16
        stride = 1
        for i in range(3):
            for j in range(n):
                if ((i > 0) and (j == 0)):
                    in_chann = chann
                    chann = chann * 2
                    stride = 2

                layers += [self.Block(in_chann, chann, stride)]

                stride = 1
                in_chann = chann

        return nn.Sequential(*layers)
    
def PlainNet(n):
    return BaseNet(PlainBlock, n)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def train_plainnet(depth: int):
    # Depth to n mapping: (20->3, 56->9, 110->18)
    n = {20:3, 56:9, 110:18}[depth]
    device = get_device() 
    # Load data with enhanced augmentation
    train_loader, test_loader = get_data('cifar10', batch_size=100)
    
    # Create model
    model = PlainNet(n).to(device)  
    initialize_weights(model)
    print(f"PlainNet-{depth} Parameter Count: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), 
                           lr=1e-4, 
                           weight_decay=5e-4)
    
    # Use cosine annealing with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-3,
        epochs=200,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    metrics = train(model, train_loader, optimizer, scheduler,epochs=150)
    # Final evaluation

    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")
    # Save final model
    torch.save({
        'metrics': metrics,
        'model_state': model.state_dict()
    }, f"models/plainnet{depth}.pth")

    return metrics


def main() -> None:
    # Train all three architectures
    all_metrics = {}
    for depth in [20, 56, 110]:
        print(f"\n{'='*40}")
        print(f"Training PlainNet-{depth}")
        print(f"{'='*40}\n")
        metrics = train_plainnet(depth)
        all_metrics[depth] = metrics

    plot_combined_metrics(all_metrics, "results/plainnet_all_loss.png", "results/plainnet_all_acc.png")
