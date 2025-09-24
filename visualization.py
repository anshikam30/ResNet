import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter

from PlainNet import *  # Ensure this file defines PlainNet(n)
from ResNet import *     # Ensure this file defines ResNet(n)
from cnn import Net

def load_plainnet_model(depth, model_path):
    n = {20: 3, 56: 9, 110: 18}[depth]
    model = PlainNet(n)
    model = model.to('cuda')
    checkpoint = torch.load(model_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def load_resnet_model(depth, model_path):
    n = {20: 3, 56: 9, 110: 18}[depth]
    model = ResNet(n)
    model = model.to('cuda')
    checkpoint = torch.load(model_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def load_cnn_model(model_path):
    model = Net().to('cuda')
    checkpoint = torch.load(model_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def get_data_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def generate_random_direction(model):
    return [torch.randn_like(p) for p in model.parameters()]


def normalize_direction(direction, model):
    norm_dir = []
    for (name, param), d in zip(model.named_parameters(), direction):
        if 'weight' in name and len(param.size()) >= 2:
            reshaped = d.view(param.size(0), -1)
            reshaped = reshaped / (reshaped.norm(dim=1, keepdim=True) + 1e-10)
            norm_dir.append(reshaped.view_as(d))
        else:
            norm_dir.append(d / (d.norm() + 1e-10))
    return norm_dir


def get_loss_surface(model, data_loader, dx, dy, alpha_range, beta_range, criterion):
    model.eval()
    original_state = {name: p.clone() for name, p in model.named_parameters()}
    surface = []
    images, labels = next(iter(data_loader))
    images,labels = images.to('cuda'),labels.to('cuda')
    for alpha in alpha_range:
        row = []
        for beta in beta_range:
            with torch.no_grad():
                for (name, p), dxi, dyi in zip(model.named_parameters(), dx, dy):
                    p.copy_(original_state[name] + alpha * dxi + beta * dyi)
                outputs = model(images)
                loss = criterion(outputs, labels)
                row.append(loss.item())
        surface.append(row)

    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(original_state[name])

    return np.array(surface)


def plot_contour(surface, alpha_range, beta_range, title):
    smoothed = gaussian_filter(surface, sigma=1.0)
    log_loss = np.log(smoothed + 1e-5)
    X, Y = np.meshgrid(alpha_range, beta_range)
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, log_loss, levels=50, cmap='coolwarm')
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.tight_layout()
    plt.savefig(f"plots/{title.replace(' ', '_')}_contour.png")
    plt.close()


def plot_3d_surface(surface, alpha_range, beta_range, title):
    smoothed = gaussian_filter(surface, sigma=1.0)
    log_loss = np.log(smoothed + 1e-5)
    A, B = np.meshgrid(alpha_range, beta_range)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(A, B, log_loss, cmap='coolwarm', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Log Loss')
    ax.view_init(elev=35, azim=45)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(f"plots/{title.replace(' ', '_')}_3D.png")
    plt.close()


def process_model(model, name, data_loader, criterion, alpha_range, beta_range):
    dx = normalize_direction(generate_random_direction(model), model)
    dy = normalize_direction(generate_random_direction(model), model)
    surface = get_loss_surface(model, data_loader, dx, dy, alpha_range, beta_range, criterion)
    plot_contour(surface, alpha_range, beta_range, title=f"{name} Loss Contour")
    plot_3d_surface(surface, alpha_range, beta_range, title=f"{name} 3D Loss Landscape")
    return surface


def main():
    model_paths = {
        'ResNet-20': ("resnet", 20, "models/resnet20.pth"),
        'ResNet-56': ("resnet", 56, "models/resnet56.pth"),
        'ResNet-110': ("resnet", 110, "models/resnet110.pth"),
        'PlainNet-20': ("plainnet", 20, "models/plainnet20.pth"),
        'PlainNet-56': ("plainnet", 56, "models/plainnet56.pth"),
        'PlainNet-110': ("plainnet", 110, "models/plainnet110.pth"),
        'CNN': ("cnn", None, "models/cnn_model.pth")
    }

    data_loader = get_data_loader()
    criterion = nn.CrossEntropyLoss()
    alpha_range = np.linspace(-1, 1, 41)  
    beta_range = np.linspace(-1, 1, 41)

    for name, (arch, depth, path) in model_paths.items():
        print(f"Processing {name}...")
        if arch == "plainnet":
            model = load_plainnet_model(depth, path)
        elif arch == "resnet":
            model = load_resnet_model(depth, path)
        elif arch == "cnn":
            model = load_cnn_model(path)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
    process_model(model, name, data_loader, criterion, alpha_range, beta_range)


if __name__ == '__main__':
    main()
