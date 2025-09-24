import torch
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_data(dataset_name: str, batch_size: int = 32, transform: transforms = transforms.ToTensor(),) -> Tuple[DataLoader, DataLoader]:
    if dataset_name == 'cifar10':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, padding=4),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),])
        test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),])
        
        train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)
    else:
        raise ValueError('Dataset not supported')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader
