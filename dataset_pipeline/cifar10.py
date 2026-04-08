from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
import torch


def get_datasets():

    train_transform = v2.Compose(
        [
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )

    test_dataset = CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    return train_dataset, test_dataset
