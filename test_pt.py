from tqdm import tqdm

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torch import optim, no_grad
import torch.nn as nn

from arg_utils import get_args


def define_model():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(16, 32, 5, 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 10),
    )


def train_and_test(model, optimizer, loss_function, epoch, datasets, device):
    model.to(device).train()
    for _ in tqdm(range(epoch), leave=True):
        for images, labels in tqdm(datasets['train'], leave=False):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(datasets['test']):
            total += len(images)
            test_output = model(images.to(device))
            pred_y = test_output.argmax(axis=1).to('cpu').numpy()
            correct += (pred_y == labels.numpy()).sum()
        print(correct, '/', total)


def main():
    args = get_args()
    train_data = MNIST(root='.', train=True, transform=ToTensor(), download=True)
    test_data = MNIST(root='.', train=False, transform=ToTensor(), download=True)
    loaders = {
        'train': DataLoader(train_data, batch_size=args.batch, shuffle=True),
        'test': DataLoader(test_data, batch_size=args.batch, shuffle=False),
    }
    model = define_model().to(args.device)
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()
    train_and_test(model, optimizer, loss, args.epoch, loaders, args.device)


if __name__ == '__main__':
    main()
