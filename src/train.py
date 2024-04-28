# https://github.com/pytorch/examples/blob/main/mnist/main.py

import argparse
import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

# For logging experiment progress
import tensorboard

# Define your own import for model
from model import vit
from model_config import config


def get_model() -> nn.Module:
    """Define your own implementation of model here

    Returns:
        nn.Module: Model to be used for training
    """

    model = vit(**config)
    return model


def train(model, device, train_loader, optimizer, epoch, loss_fn, log_every, writer):
    model.train()
    num_batches = len(train_loader)
    correct = 0
    cumulative_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        cumulative_loss += loss.item()

        loss.backward()
        optimizer.step()
        # Print the total batches encountered in this epoch, the accuracy of the epoch, loss of current batch and cumulative loss in this batch
        if (1 + batch_idx) % log_every == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tCurrent Batch Loss: {:.6f}\tCumulative Loss: {:.6f}".format(
                    epoch, batch_idx *
                    len(data), len(train_loader.dataset), 100.0 *
                    batch_idx / len(train_loader), loss.item(), cumulative_loss / batch_idx
                )
            )

            writer.add_scalar(
                "batch.train.loss", loss, batch_idx + epoch * num_batches
            )

    writer.add_scalar(
        "epoch.train.accuracy", 100.0 * correct /
        len(train_loader.dataset), epoch
    )

    writer.add_scalar(
        "epoch.train.loss", cumulative_loss / len(train_loader.dataset), epoch
    )

    return cumulative_loss / len(train_loader.dataset), 100.0 * correct / len(train_loader.dataset)


def test(model, device, test_loader, loss_fn, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    writer.add_scalar("epoch.test.loss", test_loss, epoch)
    writer.add_scalar("epoch.test.accuracy", 100.0 *
                      correct / len(test_loader.dataset), epoch)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(
                test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

    return test_loss, 100.0 * correct / len(test_loader.dataset)


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Training Classifier with ViT")

    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size",
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="Epochs to train for",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning Rate (default: 0.01)",
    )

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    parser.add_argument(
        "--log-interval", type=int, default=50, help="How many batches to accumulate and log",
    )
    parser.add_argument(
        "--save-model", action="store_true", default=True, help="When to save the model",
    )

    args = parser.parse_args()
    return args


def main():

    # Define arguments for training
    args = get_args()

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Define train and test datasets
    dataset1 = datasets.MNIST("../data", train=True,
                              download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)

    # Create a writer to log all the metrics
    writer = SummaryWriter(log_dir="../logs")

    # Add all the hyperparameters
    hyper_parameters = {k: str(v) for k, v in vars(args).items()}
    for k, v in config.items():
        hyper_parameters[f"model.{k}"] = str(v)
    
    # Create dataloaders for train and test datasets
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    loss_fn = nn.CrossEntropyLoss()

    # Create a model and the optimizer to train with
    model = get_model()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    metric_dict = {}
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer,
                                      epoch, loss_fn, args.log_interval, writer)
        test_loss, test_acc = test(
            model, device, test_loader, loss_fn, epoch, writer)
        
    metric_dict = {"final_train_loss": train_loss, "final_train_accuracy": train_acc,
                   "final_test_loss": test_loss, "final_test_accuracy": test_acc}

    writer.add_hparams(hparam_dict=hyper_parameters, metric_dict=metric_dict)

    if args.save_model:
        torch.save(model.state_dict(), "../models/vit_mnist.pt")

    writer.close()


if __name__ == "__main__":
    main()
