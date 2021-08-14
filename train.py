import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import (
    MEDataset,
    get_loader,
    LOSO_sequence_generate
)
from read_file import read_csv
from FMER import FMER


def train(epochs: int, criterion: nn.Module, optimizer: torch.optim,
          model: nn.Module, scheduler: torch.optim.lr_scheduler, train_loader: DataLoader,
          device: torch.device, model_best_name: str):
    """Train the model

    Parameters
    ----------
    epochs : int
        Epochs for training the model
    model : DSSN
        Model to be trained
    train_loader : DataLoader
        DataLoader to load in the data
    device: torch.device
        Device to be trained on
    model_best_name: str
        Name of the weight file to be saved
    """
    best_accuracy = -1
    # Set model in training mode
    model.train()

    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0

        for patches, labels in train_loader:
            patches = patches.to(device)
            labels = labels.to(device)

            output = model(patches)

            # Compute the loss
            loss = criterion(output, labels)
            train_loss += loss.item()

            # Update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the accuracy
            prediction = (output.argmax(-1) == labels)
            train_accuracy += prediction.sum().item() / labels.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        print(f"Epoch: {epoch + 1}")
        print(f"Loss: {train_loss}")
        print(f"Accuracy: {train_accuracy}")

        if train_accuracy > best_accuracy:
            torch.save(model.state_dict(), model_best_name)
            best_accuracy = train_accuracy
            print("Save model")


def evaluate(test_loader: DataLoader, model: nn.Module, device: torch.device):
    # Set into evaluation mode
    model.eval()
    test_accuracy = 0.0
    test_f1_score = 0.0

    with torch.no_grad():
        for patches, labels in test_loader:
            # Move data to device and compute the output
            patches = patches.to(device)
            labels = labels.to(device)

            output = model(patches)

            # Compute the accuracy
            prediction = (output.argmax(-1) == labels)
            test_accuracy += prediction.sum().item() / labels.size(0)
            test_f1_score += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(),
                                      average="weighted")

    return test_accuracy / len(test_loader), test_f1_score / len(test_loader)


def LOSO_train(data: pd.DataFrame, sub_column: str, args,
               label_mapping: dict, device: torch.device):
    log_file = open("train.log", "w")

    npz_file = np.load(args.npz_file)
    adj_matrix = torch.FloatTensor(npz_file["adj_matrix"]).to(device)

    # Create different DataFrame for each subject
    train_list, test_list = LOSO_sequence_generate(data, sub_column)
    test_accuracy = 0.0
    test_f1_score = 0.0

    for idx in range(len(train_list)):
        print(f"=================LOSO {idx + 1}=====================")
        train_csv = train_list[idx]
        test_csv = test_list[idx]

        # Create dataset and dataloader
        _, train_loader = get_loader(csv_file=train_csv,
                                     image_root=args.image_root,
                                     label_mapping=label_mapping,
                                     batch_size=args.batch_size,
                                     device=device,
                                     catego=args.catego)
        _, test_loader = get_loader(csv_file=test_csv,
                                    image_root=args.image_root,
                                    label_mapping=label_mapping,
                                    batch_size=len(test_csv),
                                    device=device,
                                    catego=args.catego,
                                    train=False)

        # Read in the model
        model = FMER(adj_matrix=adj_matrix,
                     num_classes=args.num_classes,
                     device=device).to(device)

        # Create criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)

        # Train the data
        train(epochs=args.epochs,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=None,
              model=model,
              train_loader=train_loader,
              device=device,
              model_best_name=f"{args.weight_save_path}/model_best_{idx}.pt")
        model.load_state_dict(torch.load(f"{args.weight_save_path}/model_best_{idx}.pt",
                                         map_location=device))

        temp_test_accuracy, temp_f1_score = evaluate(test_loader=test_loader,
                                                     model=model,
                                                     device=device)
        print(f"In LOSO {idx + 1}, test accuracy: {temp_test_accuracy:.4f}, f1-score: {temp_f1_score:.4f}")
        log_file.write(f"LOSO {idx + 1}: Accuracy: {temp_test_accuracy:.4f}, F1-Score: {temp_f1_score:.4f}\n")
        test_accuracy += temp_test_accuracy
        test_f1_score += temp_f1_score

    print(f"LOSO accuracy: {test_accuracy / len(train_list):.4f}, f1-score: {test_f1_score / len(train_list):.4f}")
    log_file.write(
        f"Total: Accuracy {test_accuracy / len(train_list):.4f}, F1-Score: {test_f1_score / len(train_list):.4f}\n")
    log_file.close()


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",
                        type=str,
                        required=True,
                        help="Path for the csv file for training data")
    parser.add_argument("--image_root",
                        type=str,
                        required=True,
                        help="Root for the training images")
    parser.add_argument("--npz_file",
                        type=str,
                        required=True,
                        help="Place for the npz file")
    parser.add_argument("--catego",
                        type=str,
                        required=True,
                        help="SAMM or CASME dataset")
    parser.add_argument("--num_classes",
                        type=int,
                        default=5,
                        help="Classes to be trained")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Training batch size")
    parser.add_argument("--weight_save_path",
                        type=str,
                        default="model",
                        help="Path for the saving weight")
    parser.add_argument("--epochs",
                        type=int,
                        default=15,
                        help="Epochs for training the model")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=3e-4,
                        help="Learning rate for training the model")
    args = parser.parse_args()

    # Training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read in the data
    data, label_mapping = read_csv(args.csv_path)

    # Create folders for the saving weight
    os.makedirs(args.weight_save_path, exist_ok=True)

    # Train the model
    LOSO_train(data=data,
               sub_column="Subject",
               label_mapping=label_mapping,
               args=args,
               device=device)
