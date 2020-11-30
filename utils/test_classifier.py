import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple


def test_classifier(classifier: nn.Module, dataset: Dataset, batch_size: int = 32) -> Tuple[float, np.array]:
    """
    This function computes the accuracy on the test dataset

    Parameters
    ----------
    classifier
        trained classifier
    dataset
        dataset to compute the accuracy for
    batch_size
        size of the batch

    Returns
    -------
    Accuracy on the test set
    """

    # define data loader
    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set model in evaluation mode
    classifier = classifier.to(device)
    classifier.eval()

    # define  accuracy
    accuracy = 0
    predictions = []

    with torch.no_grad():
        for i, data in enumerate(test_dl):
            # unpack and send data to device
            X, y = data
            X, y = X.to(device), y.to(device)

            # predict the class
            y_pred = classifier(X)

            # gather predictions
            y_pred = torch.argmax(y_pred, dim=1)
            accuracy += torch.sum(y_pred == y).item()
            predictions += list(y_pred.cpu().numpy())

    return accuracy / len(dataset), np.array(predictions)