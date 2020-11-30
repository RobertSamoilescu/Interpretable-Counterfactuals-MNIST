import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim.optimizer import Optimizer


def fit(model: nn.Module, train_ds: Dataset, optimizer: Optimizer,
                     criterion: nn.Module, epochs: int = 4, batch_size: int = 32,
                     vis_inter: int = 500, tuple_flag: bool = True):
    """
    This function trains a classifier or an auto-encoder
    on the MNIST dataset

    Parameters
    ----------
    model
        Network to be trained
    train_ds
        Training dataset
    optimizer
        Network optimizer
    criterion
        Optimization criterion (objective)
    epochs
        Number of epochs to train the classifier
    batch_size
        Size of the batch
    vis_inter
        Visualization interval
    tuple_flag
        Flag to determine the appropriate target. For autoencoder
        the target is the same as input. For classification is
        the corresponding class
    Returns
    -------
    Trained classifier
    """

    # define the data loader
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    model = model.to(device)
    model.train()

    # define running loss
    r_loss = None

    for epoch in range(epochs):
        for i, data in enumerate(train_dl):
            # extract data and send it to device
            X, y = data
            X, y = X.to(device), y.to(device)

            if tuple_flag:
                # get the predictions
                y_pred = model(X)

                # compute loss
                loss = criterion(y_pred, y)
            else:
                # get the reconstruction
                enc_X, dec_X = model(X)

                # compute loss
                loss = criterion(dec_X, X)

            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute running loss
            r_loss = loss.item() if r_loss is None else 0.99 * r_loss + 0.01 * loss.item()

            if i % vis_inter == 0:
                print("Loss: %.4f, \tEpoch: %d \tIteration: %d" % (r_loss, epoch, i))

    return model
