import utils
import models

import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid


def display_ae(ae: models.AE, test_ds: Dataset, batch_size: int = 4):
    """
    Displays the reconstruction from the test set

    Parameters
    ----------
    ae
        Trained auto-encoder
    test_ds
        Test dataset (MNIST)
    batch_size
        Size of the batch to be displayed
    """

    # define data loader
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define iterator over test data
    test_iter = iter(test_dl)
    data = next(test_iter)

    # unpack data and send it to device
    X, y = data
    X, y = X.to(device), y.to(device)

    # set model for evaluation
    ae = ae.to(device)
    ae.eval()

    # get output of the auto-encoder
    with torch.no_grad():
        X_enc, X_dec = ae(X)

    # transform X_dec to match input shape
    X_dec = X_dec.reshape(*X.shape)

    # send data back to cpu and display
    X, X_dec = X.cpu(), X_dec.cpu()
    X_dec = torch.clamp(X_dec, min=-1., max=1.)
    all_X = torch.cat((X, X_dec), dim=0)

    # display results
    utils.show(make_grid(all_X, nrow=batch_size, normalize=True))