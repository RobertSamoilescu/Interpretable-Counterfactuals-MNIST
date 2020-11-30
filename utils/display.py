import torch
import numpy as np
import matplotlib.pyplot as plt


def show(img: torch.tensor):
    """
    Display the received image

    Parameters
    ---------
    img
        image to be displayed

    Returns
    -------
    None
    """

    # transform to numpy
    npimg = img.numpy()

    # plot image
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()