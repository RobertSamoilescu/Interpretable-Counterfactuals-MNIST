import torch
import torch.nn.functional as F

import models


def Lpred(x: torch.tensor, delta: torch.tensor, t0: int, clf: models.Classifier):
    """
    This function computes the Lpred loss

    Parameters
    ----------
    x
        original input image
    delta
        noise added
    t0
        the original predicted class
    clf
        trained classifier

    Returns
    -------
    The prediction loss
    """
    ctf = torch.clamp(x + delta, min=-1, max=1)
    y = F.softmax(clf(ctf), dim=1)

    # compute the fpred score for the original class
    fpred_t0 = y[0, t0]


    # compute the max fpred score for the rest
    # of the classes
    y_rest = torch.cat((y[:, :t0], y[:, t0 + 1:]), dim=1)
    fpred_rest = torch.max(y_rest, dim=1)[0][0]

    # return the clipped output
    return torch.clamp(fpred_t0 - fpred_rest, min=0)


def L1(delta: torch.tensor):
    """
    Computes the L1 loss

    Parameters
    ----------
    delta
        noise added

    Returns
    -------
    The L1 loss
    """
    return torch.sum(torch.abs(delta))


def L2(delta: torch.tensor):
    """
    Computes the L2 loss

    Parameters
    ----------
    delta
        noise added

    Returns
    -------
    The L2 loss
    """
    return torch.sum(delta ** 2)


def LAE(x: torch.tensor, delta: torch.tensor, ae: models.AE):
    """
    Compute AutoEncoder loss

    Parameters
    ----------
    x
        original input image
    delta
        noise added
    ae:
        trained autoencoder

    Returns
    -------
    The AutoEncoder loss
    """
    # clamp the ctf in [-1, 1]
    ctf = torch.clamp(x + delta, min=-1, max=1)

    # pass ctf through the AE
    _, out_ae = ae(ctf)

    # reshape output and clamp it in [-1, 1]
    out_ae = out_ae.reshape(*x.shape)
    out_ae = torch.clamp(out_ae, min=-1, max=1)

    return L2(ctf - out_ae)


def Lproto(x: torch.tensor, delta: torch.tensor, proto: torch.tensor, ae: models.AE):
    """
    This function compute the prototype loss

    Parameters
    ----------
    x
        original input image
    delta
        added noise
    proto
        prototype for the target class
    ae
        trained auto-encoder

    Returns
    -------
    The Prototype loss
    """
    # clamp ctf to be in [-1, 1]
    ctf = torch.clamp(x + delta, min=-1, max=1)

    # pass ctf to auto-encoder
    enc_out, dec_out = ae(ctf)
    return L2(enc_out - proto)
