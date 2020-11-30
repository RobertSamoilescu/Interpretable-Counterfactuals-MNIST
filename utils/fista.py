import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from copy import deepcopy


class FISTA(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.1):
        """
        Constructor

        Parameters
        ----------
        params
            Parameters to be learned
        lr
            Learning rate
        beta
            L1 penalty coefficient
        """

        defaults = dict(lr=lr, beta=beta)
        super(FISTA, self).__init__(params, defaults)

        # save previous delta parameters
        self.prev_delta = deepcopy(self.param_groups[0]["params"])

        # iteration counter
        self.k = 0

    def __shrinkage_thresholding(self, param: torch.tensor):
        """
        Shrinkage thresholding function

        Parameters
        ----------
        param
            Parameters for which we apply the shrinkage thresholding

        Returns
        -------
        Parameters after applying the shrinking thresholding
        """
        beta = self.defaults["beta"]
        param[param > beta].data = param[param > beta].data - beta
        param[param < -beta].data = param[param < -beta].data + beta
        param[torch.abs(param) <= beta].data = torch.zeros_like(param[torch.abs(param) <= beta].data)
        return param.data

    @torch.no_grad()
    def step(self, closure=None):
        """ Implementation of the FISTA optimization procedure """

        params = self.param_groups[0]["params"]
        lr = self.defaults["lr"]

        # computing new delta parameters
        for i in range(len(params)):
            if params[i].grad is None:
                continue

            params[i].data = params[i].data - lr * params[i].grad.data
            params[i].data = self.__shrinkage_thresholding(params[i])

        # update new delta
        delta = []
        for param in params:
            delta.append(param.clone())

        # compute the parameters from delta and prev delta (momentum)
        for i in range(len(params)):
            if params[i].grad is None:
                continue

            # momentum update
            params[i].data = delta[i].data + self.k / (self.k + 3) * (delta[i].data - self.prev_delta[i].data)

        # update old delta to point to the new one
        self.prev_delta = delta

        # increment steps
        self.k += 1
