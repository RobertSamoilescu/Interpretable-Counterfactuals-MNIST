import torch
from torch.utils.data import Dataset

import utils
import models

import numpy as np
from functools import reduce
from typing import Dict


class CounterFactualSearch(object):
    def __init__(self, dataset: Dataset, ae: models.AE, clf: models.Classifier, num_classes: int = 10):
        self.dataset = dataset
        self.num_classes = num_classes

        # define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # send to device
        self.ae = ae.to(self.device).eval()
        self.clf = clf.to(self.device).eval()

        # pass dataset through the classifier to get the predictions
        _, self.y_pred = utils.test_classifier(self.clf, self.dataset, batch_size=32)

        # compute the indices in the dataset for each class
        self.class_split = {}
        for i in range(self.num_classes):
            self.class_split[i] = np.where(self.y_pred == i)[0]

        # compute the encodings for each class
        self.encodings = {}
        for i in range(self.num_classes):
            sub_dataset = [self.dataset[x][0].clone() for x in self.class_split[i]]
            self.encodings[i] = []

            # this step might be inefficient
            # should be working with batches
            for img in sub_dataset:
                img = img.unsqueeze(0).to(self.device)
                enc_img, _ = self.ae(img)
                enc_img = enc_img.squeeze(0).detach().cpu().numpy()
                self.encodings[i].append(enc_img)

    def compute_prototypes(self, enc_img: np.array, t0: int, K: int):
        # compute K closest encoding from all the other classes
        # different than the t0 class
        prototypes = {}

        for i in range(self.num_classes):
            if i == t0:
                continue

            # compute differences in the encoding space
            diffs = [np.linalg.norm(x - enc_img) for x in self.encodings[i]]

            # get sorted list of indices
            sort_idx = sorted(range(len(diffs)), key=lambda x: diffs[x])

            # compute the prototype encoding
            closest_enc = [self.encodings[i][x] for x in sort_idx]
            proto = reduce(lambda x, y: x + y, closest_enc[:K]) / K

            # save the prototype
            prototypes[i] = proto

        return prototypes

    def closest_prototype(self, enc_img: np.array, prototypes: Dict):
        other_classes = list(prototypes.keys())
        prototype = other_classes[0]
        min_dist = np.linalg.norm(prototype - enc_img)

        for i in other_classes:
            dist = np.linalg.norm(prototypes[i] - enc_img)
            if dist < min_dist:
                prototype = prototypes[i]
                min_dist = dist

        return prototype

    def optimiziation(self, img: torch.tensor, protoype: torch.tensor,
                      t0: int,  beta: float, c: float, gamma: float,
                      theta: float, lr: float, num_iters: int, verbose: bool):
        # define noise
        delta = torch.zeros(img.shape).to(self.device)
        delta.requires_grad = True

        # define optimizer
        optimizer = utils.FISTA([delta], lr=lr, beta=beta)

        # training loop
        for i in range(num_iters):
            Lpred = utils.Lpred(img, delta, t0, self.clf)

            if Lpred == 0:
                break

            loss = c * Lpred
            loss += utils.L2(delta)
            loss += gamma * utils.LAE(img, delta, self.ae)
            loss += theta * utils.Lproto(img, delta, protoype, self.ae)

            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # project delta so that img + delta in [-1, 1]
            max_th, min_th = 1 - img, -1 - img
            delta.data = torch.min(delta, max_th)
            delta.data = torch.max(delta, min_th)

            if verbose and i % 100 == 0:
                print("Iteration: %5d \tLoss: %5.3f" % (i, loss.item()))

        return delta

    def search(self, img: torch.tensor, beta: float = 0.1, c: float = 1.,
               gamma: float = 100., theta: float = 100., K: int = 10,
               lr: float=1e-3, num_iters: int = 1000, verbose: bool=True):

        # label the input image using the classifier
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.clf(img)
            t0 = torch.argmax(output).item()

        # compute the encoding for the input image
        with torch.no_grad():
            enc_img, _ = self.ae(img)

        # send encoding to cpu and transform it to numpy array
        enc_img = enc_img.squeeze(0).cpu().numpy()

        # compute prototypes
        prototypes = self.compute_prototypes(enc_img, t0, K)

        # get the closest prototype
        prototype = self.closest_prototype(enc_img, prototypes)
        prototype = torch.tensor(prototype).unsqueeze(0).to(self.device)

        # apply optimization procedure
        delta = self.optimiziation(img, prototype, t0, beta, c, gamma, theta, lr, num_iters, verbose)
        delta = delta.detach()

        # for debugging, decode the prototype
        dec_prototype = self.ae.decoder(prototype).detach()

        # compute counterfactual
        ctf = torch.clamp(img + delta, min=-1, max=1)

        return {
            "img": img.cpu(),
            "ctf": ctf.cpu(),
            "proto": dec_prototype.cpu()
        }