import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        """ Constructor """
        super(Classifier, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3)
        )

        self.clf = nn.Sequential(
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.reshape(x.shape[0], -1)
        return self.clf(x)
