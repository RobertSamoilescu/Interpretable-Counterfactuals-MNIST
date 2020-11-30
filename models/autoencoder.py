import torch.nn as nn


class AE(nn.Module):
    def __init__(self):
        """ Constructor """
        super(AE, self).__init__()

        # define encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),
        )

        # define decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),
        )

    def forward(self, x):
        # pass through the encoder
        enc_x = self.encoder(x)

        # pass through the decoder
        dec_x = self.decoder(enc_x)
        return enc_x, dec_x
