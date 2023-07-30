# -*- coding: utf-8 -*-
import torch.nn as nn


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*128*3, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 128*128*3),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 3, 128, 128)  # Reshape the output

