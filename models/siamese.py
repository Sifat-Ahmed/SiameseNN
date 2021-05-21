import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=1),
            nn.MaxPool2d(2, stride=2),
            #nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 256, kernel_size=2, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            #nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.3),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 16, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2),
            # nn.BatchNorm2d(8),
            # nn.Dropout2d(p=0.1),
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output = torch.sqrt(torch.sum((output1 - output2) * (output1 - output2), 1))

        return output1, output2, output


