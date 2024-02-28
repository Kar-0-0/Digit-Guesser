import torch.nn as nn


class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.float()
        x = x.reshape((-1, 28**2))
        x = self.l2(self.relu(self.l1(x)))
        return x.squeeze()
