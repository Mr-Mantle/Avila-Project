import torch as t
import torch.nn as nn
import numpy as np
"""
This file provide an overview of all the model we used in our Avila test, no forward method will be 
provided in the following Models.
"""


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.BaseModel = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.Softmax(dim=-1)
        )


class SEnetModel(nn.Module):
    def __init__(self):
        super(SEnetModel, self).__init__()
        self.SENet = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.Sigmoid()
        )

        self.Net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.Softmax(dim=-1)
        )


class AENet(nn.Module):
    def __init__(self):
        super(AENet, self).__init__()
        self.AENet = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 9),
            nn.ReLU(),
            nn.Linear(9, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

        self.Net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.Softmax(dim=-1)
        )


class AESENet(nn.Module):
    def __init__(self):
        super(AESENet, self).__init__()
        self.AENet = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 9),
            nn.ReLU(),
            nn.Linear(9, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

        self.SENet = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.Sigmoid()
        )

        self.Net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.Softmax(dim=-1)
        )


if __name__ == '__main__':
    model = SEnetModel()
    t.save(model, "../Avila_project/Model_used/SENet.pth")