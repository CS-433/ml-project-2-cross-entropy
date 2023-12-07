import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader

from methods.base_method import BaseMethod


class MlpMethod(BaseMethod):
    def __init__(self, dataloader, batch_size=64, epochs=100):
        super(MlpMethod).__init__(dataloader)

        X = torch.from_numpy(self.dataloader.X.astype(np.float32))
        y = torch.from_numpy(self.dataloader.y.astype(np.float32))

        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.epochs = epochs
        self.mlp = MLP(181, [512, 128, 32], 1, 0.3)
        self.optimizer = torch.optim.Adam(self.mlp.parameters())
        self.loss_fn = MSELoss()

    def train(self):
        self.mlp.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in self.dataloader:
                self.optimizer.zero_grad()
                output = self.mlp(X_batch)
                loss = self.loss_fn(output, y_batch)
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def predict(self, x):
        self.mlp.eval()
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(self, input, hidden, output, dropout=0., *args, **kwargs):
        super(MLP).__init__(*args, **kwargs)

        fc_list = []

        for i, h in enumerate(hidden):
            if i == 0:
                fc_list.append(nn.Sequential(
                    nn.Linear(input, h),
                    nn.Dropout(p=dropout)
                ))
            else:
                fc_list.append(nn.Sequential(
                    nn.Linear(hidden[i - 1], h),
                    nn.Dropout(p=dropout)
                ))

        fc_list.append(nn.Sequential(
            nn.Linear(h, output),
            nn.Dropout(p=dropout)
        ))

    def forward(self, x):

        return x
