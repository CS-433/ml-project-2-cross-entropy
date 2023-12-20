import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from methods.base_method import BaseMethod


class MlpMethod(BaseMethod):
    def __init__(self, dataloader, hidden=[128, 64], dropout=0.1, wd=1e-2, batch_size=64, epochs=100, lr=0.005):
        super().__init__(dataloader)

        X = torch.from_numpy(self.dataloader.X.astype(np.float32))
        y = torch.from_numpy(self.dataloader.y.astype(np.float32))

        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.epochs = epochs
        self.mlp = MLP(186, hidden, 1, dropout)
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 50)
        self.loss_fn = MSELoss()

    def train(self):
        self.mlp.train()
        for epoch in tqdm(range(self.epochs)):
            for X_batch, y_batch in self.dataloader:
                self.optimizer.zero_grad()
                output = self.mlp(X_batch)
                loss = self.loss_fn(output, y_batch)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                print(f"{epoch}: {loss:.4f}")

    @torch.no_grad()
    def predict(self, x):
        self.mlp.eval()
        x = torch.from_numpy(x.astype(np.float32))
        return self.mlp(x).numpy()


class MLP(nn.Module):
    def __init__(self, input, hidden, output, dropout=0., *args, **kwargs):
        super().__init__(*args, **kwargs)

        fc_list = []

        for i, h in enumerate(hidden):
            if i == 0:
                fc_list.append(nn.Sequential(
                    nn.Linear(input, h, bias=False),
                    nn.Dropout(p=dropout),
                    nn.Sigmoid(),
                ))
            else:
                fc_list.append(nn.Sequential(
                    nn.Linear(hidden[i - 1], h, bias=False),
                    nn.Dropout(p=dropout),
                    nn.Sigmoid(),
                ))

        fc_list.append(nn.Sequential(
            nn.Linear(h, output, bias=False)
        ))

        self.mlp = nn.Sequential(*fc_list)

    def forward(self, x):
        x = self.mlp(x)
        return x
