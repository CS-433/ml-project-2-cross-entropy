import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_datasets(X_train, y_train, X_test, y_test):
    """
    Convert np.ndarray to TensorDataset
    """
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_set = TensorDataset(X_train_tensor, y_train_tensor)
    test_set = TensorDataset(X_test_tensor, y_test_tensor)

    return train_set, test_set

class Senon_model(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_phi = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.phi1 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out = self.input_layer(x)
        out = self.input_phi(out)
        out = self.dropout1(out)  
        out = self.layer1(out)
        out = self.phi1(out)
        out = self.dropout2(out)
        out = self.output_layer(out)
        return out


def train_epoch(model, device, train_loader, optimizer, epoch, criterion, scheduler):
    model.train()  
    loss_history = []
    lr_history = []
    total_mse = 0.0
    total_samples = 0
    
    if len(train_loader) == 0:
        raise ValueError("Train DataLoader is empty. Check your data loading process.")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()       

        loss_history.append(loss.item())
        lr_history.append(scheduler.get_last_lr()[0])
        
        # mse
        mse = nn.functional.mse_loss(output, target, reduction='sum')
        total_mse += mse.item()
        total_samples += data.size(0)    

        if batch_idx % max(1, len(train_loader) // 10) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx} batch_loss={loss.item()/len(data):0.2e}"
            )
    avg_mse = total_mse / total_samples
    rmse = np.sqrt(avg_mse)

    return loss_history, lr_history, rmse


@torch.no_grad()
def validate(model, device, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            
            # mse only for evaluation
            mse = nn.functional.mse_loss(output, target, reduction='sum')
            total_mse += mse.item()
            total_samples += data.size(0)

    val_loss /= len(val_loader)
    avg_mse = total_mse / total_samples
    rmse = np.sqrt(avg_mse)
    print(
        "Val set: Average loss: {:.4f}".format(
            val_loss,
            )
    )
    return val_loss, rmse



def run_model_training(train_set, val_set, num_epochs, lr, batch_size, device):
    # ===== Data Loading =====
    train_dataset = train_set
    val_dataset = val_set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # ===== Parameters =====
    model_kwargs = dict(
        input_size = 2,
        hidden_size = 32,
        dropout_prob = 0.5,
        )
    optimizer_kwargs = dict(
        lr=lr,
        weight_decay=1e-2,
        )
   
    # ===== Model, Optimizer, Criterion and Schedular =====
    model = Senon_model(**model_kwargs)
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), **optimizer_kwargs)
    # linear scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9) 
    # consine scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size,
    # )
    
    # ===== Train Model =====
    lr_history = []
    train_loss_history = []
    val_loss_history = []
    train_rmse_history = []
    val_rmse_history = []
    for epoch in range(1, num_epochs + 1):
        train_loss, lrs, train_rmse = train_epoch(
            model, device, train_loader, optimizer, epoch, criterion, scheduler
        )
        train_loss_history.extend(train_loss) 
        train_rmse_history.append(train_rmse) 

        val_loss, val_rmse = validate(model, device, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_rmse_history.append(val_rmse)
        
        lr_history.extend(lrs)
    
    # ===== Plot Result =====
    n_train = len(train_loss_history)
    t_train = num_epochs * np.arange(n_train) / n_train
    t_val = np.arange(1, num_epochs + 1)
    fig, axs = plt.subplots(1, 3, figsize = (15, 5))
    axs[0].plot(t_train, train_loss_history, label="Train loss")
    axs[0].plot(t_val, val_loss_history, label="Val loss")
    axs[0].legend()
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")   

    axs[1].plot(t_train, lr_history)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Learning Rate")
    
    axs[2].plot(t_val, train_rmse_history, label = 'Trian RMSE')
    axs[2].plot(t_val, val_rmse_history, label = 'Val RMSE')
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("RMSE")
    
    fig.tight_layout()
    plt.show()

