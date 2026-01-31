import torch
import torch.nn as nn
import os

class LSTMnet(nn.Module):
    def __init__(self, cfgs):
        super(LSTMnet, self).__init__()
        self.lstm = nn.LSTM(input_size=cfgs["input_size"], hidden_size=cfgs["hidden_size"],
                         num_layers=cfgs["num_layers"], batch_first=cfgs["batch_first"], 
                         dropout=cfgs["dropout_rate"])
        self.linear = nn.Linear(in_features=cfgs["hidden_size"], out_features=cfgs["output_size"])
        self.cfgs = cfgs
    def forward(self, input, h_n, c_n):
        if h_n is None or c_n is None:
            # Dynamically initialize the hidden state, adjusting the dimensions based on `batch_first`.
            batch_size = input.size(0) if self.lstm.batch_first else input.size(1)
            h_n = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(input.device)
            c_n = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(input.device)
        output, (h_n, c_n) = self.lstm(input, (h_n, c_n))
        # Extract the hidden state of the last time step.
        if self.cfgs["batch_first"]:
            last_hidden = output[:, -self.cfgs["predict_data"]:, :]  # (batch_size, predict_data, hidden_size)
            # Mapped to the target shape via a fully connected layer.
            output = self.linear(last_hidden)  # (batch_size, predict_data * output_size)
            output = output.view(-1, self.cfgs["predict_data"], self.cfgs["output_size"])  # Adjust the shape to (batch_size, predict_data, output_size)
        else:
            last_hidden = output[-1, :, :]  # (batch_size, hidden_size)
            output = self.linear(last_hidden)  # (batch_size, predict_data * output_size)
            output = output.view(self.cfgs["predict_data"], -1, self.cfgs["output_size"]) 
        
        return output, (h_n, c_n)

def save_model(cfgs, model, epoch, folder):
    """
    Save model weights
    """
    path = os.path.join(cfgs["save_model_path"], folder)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder created at {path}")
    filename = f"model_epoch{epoch}.pth"
    path = os.path.join(path, filename)

    torch.save(model.state_dict(), path)
    print(f"Model for epoch {epoch} saved at {path} !")

def load_model(cfgs, epoch, device):
    """
    Loading model weights
    """
    filename = f"model_epoch{epoch}.pth"
    path = os.path.join(cfgs["save_model_path"], f"training_epochs_{cfgs['epochs']}_stepsize_{cfgs['step_size']}_gamma_{cfgs['gamma']}")
    path = os.path.join(path, filename)
    model = LSTMnet(cfgs).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path} !")
    return model