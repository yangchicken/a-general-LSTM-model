import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os 
from dataprocess import config_loader, read_process_data, makedataloader,adjust_batch_first,denormalize_data
from model import LSTMnet, load_model, save_model
from log import setup_logger

def initialization(cfgs): 
    '''
        Initialization, including reading data, data preprocessing, 
        creating the dataset, and creating the dataloader.
    '''
    x_train, y_train, x_test, y_test ,min_val, max_val= read_process_data(cfgs)
    train_dl, test_dl =  makedataloader(x_train, y_train, x_test, y_test, cfgs)
    
    inputs_batch, label_batch = next(iter(test_dl))
    print(label_batch.shape)
    return train_dl, test_dl, min_val, max_val

def run_model(cfgs, train_dl, test_dl, min_val, max_val, device):
    '''
        Model training, prediction
    '''
    epochs = cfgs["epochs"]
    model = LSTMnet(cfgs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = cfgs["lr"])
    step_size=cfgs["step_size"]
    gamma=cfgs["gamma"]
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.MSELoss()
    # Used to store all test predictions and true values.
    all_y_pred = []
    all_y_true = []

    folder = f"training_epochs_{epochs}_stepsize_{step_size}_gamma_{gamma}"
    log_path = os.path.join(cfgs["log_path"], folder)
    # Set up the logger.
    logger = setup_logger(log_path)

    for epoch in range(1, epochs+1):
        total_loss = 0
        total_sample = 0
        h_n = None
        c_n = None

        model.train()
        for x, y in train_dl:
            x, y = adjust_batch_first(x, y, cfgs)
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
                y_pred, hidden= model(x, h_n, c_n)
            h_n, c_n = hidden
            h_n.detach_(), c_n.detach_()    # Remove gradient information.

            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_sample += 1
            with torch.no_grad():
                total_loss += loss.item()
        exp_lr_scheduler.step()
        epoch_loss = total_loss / total_sample

        test_loss = 0
        test_sample = 0
        h_n = None
        c_n = None
        
        model.eval()
        for x, y in test_dl:
            x, y = adjust_batch_first(x, y, cfgs)
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred, hidden= model(x, h_n, c_n) # y_pred.shape (batch_size, predict_data, output_size)
            h_n, c_n = hidden
            h_n.detach_(), c_n.detach_()
            loss = loss_fn(y_pred, y)
            test_sample += 1
            with torch.no_grad():
                test_loss += loss.item()
            # Save the predicted results and the actual values.
            if epoch == epochs: 
                all_y_pred.append(y_pred.cpu().detach())
                all_y_true.append(y.cpu().detach())
            
        epoch_test_loss = test_loss / test_sample
        print(f"epoch:{epoch},train_loss:{epoch_loss}, test_loss:{epoch_test_loss}")
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.6f} | Test Loss: {epoch_test_loss:.6f}")
        if epoch % 10 == 0:
            save_model(cfgs, model, epoch, folder)
    all_y_true = torch.cat(all_y_true, dim=0)
    all_y_pred = torch.cat(all_y_pred, dim=0)
    # aligned_preds = draw(cfgs, all_y_pred, all_y_true, min_val, max_val)

def draw(cfgs, all_y_pred, all_y_true, min_val, max_val):
    """
    Process the predicted results and actual values, and then plot the graph.
    """
    predict_steps = cfgs["predict_data"]  # Predicted number of steps
    output_size = cfgs["output_size"]    # The dimensions of the output at each step.
    file_path = cfgs["dataroot"]
    label_cols = cfgs["label_cols"]

    time_len = all_y_true.shape[0] + predict_steps - 1
    aligned_preds = np.zeros((time_len, output_size))
    counts = np.zeros((time_len, output_size))

    for i in range(all_y_pred.shape[0]):
        for t in range(predict_steps):
            aligned_preds[i+t] += all_y_pred[i, t].cpu().detach().numpy()
            counts[i+t] += 1
    aligned_preds /= np.maximum(counts, 1)

    # for i in range(all_y_pred.shape[0]):
    #     aligned_preds[i] += all_y_pred[i, 0].cpu().detach().numpy()
    all_y_true = all_y_true[:, 0, :].cpu().detach().numpy()
    # Convert the normalized data back to the original data.
    aligned_preds = denormalize_data(aligned_preds, min_val, max_val, cfgs)
    all_y_true = denormalize_data(all_y_true, min_val, max_val, cfgs)
    # Get the table header
    header = pd.read_csv(file_path, nrows=0).columns.tolist()
    header = [header[i] for i in label_cols]
    
    # Create savefig folder if it doesn't exist
    save_dir = "savefig"
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(output_size):
        plt.figure()
        plt.plot(all_y_true[:, i].flatten(), label = f"True Data of {header[i]}")
        plt.plot(aligned_preds[:, i].flatten(), label = f"predicted Date of {header[i]}", linestyle = "dashed")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{header[i]}_prediction.png"), dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Figure saved to {os.path.join(save_dir, f'{header[i]}_prediction.png')}")
    return aligned_preds[-predict_steps, :]

def predict(cfgs, test_dl, min_val, max_val, device):
    model = load_model(cfgs, 200, device)
    h_n = None
    c_n = None

    predictlist = []
    originlist = []
    model.eval()
    for x, y in test_dl:
        x, y = adjust_batch_first(x, y, cfgs)
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred, hidden= model(x, h_n, c_n)
        h_n, c_n = hidden
        h_n.detach_(), c_n.detach_()
        predictlist.append(y_pred)
        originlist.append(y)
    all_y_true = torch.cat(originlist, dim=0)
    all_y_pred = torch.cat(predictlist, dim=0)
    aligned_preds = draw(cfgs, all_y_pred, all_y_true, min_val, max_val)

    return aligned_preds
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main program for LSTM')
    parser.add_argument('--cfgs', type=str, default='./cfgs.yaml', 
                        help="path of config file")
    parser.add_argument('--dataroot', type=str, default='./data.csv',
                        help='path of data file')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--predict', type=bool, default=True)
    opt = parser.parse_args()
    cfgs = config_loader(opt.cfgs)

    train_dl, test_dl, min_val, max_val= initialization(cfgs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if opt.train:
        run_model(cfgs, train_dl, test_dl, min_val, max_val, device)
    if opt.predict:
        aligned_preds = predict(cfgs, test_dl, min_val, max_val, device)

    

