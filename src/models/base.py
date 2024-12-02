import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
import lightning as L
from pytorch_lightning import LightningModule, LightningDataModule
from scipy import stats

import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[2]  # Adjust the level as needed
sys.path.append(str(project_root))

from src.data import preprocess
from src.config import RAW_DATA_DIR
from src.config import PROCESSED_DATA_DIR
from src.data.load import loadData
# from ..data.preprocess import feature_selection
# from ..config import RAW_DATA_DIR
# from ..config import PROCESSED_DATA_DIR
# from ..data.load import loadData

import os

class Data(Dataset):
    def __init__(self, df):
        self.x = torch.tensor(df.drop(columns=['price_change','Date','month', 'day', 'change', 'year', 'Dividends', 'Stock Splits']).values, dtype = torch.float32)
        self.y = torch.tensor(df['change'].values, dtype=torch.float32)
        self.len = df.shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    

class StockDataModule(L.LightningDataModule):
    def __init__(self, batch_size=8):
        super().__init__()
        self.batch_size = batch_size

    # Feature selection and engineering
    def setup(self, stage=None):
        loadData()
        df = pd.read_csv(RAW_DATA_DIR + os.listdir(RAW_DATA_DIR)[-1])
        preprocess.feature_selection(df)

        # Splitting the data into train, validation, and test
        split_index1 = int(df.shape[0] * 0.2)
        split_index2 = int(df.shape[0] * 0.1)

        self.test_df = df.iloc[:split_index2]
        self.val_df = df.iloc[split_index2:split_index1]
        self.train_df = df.iloc[split_index1:]
        # print("saving dfs to : ", PROCESSED_DATA_DIR + "/train_df.csv")
        self.train_df.to_csv(PROCESSED_DATA_DIR + "/train_df.csv", index=False)
        self.test_df.to_csv(PROCESSED_DATA_DIR + "/test_df.csv", index=False)
        self.val_df.to_csv(PROCESSED_DATA_DIR + "/val_df.csv", index=False)


    def train_dataloader(self):
        return DataLoader(Data(self.train_df), batch_size=self.batch_size, shuffle=False, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(Data(self.val_df), batch_size=self.batch_size, shuffle=False, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(Data(self.test_df), batch_size=self.batch_size, shuffle=False, drop_last=True)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.l = nn.Linear(in_features=hidden_size, out_features=output_size)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize RNN weights using Xavier initialization
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)  # Input-hidden weights
            elif 'weight_hh' in name:
                init.orthogonal_(param)  # Hidden-hidden weights
            elif 'bias' in name:
                init.zeros_(param)  # Initialize biases to zeros

        # Initialize the linear layer with Xavier initialization
        init.xavier_uniform_(self.l.weight)
        if self.l.bias is not None:
            init.zeros_(self.l.bias)

    def forward(self, x):
        x = x.unsqueeze(0)
        output, h_n = self.rnn(x)
        x = self.l(h_n[-1]).squeeze()
        # No need to pass the logits of the final linear layer through a sigmoid function because I am using
        # the BCEWithLogitsLoss which applies it internally
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.fll = nn.Linear(in_features=hidden_size, out_features=1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize RNN weights using Xavier initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)  # Input-hidden weights
            elif 'weight_hh' in name:
                init.orthogonal_(param)  # Hidden-hidden weights
            elif 'bias' in name:
                init.zeros_(param)  # Initialize biases to zeros

        # Initialize the linear layer with Xavier initialization
        init.xavier_uniform_(self.fll.weight)
        if self.fll.bias is not None:
            init.zeros_(self.fll.bias)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.lstm(x)
        x = self.fll(x[1][-1])
        return x.squeeze(-1)[-1]

class litModule(L.LightningModule):
    def __init__(self, model, criterion, lr):
        super().__init__()
        self.model = model
        self.loss = criterion
        self.spear = stats.spearmanr
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # In theory, you don't need the with no torch grad because the lightning module does it for you
        with torch.no_grad():
            yhat = self.model(x)
            loss = self.loss(yhat, y)
            self.log('val_loss', loss)

            spearman = self.spear(yhat, y).statistic
            self.log('val_spearman', spearman, on_step = False, on_epoch = True, prog_bar = True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
