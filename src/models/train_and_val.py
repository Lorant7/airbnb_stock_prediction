import torch
import pandas as pd
import numpy as np
from torch import nn
import lightning as L

import base
from src.utils.wandb_logging import WandBLogger
from src.models.base import StockDataModule
from src.config import BATCH_SIZE
# from ..utils.wandb_logging import WandBLogger
# from .base import StockDataModule
# from ..config import BATCH_SIZE


architecture = 'LSTM'
# BATCH_SIZE = 16 : change it in the config.py file
epochs = 25
lr = 0.01
num_layers = 16
hidden_size = 128

criterion_name = 'BCEWithcLogitsLoss'
time_encoding = 'sin-cos'
input_size = 9
output_size = 1 # Because we are trying to predict weather the Closing value will be higher or lower than the current Closing Value
optim_name = 'Adam'


criterion = nn.BCEWithLogitsLoss()

model = base.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
lit_model = base.litModule(model, criterion=criterion, lr=lr)

logger = WandBLogger(architecture, lr, BATCH_SIZE, epochs, time_encoding, criterion_name, num_layers, hidden_size, optim_name)

dm = StockDataModule(batch_size=BATCH_SIZE)

trainer = L.Trainer(max_epochs=epochs, logger = logger.logger)
trainer.validate(model = lit_model, datamodule = dm)
trainer.fit(model = lit_model, datamodule = dm)

logger.endLog(model)
