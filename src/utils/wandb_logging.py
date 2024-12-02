# use weights and biases logger
import sys
import os
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger


class WandBLogger():

    def __init__(self, architecture, lr, batch_size, epochs, time_encoding, criterion_name, num_layers, hidden_size, optim_name):
        self.architecture = architecture

        
        # Add the 'project' directory to the Python path
        sys.path.append(os.path.abspath("../"))
        wandb.init(dir= '../model/' + architecture)

        self.logger = WandbLogger(project='airbnb_stock_price_prediction', name = architecture)
        self.logger.experiment.config.update({
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "arch": architecture,
            "time_enc": time_encoding,
            "loss": criterion_name,
            "number_of_layers": num_layers,
            "hidden_size": hidden_size,
            "opt": optim_name
        })


    def endLog(self, model):
        model_path = '../../model/' + self.architecture + '/artifacts/' + wandb.run.id + '-' + wandb.run.name+'.pth'
        torch.save(model, model_path)

        artifact = wandb.Artifact(wandb.run.id + '-' + wandb.run.name, type='model')
        artifact.add_file(model_path)

        wandb.finish()