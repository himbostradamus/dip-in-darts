import cnn6
import CERDataset

import matplotlib.pyplot as plt
import numpy as np

from nni import report_intermediate_result, report_final_result
from nni.retiarii.evaluator.pytorch import LightningModule
from nni.retiarii.evaluator.pytorch.lightning import DataLoader
import nni.retiarii.nn.pytorch as nn

import torch
from torch.optim import Optimizer
from torch import tensor
from torch.utils.data import Dataset

from typing import Any

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


class EvalLandmark(LightningModule):
    def __init__(
                 self, 
                 learning_rate=0.001,
                 mode="pretrain2",
                 log_every=25,

                ):
        super().__init__()
        self.loss = nn.MSELoss()
        self.validation_loss_list = []
        self.train_loss_list = []
        self.log_every = log_every
        self.learning_rate = learning_rate

        if mode == 'pretrain1':
            self.landmarkdir = 'TrainingLabels1'
            self.imagedir = 'TrainingImages1'
            self.params_name = '1pretrain_params.pt'
            self.train_loss_name = '1pretrain_train_loss.npy'
            self.validation_loss_name = '1pretrain_val_loss.npy'
            self.figure_name = '1pretrain_loss.png'
        elif mode == 'pretrain2':
            self.landmarkdir = 'TrainingLabels1'
            self.imagedir = 'TrainingImages2'
            self.params_name = '2pretrain_params.pt'
            self.train_loss_name = '2pretrain_train_loss.npy'
            self.validation_loss_name = '2pretrain_val_loss.npy'
            self.figure_name = '2pretrain_loss.png'


    def configure_optimizers(self) -> Optimizer:
        """
        We are doing automatic implementation
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def on_train_start(self):
        self.model = self.model.to(self.device)
        self.validation_loss = 0

    def forward(self, x):
        return self.model(x)
            
    def training_step(self, batch, batch_idx):
        # Predict
        pred = self.forward(batch['image'])

        # Eval Loss
        self.train_loss = self.loss(pred, batch['landmarks'])

        # append for plotting
        self.train_loss_list.append(self.train_loss.item())

        # logging for lightning and nni
        self.log('train_loss', self.train_loss)

        return self.train_loss
        
    def validation_step(self, batch, batch_idx):
        # Predict
        pred = self.forward(batch['image'])

        # Eval Loss
        self.validation_loss = self.loss(pred, batch['landmarks'])

        # append for plotting
        self.validation_loss_list.append(self.validation_loss.item())

        # logging for lightning and nni
        self.log('validation_loss', self.validation_loss, prog_bar=False)

    def validation_epoch_end(self, *args, **kwargs):
        if self.current_epoch % self.log_every == 0 and self.current_epoch != 0:      
            report_intermediate_result({
                'Epoch': self.current_epoch,
                'train_loss': round(self.train_loss.item(),5),
                'val_loss': round(self.validation_loss.item(),5),
                })
            # print(f"Epoch: {self.current_epoch}")
            # print(f"Train Loss: {self.train_loss.item()}")
            # print(f"Validation Loss: {self.validation_loss.item()}")

    def on_train_end(self):
        print("Done!")
        report_final_result({
            'train_loss': self.train_loss.item(),
            'val_loss': self.validation_loss.item(),
            })
        self.plot_loss(
            self.train_loss_list, 
            self.validation_loss_list, 
            self.train_loss_name, 
            self.validation_loss_name, 
            self.figure_name
            )
    
    def plot_loss(self, train_loss_list, validation_loss_list, train_loss_name, validation_loss_name, figure_name):
        x = np.arange(1, self.current_epoch + 1)    
        train_loss = np.array(train_loss_list)
        np.save(train_loss_name, train_loss)
        validation_loss = np.array(validation_loss_list)
        np.save(validation_loss_name, validation_loss)
        plt.plot(x, train_loss, label='trainloss')
        plt.plot(x, validation_loss, label='validation_loss')
        plt.legend()
        plt.savefig(figure_name)