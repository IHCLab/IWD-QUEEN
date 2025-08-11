import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from os.path import join
from tqdm import tqdm
from model.IWD_QUEEN import CoreNet
from init import *
import json

file_path = './config.json'

with open(file_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)


class Train(nn.Module):
    def __init__(self, lambda_coeff=0.5):
        super().__init__()
        self.lambda_coeff = lambda_coeff
        self.train_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = \
            init_data(cfg['train_data_path'], cfg['batch_size'])
        self.Net = CoreNet().to(self.train_dev)
        self.opt = init_optimizer(self.Net.parameters())
        self.step_scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=100, gamma=0.9)
        self.loss_bce=nn.BCELoss()

    def kappa_loss(self, outputs, targets):
            batch_size = outputs.shape[0]
            outputs = outputs.view(batch_size, -1)  
            targets = targets.view(batch_size, -1)  

            N = outputs.shape[0] 

            numerator = 2 * (outputs * targets).sum() - (outputs.sum() * targets.sum()) / N
            denominator = (outputs**2).sum() + (targets**2).sum() - 2 * (outputs * targets).sum() / N

            loss = 1 - numerator / (denominator + 1e-8)
            return loss

    def Net_loss(self, model_output, gt):
        gt = gt.float()
        loss_bce = self.loss_bce(model_output, gt)  
        loss_kappa = self.kappa_loss(model_output, gt)  

        loss = self.lambda_coeff * loss_bce + (1 - self.lambda_coeff) * loss_kappa
        return loss
    
    def train_part(self, train_loader,):
            train_loss_list_G = []     
            self.Net.train()
            for _, train_data in enumerate(train_loader, start=1):
                data = train_data['ddm_image'].to(self.train_dev)
                gt = train_data['target'].to(self.train_dev)
                self.opt.zero_grad()
                estimate_output = self.Net(data)
                errG = self.Net_loss(estimate_output,gt)
                errG.backward()
                self.opt.step()
                train_loss_list_G.append(errG.item())
            self.step_scheduler.step()
            train_loss = np.mean(train_loss_list_G)

            return train_loss
    
    def forward(self):
        for i in tqdm(range(cfg['epochs'])):
            train_loss_G = self.train_part(self.train_loader)

            if (i + 1) % 5 == 0 or (i + 1) == cfg['epochs']:
                torch.save({'Net': self.Net.state_dict()},
                        join('./train_result', "model_IWD_QUEEN.pt"))

if __name__ == "__main__" :
    model = Train()
    model()
