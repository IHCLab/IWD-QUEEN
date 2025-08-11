import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader import DDM_dataset

def init_optimizer(all_parameter):
    opt = torch.optim.Adam(all_parameter,0.01)
    return opt

def init_data(train_data_path,batch_size):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        
    #create a dataset
    train_set = DDM_dataset(train_data_path)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=10,worker_init_fn=seed_worker)

    return train_loader
