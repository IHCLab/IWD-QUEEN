import os
import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np

class DDM_dataset(Dataset):
    def __init__(self, base):
        self.file_name = base
        self.epsilon = 1e-25

        assert os.path.exists(base), 'Images folder does not exist'

        self.ddm = None
        self.index = None
        self.location = None
            
        for month_file in os.listdir(base):
            month_path = os.path.join(base, month_file)
            if os.path.isdir(month_path):
                for file in os.listdir(month_path):
                    if file.endswith('.mat'):
                        file_name = os.path.join(month_path, file)
                        data = scipy.io.loadmat(file_name)
                        if self.ddm is None:
                            self.ddm = data['ddm']
                        else:
                            self.ddm = np.concatenate((self.ddm, data['ddm']), axis=2)

                        if self.index is None:
                            self.index = data['index']
                        else:
                            self.index = np.concatenate((self.index, data['index']), axis=0)

                        if self.location is None:
                            self.location = data['location']
                        else:
                            self.location = np.concatenate((self.location, data['location']), axis=0)


    def __len__(self):
        return self.ddm.shape[2]

    def __getitem__(self, index):
		
        location = self.location[index,:]
        ddm_image = self.ddm[:, :, index]
        ddm_image = (ddm_image - np.min(ddm_image)) / (np.max(ddm_image) - np.min(ddm_image) + self.epsilon)
        ddm_image = torch.tensor(ddm_image, dtype=torch.float32)  

        target = torch.tensor(self.index[index], dtype=torch.float) 
        all_dict = {"ddm_image":ddm_image,
                    "target":target,
                    "location":location,
        }
        return all_dict

