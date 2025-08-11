import torch
import argparse
from model.IWD_QUEEN import CoreNet as IWD_QUEEN
from model.IWD_Transformer import CoreNet as IWD_Transformer
from init import *
from scipy.io import savemat
import numpy as np
import time

parser = argparse.ArgumentParser(description='Inference script for IWD models')
parser.add_argument('--model_name', type=str, required=True,
                    choices=['IWD_QUEEN', 'IWD_Transformer'], help='Model to use')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to the trained model checkpoint')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to inference dataset')
parser.add_argument('--save_path', type=str, required=True,
                    help='Output path for saving .mat file')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model_name == 'IWD_QUEEN':
    model = IWD_QUEEN().to(device)
elif args.model_name == 'IWD_Transformer':
    model = IWD_Transformer().to(device)

checkpoint = torch.load(args.model_path, weights_only=True)
model.load_state_dict(checkpoint['Net'])
model.eval()

test_loader = init_data(args.data_path, 100)

all_predict = []
all_location = []
gt_output = []
time_total = 0

with torch.no_grad():
    for batch_idx, train_data in enumerate(test_loader):
        data = train_data['ddm_image'].to(device)
        gt = train_data['target'].squeeze(dim=1).to(device)
        location = train_data['location']

        start = time.time()
        probabilities = model(data)
        end = time.time()
        time_total += end - start

        all_predict.append(probabilities.cpu().numpy())
        all_location.append(location.cpu().numpy())
        gt_output.append(gt.cpu().numpy())

all_predict = np.concatenate(all_predict)
all_location = np.concatenate(all_location)
gt_output = np.concatenate(gt_output)

output = (all_predict[:, 0] > 0.5).astype(np.int64)

results_dict = {
    'all_output': output,
    'all_location': all_location,
    'gt_output': gt_output,
    'inference_time': time_total
}
savemat(args.save_path, results_dict)
