import torch, gc
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import os
import torch.nn.functional as F
import torch_scatter
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, HypergraphConv, global_add_pool, global_max_pool
from torch_geometric.utils import softmax
import argparse
import os
import os.path as osp
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold
import torch_geometric.transforms as T
import random
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from scipy.sparse import coo_matrix
from torch.nn.utils.rnn import pad_sequence
import subprocess
import json
import pprint
import time
from torch_geometric.utils import to_networkx
from datetime import datetime
import scipy.stats as stats
from torch import Tensor

import warnings
warnings.filterwarnings("ignore")

# dataset name
dataset_name = 'dparsf_aal_filt_global.npz'

# argument
def arg_parse():
    parser = argparse.ArgumentParser(description='SIGNET')
    parser.add_argument('--dataset', type=str, default='mutag')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=9999)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--encoder_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--pooling', type=str, default='add', choices=['add', 'max'])
    parser.add_argument('--readout', type=str, default='concat', choices=['concat', 'add', 'last'])
    parser.add_argument('--explainer_model', type=str, default='gin', choices=['mlp', 'gin'])
    parser.add_argument('--explainer_layers', type=int, default=5)
    parser.add_argument('--explainer_hidden_dim', type=int, default=8)
    parser.add_argument('--explainer_readout', type=str, default='add', choices=['concat', 'add', 'last'])
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--lstm_layers', type=int, default=1)
    parser.add_argument('--decoder_layers', type=int, default=1)
    parser.add_argument('--alpha', type=float, default = 10.0)
    parser.add_argument('--temporal_kernel_size', type=int, default=5)
    parser.add_argument('--edge_importance_weighting', type=bool, default=False)
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    return parser.parse_args()

# gpu check
DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    temp = [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]
    for i, itm in enumerate(temp):
        if itm['index']=='3':
            return itm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_split(args, fold=5):
    dataset = Mdd(root=data_dir + '/mutag')
    DS = args.dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    dataset = TUDataset(path, name=DS)
    data_list = []
    label_list = []

    for data in dataset:
        data_list.append(data)
        label_list.append(data.y.item())

    kfd = StratifiedKFold(n_splits=fold, random_state=0, shuffle=True)

    splits = []
    for k, (train_index, test_index) in enumerate(kfd.split(data_list, label_list)):
        splits.append((train_index, test_index))

    return splits


def sparsify_adjacency_matrix(adj_matrix, keep_percentage):
    # num_rows, num_cols = adj_matrix.shape

    threshold_value = np.percentile(adj_matrix, 100 - keep_percentage)
    sparse_adj_matrix = np.where(adj_matrix >= threshold_value, adj_matrix, 0.0)

    return sparse_adj_matrix

class Mdd(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        mdd_file = np.load('/DataRead/ABIDE/ABIDE_I_Raw/'+dataset_name, allow_pickle=True)
        mdd_signals = mdd_file['signal']
        mdd_labels = mdd_file['DX_GROUP']
        mdd_labels[mdd_labels==2]=-1
        mdd_fcs = np.load('.npy')
        mdd_fcs_adj = np.load('.npy')
        n_subject = mdd_fcs.shape[0]
        n_time = mdd_fcs.shape[1] 
        n_node = mdd_fcs.shape[2] 
        n_feature = mdd_fcs.shape[3]

        # adj self loop 제거
        for i in range(mdd_fcs.shape[0]):
            for j in range(mdd_fcs.shape[1]):
                for k in range(mdd_fcs.shape[2]):
                    mdd_fcs_adj[i][j][k][k]=0.0

        # adj 희소화
        for i in range(mdd_fcs.shape[0]):
            for j in range(mdd_fcs.shape[1]):
                mdd_fcs_adj[i][j] = sparsify_adjacency_matrix(mdd_fcs_adj[i][j], 1)

        mdd_labels[mdd_labels == -1] = 0

        data_list = []
        for i in range(n_subject):
            edge_indexs =[]
            fcs = []
            for j in range(n_time):
                edge_index = coo_matrix(mdd_fcs_adj[i][j])
                fc = edge_index.data

                edge_index = np.stack((edge_index.row, edge_index.col), axis=0)
                edge_index = edge_index.astype(np.int64)
                
                edge_indexs.append(edge_index)
                fcs.append(fc)

            edge_index = np.stack(edge_indexs, axis = 0)
            
            unique_node = []
            for  j in range(n_time):
                unique_node.append(np.unique(edge_index[j]))

            edge_index = torch.tensor(edge_index)
            fc = np.stack(fcs, axis = 0)
            fc = torch.tensor(fc).float()

            mdd_label = torch.tensor(mdd_labels[i]).float().reshape(-1, 1)
            
            mdd_signal =[]
            for j in range(n_time):
                mdd_signals_mid=[]
                for k in range(n_node):
                    temp = mdd_fcs[i][j][k]
                    mdd_signals_mid.append(temp)
                mdd_signals_mid = np.stack(mdd_signals_mid, axis = 0)
                mdd_signal.append(mdd_signals_mid)
            mdd_signal = np.stack(mdd_signal, axis = 0)
            mdd_signal = torch.tensor(mdd_signal).float()

            data_list.append(Data(signal=mdd_signal, label=mdd_label, edge_index=edge_index, fc=fc, unique_node=unique_node, adj=mdd_fcs_adj[i]))
        data, slices = self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0])


def get_random_split_idx(dataset, random_state=None, val_per=0.2, test_per=0.2, classification_mode=False):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    n_val = int(val_per * len(idx))
    n_test = int(test_per * len(idx))
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx_raw = idx[n_val + n_test:]
    normal_mask = (dataset.data.label[train_idx_raw] == 0).numpy()
    if classification_mode:
        train_idx = train_idx_raw #[normal_mask]
    else: 
        train_idx = train_idx_raw[normal_mask]

    ano_mask_test = (dataset.data.label[test_idx] == 1).numpy() 
    explain_idx = test_idx[ano_mask_test]

    return {'train': train_idx, 'val': val_idx,'test': test_idx, 'explain': explain_idx}

def get_loaders_mdd(batch_size, batch_size_test, dataset, split_idx=None):
    train_loader = DataLoader(dataset[split_idx['train']], batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(dataset[split_idx['val']], batch_size = batch_size_test, shuffle=False)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size = batch_size_test, shuffle=False)
    explain_loader = DataLoader(dataset[split_idx['explain']], batch_size= 1, shuffle=False)
    return {'train':train_loader, 'val': val_loader,'test':test_loader, 'explain':explain_loader}

def get_data_loaders(dataset_name, batch_size, batch_size_test=None, random_state=0, data_dir='data'):
    # dataset = Mdd(root=data_dir + '/mdd1_2')
    dataset = Mdd(root=data_dir +'/mdd_aal')

    dataset.data.label = dataset.data.label.squeeze()
    
    dataset.data.label = 1 - dataset.data.label
    split_idx = get_random_split_idx(dataset, random_state)
    loaders = get_loaders_mdd(batch_size, batch_size_test, dataset=dataset, split_idx=split_idx)
    num_feat = dataset.data.signal.shape[2] 
    num_edge_feat = 0
    num_node = dataset.data.signal.shape[1]
    num_time = dataset.data.signal.shape[0] // dataset.data.label.shape[0]

    meta = {'num_node': num_node,'num_feat': num_feat, 'num_edge_feat': num_edge_feat, 'num_time': num_time}

    return loaders, meta

    

def run(args, device, seed, split=None):
    set_seed(seed)
    loaders, meta = get_data_loaders(args.dataset, args.batch_size, args.batch_size_test, random_state=seed)
    n_feat = meta['num_feat']
    n_edge_feat = meta['num_edge_feat']
    n_node = meta['num_node']
    n_time = meta['num_time']
    model = # model 추가해야함
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_rec_mean = nn.MSELoss()
    loss_rec_none = nn.MSELoss(reduction='none')

    explain_loader = loaders['explain']
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    best_model_name =''
    best_auc = 0
    
    epochs_vis = list(range(args.epochs))
    val_auc_vis = []
    test_auc_vis = []

    for epoch in range(1, args.epochs+1):
        model.train()

        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            ret = model(data)
            loss = # ret이랑 ground truth 이용해서 loss 계산
            loss.backward()
            optimizer.step()
            print(f'loss: {loss}')

        if epoch % args.log_interval == 0:
            model.eval()
            all_ad_true = []
            all_ad_score = []
            for data in val_loader:
                all_ad_true.append(data.label.cpu())
                ad_true_check =data.label.cpu()
                data = data.to(device)
                with torch.no_grad():
                    ret = model(data)
                    loss = # ret이랑 ground truth 이용해서 loss 계산
                    anomaly_score = # loss 이용해서 계산

            ad_true = torch.cat(all_ad_true)
            ad_score = torch.cat(all_ad_score)
            ad_auc_val = roc_auc_score(ad_true, ad_score)
            # to select optimal thresholding value
            fpr, tpr, thresholds = roc_curve(ad_true, ad_score)


# main
if __name__ == '__main__':
    args = arg_parse()

    # device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    print('Current cuda device: ', torch.cuda.current_device()) 
    print('Count of using GPUs: ', torch.cuda.device_count())


    splits=[None]*args.num_trials
    for trial in range(args.num_trials):
        run(args, device, seed=trial, split=splits[trial])

