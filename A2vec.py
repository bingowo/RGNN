import opengsl
import time
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
import nni
import os
from tqdm import tqdm
import random
os.chdir("/mbc/GSL/RGNN")

class EdgeData(Dataset):
    def __init__(self, adj, acc, max_v):
        edge = []
        self.n = len(adj)
        A = adj.to_dense()
        for i, t in enumerate(adj):
            for j in t.coalesce().indices()[0]:
                edge.append([i,int(j)])
                assert(A[i][j] == 1)
                
        self.edge = edge
        self.n_edge = len(edge)
        self.acc = acc

        self.result = list()
        for idx in range(int(self.n_edge*self.acc)):
            if idx >= self.n_edge:
                x = random.randint(0, self.n-1)
                y = random.randint(0, self.n-1)
                value = 0
            else:
                x = self.edge[idx][0]
                y = self.edge[idx][1]
                value = max_v
            self.result.append((torch.tensor(x), torch.tensor(y), torch.tensor(value, dtype=torch.float)))

    def __getitem__(self, idx):
        return self.result[idx]

    def __len__(self):
        return int(self.n_edge*self.acc)
#torch.nn.functional.normalize()
@torch.no_grad()
def normalize(feat):
    n = feat.shape[0]
    for i in range(n):
        x = feat[i]
        mean = x.mean()
        std = x.std()
        feat[i] = (x-mean)/std
    return feat

if __name__ == "__main__":
    datasets = ['cora', 'citeseer', 'pubmed', 'blogcatalog', 'flickr', \
                'amazon-ratings', 'roman-empire', 'questions', 'minesweeper', 'wiki-cooc']
    config_path = "./config.yaml"

    data = datasets[0]
    runs = 1
    if nni.get_trial_id()=="STANDALONE":
        config_path = "./config/{}.yaml".format(data)
        runs = 1

    conf = opengsl.config.load_conf(path=config_path)
    # conf = opengsl.config.load_conf(method='gcn', dataset="roman-empire")
    print(conf)
    # feat_norm  :    true for homephily dataset
    dataset = opengsl.data.Dataset(data, n_splits=1, feat_norm=conf.dataset['feat_norm'])
    print("Dataset: [{}]".format(data))

    dims = 1024
    max_v = 64
    feat = torch.rand((dataset.n_nodes, dims), requires_grad=True, device="cuda")
    feat = normalize(feat)
    optim = torch.optim.Adam([feat], lr=conf.training['lr'], weight_decay=conf.training['weight_decay'])
    '''
        test for dataset
    '''
    n = dataset.n_nodes
    edge_data = EdgeData(dataset.adj, acc=2, max_v=max_v)
    edge_loader = DataLoader(edge_data, batch_size=int(dataset.n_edges), shuffle=True)
    loss_fn = torch.nn.functional.mse_loss
    for epoch in range(1000):
        tot_x, tot_y, tot = 0,0,0
        for x,y,v in edge_loader:
            x = x.cuda()
            y = y.cuda()
            v = v.cuda()

            x = feat[x]
            y = feat[y]
            s = x*y
            s = s.sum(axis=1)

            x = x*x
            x = x.sum(axis=1)
            y = y*y
            y = y.sum(axis=1)
            
            optim.zero_grad()
            ones= torch.ones_like(x, device="cuda") * dims
            
            loss = loss_fn(s, v)
            loss_x = loss_fn(x, ones)
            loss_y = loss_fn(y, ones) 
            
            tot += loss.item()
            tot_x += loss_x.item()
            tot_y += loss_y.item()

            loss += loss_x + loss_y
            loss.backward()
            optim.step()

        print(epoch, tot, tot_x, tot_y)
    torch.save(feat, "{}_feat_{}.pth".format(data, max_v))