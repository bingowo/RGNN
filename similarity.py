import opengsl
import time
from copy import deepcopy
import torch
import nni
from tqdm import tqdm
import os
os.chdir("/mbc/GSL/RGNN")


if __name__ == "__main__":
    datasets = ['cora', 'citeseer', 'pubmed', 'blogcatalog', 'flickr', \
                'amazon-ratings', 'roman-empire', 'questions', 'minesweeper', 'wiki-cooc']
    config_path = "./config.yaml"

    data = datasets[0]
    runs = 1
    if nni.get_trial_id()=="STANDALONE":
        config_path = "./config/{}.yaml".format(data)
        runs = 10

    conf = opengsl.config.load_conf(path=config_path)
    # conf = opengsl.config.load_conf(method='gcn', dataset="roman-empire")
    print(conf)
    # feat_norm  :    true for homephily dataset
    dataset = opengsl.data.Dataset(data, n_splits=1, feat_norm=conf.dataset['feat_norm'])
    print("Dataset: [{}]".format(data))

    A = dataset.adj.to_dense()
    name = "feat"
    name = "cora_feat_256"
    feat = torch.load(name + ".pth")
    print(feat.shape)
    edge = []
    n=dataset.n_nodes
    mat = torch.zeros([n,n])
    for i in range(n):
        x = feat[i]
        r = x * feat
        r = torch.sum(r, dim=1)
        mat[i] = r
        mat[i][i]=0
    
    print(mat[0][0])
    print(mat.max())

    edge = mat.view(-1)
    print(edge.shape)
    sorted, indices = torch.sort(edge, descending=True)
    cnt = 0
    for i in range(int(dataset.n_edges*2)):
        v = int(indices[i])
        x = v // n
        y = v % n

        if A[x][y] == 1:
            cnt += 1
            # print(i,x,y,A[x][y],sorted[i])
    print(cnt/int(dataset.n_edges*2))