import opengsl
import time
from copy import deepcopy
import torch
import nni
import os
import numpy as np
os.chdir("/mbc/GSL/RGNN")


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):                             # Q: [n_heads, len, head_dim]
                                                                       # K: [n_heads, len, head_dim]
                                                                       # V: [n_heads, len, head_dim]
                                                                       # attn_mask: [n_heads, len, len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.shape[-1])   # scores : [n_heads, len, len]
        scores.masked_fill_(attn_mask, -1e9)                           # 如果时停用词P就等于 0 
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                # [n_heads, len, head_dim]
        return context, attn
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, edim, num_heads=1, qdim=None, kdim=None, vdim=None):
        super(MultiHeadAttention, self).__init__()
        if qdim == None: qdim = edim
        if kdim == None: kdim = edim
        if vdim == None: vdim = edim

        self.W_Q = torch.nn.Linear(qdim, edim, bias=False)
        self.W_K = torch.nn.Linear(kdim, edim, bias=False)
        self.W_V = torch.nn.Linear(vdim, edim, bias=False)
        self.edim = edim
        self.qdim = qdim
        self.kdim = kdim
        self.vdim = vdim
        self.n_heads = num_heads
        self.head_dim = edim // num_heads
        
    def forward(self, input_Q, input_K, input_V, attn_mask=None):    # input_Q: [len, qdim]
                                                                # input_K: [len, kdim]
                                                                # input_V: [len, vdim]
                                                                # attn_mask: [len, len]
        N = input_Q.size(0)
        if attn_mask == None:
            attn_mask = torch.zeros([N,N], device='cuda').to(torch.bool)
        assert input_Q.shape == (N, self.qdim)
        assert input_K.shape == (N, self.kdim)
        assert input_V.shape == (N, self.vdim)
        assert attn_mask.shape == (N, N)

        Q = self.W_Q(input_Q).view(-1, self.n_heads, self.head_dim).transpose(0,1)  # Q: [n_heads, len, head_dim]
        K = self.W_Q(input_K).view(-1, self.n_heads, self.head_dim).transpose(0,1)  # K: [n_heads, len, head_dim]
        # Q = input_Q.unsqueeze(0)                    # Q: [n_heads, len, head_dim]
        # K = input_K.unsqueeze(0)                    # K: [n_heads, len, head_dim]
        V = self.W_V(input_V).view(-1, self.n_heads, self.head_dim).transpose(0,1)  # V: [n_heads, len, head_dim]

        attn_mask = attn_mask.unsqueeze(0).repeat(self.n_heads, 1, 1)              # attn_mask : [n_heads, len, len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)          # context: [n_heads, len, head_dim]
                                                                                 # attn: [n_heads, len, len]
        context = context.transpose(0, 1).reshape(N, -1) # context: [len, edim]

        return context, attn

class AttentionLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attn = MultiHeadAttention(edim=hidden_size, num_heads=num_heads, qdim=1024 ,kdim=1024 ,vdim=hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.drop = torch.nn.Dropout(p=dropout)

    def forward(self, input: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        L, _ = emb.shape
        assert input.shape == (L, self.hidden_size)

        output, _ = self.attn(emb, emb, input)
        output = self.norm(output)
        return input + self.drop(output)


class FCLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.w2 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.drop = torch.nn.Dropout(p=dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.w1(input))
        output = self.w2(hidden)
        output = self.norm(output)
        return input + self.drop(output)


class EncodeLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.attn = AttentionLayer(hidden_size, num_heads, dropout)
        self.fc = FCLayer(hidden_size, dropout)

    def forward(self, input: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(self.attn(input, emb))


class AstAttention(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        num_heads: int = 1,
        max_length=2048,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dense = torch.nn.Linear(input_size, hidden_size)
        self.layers = torch.nn.ModuleList([EncodeLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.dense2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor, emb: torch.Tensor):
        N, F = input.shape
        assert F == self.input_size
        # assert mask.shape == (B, N, N)

        hidden = self.dense(input)
        for layer in self.layers:
            hidden = layer(hidden, emb)

        output = self.norm(hidden)
        output = self.dense2(output)

        return output


class TSolver(opengsl.method.Solver):
    def __init__(self, conf, dataset):
        '''
        Create a solver for gsl to train, evaluate, test in a run.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
        self.method_name = "AstAttention"
        print("Solver Version : [{}]".format("AstAttention"))
        self.emb = dataset.emb

    def learn(self, debug=False):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            output = self.model(self.feats, self.emb)

            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate(self.val_mask)

            # save
            if acc_val > self.result['valid']:
                improve = '*'
                self.weights = deepcopy(self.model.state_dict())
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train

            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve))
            
            if nni.get_trial_id()!="STANDALONE":
                nni.report_intermediate_result(acc_val)

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))

        return self.result, 0

    def evaluate(self, test_mask):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.feats, self.emb)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def test(self):
        self.model.load_state_dict(self.weights)
        return self.evaluate(self.test_mask)

    def set_method(self):
        self.model = AstAttention(self.dim_feats, conf.model['n_hidden'], self.num_targets, \
                                  conf.model['n_layers'], conf.model['n_heads'], self.n_nodes).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                 weight_decay=self.conf.training['weight_decay'])


if __name__ == "__main__":
    datasets = ['cora', 'citeseer', 'pubmed', 'blogcatalog', 'flickr', \
                'amazon-ratings', 'roman-empire', 'questions', 'minesweeper', 'wiki-cooc']
    config_path = "./config.yaml"

    data = datasets[0]
    runs = 1
    if nni.get_trial_id()=="STANDALONE":
        # config_path = "./config/{}.yaml".format(data)
        runs = 1

    conf = opengsl.config.load_conf(path=config_path)
    # conf = opengsl.config.load_conf(method='gcn', dataset="roman-empire")
    print(conf)
    # feat_norm  :    true for homephily dataset
    dataset = opengsl.data.Dataset(data, n_splits=1, feat_norm=conf.dataset['feat_norm'])

    emb_path = "cora_feat_256"
    emb = torch.load(emb_path + ".pth")
    emb.requires_grad = False
    dataset.emb = emb
    print(emb.shape)

    print("Dataset: [{}]".format(data))
    solver = TSolver(conf,dataset)

    exp = opengsl.ExpManager(solver)
    acc, _ = exp.run(n_runs = runs, debug=True) 
    if nni.get_trial_id()!="STANDALONE":
        nni.report_final_result(acc)