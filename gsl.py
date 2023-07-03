import opengsl
import time
from copy import deepcopy
import torch
import nni
import os
os.chdir("/mbc/GSL/RGNN")

class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim):
        super(GSL_Model, self).__init__()
        self.linear1=torch.nn.Linear(in_dim, hidden_dim)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(hidden_dim, output_dim) #2个隐层
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class GSL_Model(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim):
        super(GSL_Model, self).__init__()
        self.sigmod = False
        if output_dim == 1:
            self.sigmod = True
        self.linear1=torch.nn.Linear(in_dim, hidden_dim)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(hidden_dim, hidden_dim) #2个隐层
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(hidden_dim, output_dim)
  
    def forward(self, x, adj):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        if self.sigmod:
            x = torch.nn.Sigmoid()(x).squeeze(1)
        return x


class GSL(opengsl.method.Solver):
    def __init__(self, conf, dataset):
        '''
        Create a solver for gsl to train, evaluate, test in a run.
        Parameters
        ----------
        conf : config file
        dataset: dataset object containing all things about dataset
        '''
        super().__init__(conf, dataset)
        self.method_name = "gsl"
        print("Solver Version : [{}]".format("gsl"))

    def learn(self, debug=False):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            output = self.model(self.feats, self.adj)

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
            output = self.model(self.feats, self.adj)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = self.loss_fn(logits, labels)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def test(self):
        self.model.load_state_dict(self.weights)
        return self.evaluate(self.test_mask)

    def set_method(self):
        self.model = GSL_Model(self.dim_feats, conf.model['n_hidden'], self.num_targets).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                 weight_decay=self.conf.training['weight_decay'])


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
    solver = GSL(conf,dataset)

    exp = opengsl.ExpManager(solver)
    acc, _ = exp.run(n_runs = runs, debug=False) 
    if nni.get_trial_id()!="STANDALONE":
        nni.report_final_result(acc)