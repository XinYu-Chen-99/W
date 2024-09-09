import numpy as np
from torch.utils.data import dataset
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from wave import *
from WPR import *
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

path=r"data/nmr_dataset_0.2.mat"
matdata = h5py.File(path)

num = 40000
fn = 256
epochs = 200
lr = 0.00003
bs = 32

under = matdata['norm_under_fre'][:].tolist()
ideal = matdata['norm_ideal_fre'][:].tolist()
under = list(under)
ideal = list(ideal)
train_loss = []
test_loss = []
ones_tensor = torch.ones([32, 1, 256])
ones_tensor = torch.divide(ones_tensor, 1).cuda()
train_list = np.zeros((num, 2*fn))
for z in range(num):
    train_list[z] = under[z] + ideal[z]

train_size = int(0.9* num)
test_size = (num - train_size)
train_dataset, test_dataset = torch.utils.data.random_split(train_list, [train_size, test_size])

class MyDataset(dataset.Dataset):
    def __init__(self, data=None):
        self.data = data
        self.data_lengths = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        src_data = data[:fn]
        trg_data = data[fn:]
        src_data = src_data.reshape(1, fn)
        trg_data = trg_data.reshape(1, fn)
        return src_data, trg_data

    def __len__(self):
        return self.data_lengths
log=r"rescheakpoint"
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, net):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, net)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, net)
            self.counter = 0

    def save_checkpoint(self, val_loss, net):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(net.state_dict(), log)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

class MANELoss(torch.nn.Module):
    def __init__(self):
        super(MANELoss,self).__init__()
        return

    def forward(self,x,label):
        y_true1 = torch.add(label, ones_tensor)
        y_pred1 = torch.add(x, ones_tensor)
        a = torch.abs(torch.subtract(y_pred1, y_true1))
        a = torch.div(a, y_true1)
        a = torch.mean(a)
        return a

def net_train():

    train_data = MyDataset(train_dataset)
    test_data = MyDataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=bs, shuffle=True)

    for epoch in range(epochs):
        print("迭代第{}次".format(epoch + 1))
        net.train()
        train_loss_list = []


        for step, (src_data, trg_data) in enumerate(train_loader):
            src_data = src_data.type(torch.FloatTensor)
            trg_data = trg_data.type(torch.FloatTensor)
            src_data = src_data.cuda()
            trg_data = trg_data.cuda()
            output = net(src_data)

            loss = criterion(output, trg_data)
            optimizer.zero_grad()
            loss.backward()

            train_loss_list.append(loss.item())
            a=np.average(train_loss_list)
            optimizer.step()

        print("训练loss:{}".format(np.average(train_loss_list)))
        valid_loss_list = []

        net.eval()
        for step, (src_data, trg_data) in enumerate(test_loader):
            src_data = src_data.type(torch.FloatTensor)
            trg_data = trg_data.type(torch.FloatTensor)
            src_data = src_data.cuda()
            trg_data = trg_data.cuda()
            output = net(src_data)
            val_loss = criterion(output, trg_data)
            valid_loss_list.append(val_loss.item())

        avg_valid_loss = np.average(valid_loss_list)
        print("验证集loss:{}".format(avg_valid_loss))
        early_stopping(avg_valid_loss, net)
        if early_stopping.early_stop:
            print("此时早停！")
            break

        lr = optimizer.param_groups[0]['lr']
        print("epoch={}, lr={}".format(epoch + 1, lr))

if __name__ == "__main__":
    net = WPR(1,1)#.to(device0)
    optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=0.0001)
    net = net.cuda()
    criterion = MANELoss().cuda()
    early_stopping = EarlyStopping(patience=10, verbose=True)
    net_train()
    torch.save(net, "WPR_rate0.2.pth")