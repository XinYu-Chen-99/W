import numpy as np
import scipy.io as scio
#import matplotlib.pyplot as plt
from torch.utils.data import dataset
import h5py
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import matplotlib.pyplot as plt
from WPR import *



class MyDataset(dataset.Dataset):
    def __init__(self, data=None):
        self.data = data
        self.data_lengths = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        src_data = data[:fn]
        trg_data = data[fn:]

        src_data = src_data.reshape(1,fn)
        trg_data = trg_data.reshape(1,fn)
        # src_data = np.expand_dims(src_data, axis=0)
        # # ori_data = np.expand_dims(ori_data, axis=2)
        # trg_data = np.expand_dims(trg_data, axis=0)
        # src_data = np.expand_dims(src_data, axis=0)
        # src_data = np.expand_dims(src_data, axis=1)
        # trg_data = np.expand_dims(trg_data, axis=0)
        # trg_data = np.expand_dims(trg_data, axis=1)
        return src_data, trg_data

    def __len__(self):
        return self.data_lengths

def getLevels(min, fac, num):
    return np.array([min*(fac**i) for i in range(num)])


def plot_contour(ax, ft_outer, col='viridis', lvl=None, invert=False):
    #ft_outer = ft_outer.numpy()[0, :, :]

    if lvl is None:
        lvl = getLevels(np.max(ft_outer.numpy()) * 0.08, 1.3, 22)
        # lvl = np.arange(0, 20, 1)

    plt.contour(ft_outer, levels=lvl, cmap=col)
    #ax.contour(ft_outer, levels=lvl, cmap=col)

def net_test_2d():
    test_data = MyDataset(test_list)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=1,
                                              shuffle=False)
    net.eval()


    for step, (src_data, trg_data) in enumerate(test_loader):
        src_data = src_data.type(torch.FloatTensor)
        trg_data = trg_data.type(torch.FloatTensor)
        src_data = src_data.cuda()
        trg_data = trg_data.cuda()
        output= net(src_data)
        output = output.cpu()
        total_output[:,step,:]=output.detach()
        out = total_output

    out = out.cpu()
    out = out.detach().reshape(N2, N1).numpy()
    out = np.transpose(out * max_under)
    recon = torch.tensor(out)
    plt.figure(7)
    plt.title('WPRrecon')
    ax1 = (1, 1)
    plot_contour(ax1, recon)
    plt.show()

if __name__ == "__main__":
    N1 = 256
    N = int(N1/2)
    net = torch.load("model/WPR_rate0.2.pth")
    path = r"data/gb3.mat"
    matdata = h5py.File(path)
    # matdata = scio.loadmat(path)
    N1 = 256
    fn = 256
    ideal = np.array(matdata['norm_ideal_fre'][:].tolist())
    ideal1 = np.array(matdata['norm_ideal_fre'][:].tolist())
    under = np.array(matdata['norm_under_fre'][:].tolist())
    N2 = len(ideal)

    max_ideal = np.expand_dims(np.max(ideal, axis=1), axis=1)
    ideal = ideal / max_ideal
    a = np.max(ideal)
    max_under = np.expand_dims(np.max(under, axis=1), axis=1)
    under = under / max_under
    b = np.max(under)

    test_list = np.c_[under, ideal]
    test_dataset = test_list
    total_output = torch.zeros(1,N2, N1)


    net_test_2d()