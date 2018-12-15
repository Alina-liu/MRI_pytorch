import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
data_train = np.load('./datalist/train_an.npy')
data_valid = np.load('./datalist/valid_an.npy')
data_test  = np.load('./datalist/test_an.npy')
file_train = data_train[:, 0]
label_train = data_train[:, 1]
file_valid = data_valid[:,0]
label_valid = data_valid[:,1]
file_test = data_test[:,0]
label_test = data_test[:,1]


def default_loader(path):
    img_pil = nib.load(path)
    img_pil = img_pil.get_data()
    img_pil = torch.from_numpy(img_pil)
    return img_pil


class TrainSet(Dataset):

    def __init__(self,loader = default_loader):
        self.image = file_train
        self.label = label_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.image[index]
        img = self.loader(fn)
        label = self.label[index]
        return img,label

    def __len__(self):
        return len(self.image)

class ValidSet(Dataset):
    def __init__(self,loader = default_loader):
        self.image = file_valid
        self.label = label_valid
        self.loader = loader

    def __getitem__(self, index):
        fn = self.image[index]
        img = self.loader(fn)
        label = self.label[index]
        return img,label

    def __len__(self):
        return len(self.image)

class TestSet(Dataset):

    def __init__(self,loader = default_loader):
        self.image = file_test
        self.label = label_test
        self.loader = loader

    def __getitem__(self, index):
        fn = self.image[index]
        img = self.loader(fn)
        label = self.label[index]
        return img,label

    def __len__(self):
        return len(self.image)
