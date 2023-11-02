# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Type

import torch.nn.functional as F
import torchvision.transforms as transforms
#from backbone.ResNet18 import resnet18
from torch.utils.data import DataLoader, Dataset

#from datasets.transforms.permutation import Permutation
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
from torchvision import datasets, models, transforms
import torch, os, numpy as np, scipy.io
from sklearn.model_selection import train_test_split

class TensorDataset(Dataset):
    def __init__(self, data, labels, original_index=None):
        self.data = data.detach().float()
        self.targets = labels.detach().int()
        self.original_index = original_index

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]

class HARData(Dataset):

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        super(HARData, self).__init__(root, train, transform,
                                      target_transform, download)

    def __getitem__(self, index: int):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        return img, target, img

def load_training(dataset,num_tasks):

    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root='./images/' + dataset, transform=transform)
    batch_size = int(len(data)/num_tasks)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    return data, train_loader

def split_same_size(a, size):
    groups = np.split(a, np.arange(size,len(a),size))
    return [(i) for i in groups if len(i)==size]

def HAD_data(subject,device):
    sizeSeries = 500
    seriesList = []
    labelsList = []

    path = '../data/USC-HAD/Subject'+ str(subject)
    for file in os.listdir(path):
        mat = scipy.io.loadmat(path + '/' + file)
        try:
            act = mat['activity_number'][0]
        except:
            act = mat['activity_numbr'][0]

        series = split_same_size(mat['sensor_readings'], sizeSeries)
        activitySubject = np.concatenate(series).reshape(len(series),sizeSeries,-1)
        activitylabel = np.repeat(int(act)-1, activitySubject.shape[0])
        seriesList.append(activitySubject)
        labelsList.append(activitylabel)

    inputData = np.concatenate(seriesList).transpose(0, 2, 1)
    inputLabels = np.concatenate(labelsList)

    train_x, test_x, train_y, test_y = train_test_split(inputData, inputLabels, train_size=0.8, random_state=1)
    train_set = TensorDataset(torch.from_numpy(train_x).to(device), torch.from_numpy(train_y).to(device))
    test_set = TensorDataset(torch.from_numpy(test_x).to(device), torch.from_numpy(test_y).to(device))
    train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_set,batch_size=64,shuffle=True)
    
    return train_set, test_set


def activities_data(subject,device):
    seriesList = []
    labelsList = []

    path = '../data/DailyActivities/'
    for activity in os.listdir(path):
        pathSubject = path + activity +'/p' + str(subject)
        for segment in os.listdir(pathSubject):
            data = np.loadtxt(pathSubject + '/' + segment, delimiter=',')
            seriesList.append(np.expand_dims(data, axis=0))
            labelsList.append(int(activity[1:])-1)
            
    inputData = np.concatenate(seriesList).transpose(0, 2, 1)
    inputLabels = np.asarray(labelsList)
    
    train_x, test_x, train_y, test_y = train_test_split(inputData, inputLabels, train_size=0.8, random_state=1)
    
    train_set = TensorDataset(torch.from_numpy(train_x).to(device), torch.from_numpy(train_y).to(device))
    test_set = TensorDataset(torch.from_numpy(test_x).to(device), torch.from_numpy(test_y).to(device))
    train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_set,batch_size=64,shuffle=True)
    
    return train_set, test_set


class ContinualHAR(ContinualDataset):

    NAME = 'har'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    backbone = None

    def get_data_loaders(self, source=None, subject=None, device=None):
        if source == 'ucr':
            train_dataset, val_dataset, test_dataset = get_sets(args,args.dataset)
        elif source == 'usc':
            train_target, test_target = HAD_data(subject,device)
        elif source == 'activities':
            train_target, test_target = activities_data(subject,device)

        return train_target, test_target

    @staticmethod
    def store_backbone(inception_model):
        ContinualHAR.backbone = inception_model
    
    @staticmethod
    def get_backbone():
        #Open the trained model
        return ContinualHAR.backbone

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_scheduler(model, args):
        return None
    
    @staticmethod
    def get_epochs():
        return 50
    
    @staticmethod
    def get_batch_size() -> int:
        return 128

    @staticmethod
    def get_minibatch_size() -> int:
        return ContinualHAR.get_batch_size()
