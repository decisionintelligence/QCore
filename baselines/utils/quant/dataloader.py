# *
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
# *
import torch, pathlib
import torchvision, pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset, ConcatDataset
import os
from sklearn.cluster import KMeans
import numpy as np, pandas as pd


from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import cast, Any, Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class TensorDataset(Dataset):
    def __init__(self, data, labels, original_index=None):
        self.data = data.detach().float()
        self.targets = labels.detach().int()
        self.original_index = original_index

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


def getTrainingData(config):
    source = config.source 
    tuning = config.tuning
    path = config.data_folder
    core_size = config.core_size
    data = config.dataset
    load_core=None
    
    if source == 'ucr':
        if tuning == 'coreset':
            if not isinstance(load_core, np.ndarray):
                df = pd.read_csv('./index/' + config.dataset +'_index.csv')
                load_core = df["examples"].values
            
            train_dataset, val_dataset, test_dataset = get_sets(config,config.dataset)
            core_target = np.array(train_dataset.targets)[load_core]
            core_data = np.array(train_dataset.data)[load_core]
#             print(len(datacore))
#             condensed = []
#             complete = []

#             transformed = np.array([i for i in range(0,train_dataset.__len__())])

#             for item in transformed:
#                 complete.append(train_dataset.__getitem__(item)[0].unsqueeze(0))

#             complete_data = torch.vstack(complete)
#             core_data = complete_data[load_core]
            
#             condensed = [core_data]
#             targets = [core_target]
#             print(torch.vstack(condensed).size())
#             print(torch.tensor(datacore).size())
#             targets_all = np.concatenate(targets)

            target_tensor = torch.Tensor(core_target)
            train_dataset = TensorDataset(torch.tensor(core_data), target_tensor.long(), load_core)
            
            return train_dataset
        
        elif tuning == 'stream':
#             class ArgsStream(object):
#                 pass

#             argsStream = ArgsStream()

#             argsStream.val_size = 0.1
#             argsStream.size = config.size
#             argsStream.task = config.task
#             argsStream.data_folder = pathlib.Path('../LightTS/dataset/TimeSeriesClassification')

#             argsStream.dataset = config.stream_dataset
#             df = pd.read_csv('../LightTS/TimeSeries.csv',header=None)
#             num_classes = int(df[(df == argsStream.dataset).any(axis=1)][1])
#             num_classes = 1 if num_classes == 2 else num_classes
#             argsStream.num_classes = num_classes

            train_stream, init, finish = get_stream(config)
            train_stream.original_index = np.arange(init, finish, 1) 
            
            return train_stream

        elif tuning == 'combined':
            df = pd.read_csv('./index/' + config.dataset +'_update_index.csv')
            df1 = df.loc[df['origin'] == 'Training']
            df2 = df.loc[df['origin'] == 'Stream']

            load_core1 = df1["examples"].values
            load_core2 = df2["examples"].values

            train_dataset1, _, _ = get_sets(config, config.dataset)
            core_target1 = np.array(train_dataset1.targets)[load_core1]
            core_data1 = np.array(train_dataset1.data)[load_core1]
        
            train_dataset2, _, _ = get_sets(config, config.stream_dataset)
            core_target2 = np.array(train_dataset2.targets)[load_core2]
            core_data2 = np.array(train_dataset2.data)[load_core2]
            
            #core = [torch.tensor(core_data1),torch.tensor(core_data2)]
            #targets = [core_target1,core_target2]

            #targets_np = np.concatenate(targets)
            #target_tensor = torch.Tensor(targets_np)
            #coreset = TensorDataset(torch.vstack(core), target_tensor.long())
            
            target_tensor1 = torch.Tensor(core_target1)
            train_dataset1 = TensorDataset(torch.tensor(core_data1), target_tensor1.long(), load_core1)
            target_tensor2 = torch.Tensor(core_target2)
            train_dataset2 = TensorDataset(torch.tensor(core_data2), target_tensor2.long(), load_core2)
            
            return train_dataset1, train_dataset2
            
        elif tuning == 'coreset+stream':
            #Core
            if not isinstance(load_core, np.ndarray):
                if config.task_stream == 0:
                    load_core = np.loadtxt('./index/' + config.dataset + '_' + str(config.core_size) + '_' +
                                           str(config.init_seed) + '_' + str(config.core_use) +'_index.csv').astype(int)
                else:
                    load_core = np.loadtxt('./index/' + config.dataset + '_' + str(config.core_size) + '_' +
                                           str(config.init_seed) +'_temp_index.csv').astype(int)
                    
            indices = load_core #[:core_size]
            #train_dataset = torch.utils.data.Subset(train_dataset, indices)
            
            if set_in is not None:
                train_dataset = set_in
                
            core_target = np.array(train_dataset.targets)[indices]
            
            condensed = []
            complete = []

            transformed = np.array([i for i in range(0,train_dataset.__len__())])

            for item in transformed:
                complete.append(train_dataset.__getitem__(item)[0].unsqueeze(0))

            complete_data = torch.vstack(complete)
            core_data = complete_data[indices]
            
            class ConfigG(object):
                pass

            config2 = ConfigG()

            config2.val_size = 0.1
            config2.batch_size = 128
            config2.size = config.size 
            config2.task = config.task_stream
            config2.data_folder = pathlib.Path('../Distiller/dataset/TimeSeriesClassification')

            config2.dataset = config.stream_dataset
            df = pd.read_csv('../Distiller/TimeSeries.csv',header=None)
            num_classes = int(df[(df == config2.dataset).any(axis=1)][1])
            if num_classes == 2:
                num_classes = 1
            config2.num_classes = num_classes

            train2_dataset, val2_dataset, test2_dataset = get_sets(config2)
            
            #Random
            #random_input = np.random.randint(len(train2_dataset), size=int(len(train2_dataset)*clusters))            
            random_target = np.array(train2_dataset.targets) #[random_input]
            
            
            random_condensed = []
            random_complete = []

            random_transformed = np.array([i for i in range(0,train2_dataset.__len__())])

            for item in random_transformed:
                random_complete.append(train2_dataset.__getitem__(item)[0].unsqueeze(0))

            random_complete_data = torch.vstack(random_complete)
            random_data = random_complete_data #[random_input]
            
            condensed = [core_data,random_data]
            targets = [core_target,random_target]
            #condensed = [core_data]
            #targets = [core_target]
            
            #train_dataset = ConcatDataset(condensed)
            targets_np = np.concatenate(targets)
            target_tensor = torch.Tensor(targets_np)
            train_dataset = TensorDataset(torch.vstack(condensed), target_tensor.long())
            #train_dataset = TensorDataset(core_data, target_tensor.long())
            train_loader = DataLoader(train_dataset,batch_size=batch_size,sampler=sampler,num_workers=2)
            condensed2 = [random_data]
            targets2 = [random_target]
            #condensed = [core_data]
            #targets = [core_target]
            
            #train_dataset = ConcatDataset(condensed)
            targets_np2 = np.concatenate(targets2)
            target_tensor2 = torch.Tensor(targets_np2)
            train_dataset2 = TensorDataset(torch.vstack(condensed2), target_tensor2.long())
            #train_dataset = TensorDataset(core_data, target_tensor.long())
            stream = DataLoader(train_dataset2,batch_size=batch_size,sampler=sampler,num_workers=2)
            
            return train_dataset, targets_np, train_loader, stream

        elif tuning == 'random':
            #Random
            random_input = np.random.randint(len(train_dataset), size=int(len(train_dataset)*core_size))            
            random_target = np.array(train_dataset.targets)[random_input]
            
            random_condensed = []
            random_complete = []

            random_transformed = np.array([i for i in range(0,train_dataset.__len__())])

            for item in random_transformed:
                random_complete.append(train_dataset.__getitem__(item)[0].unsqueeze(0))

            random_complete_data = torch.vstack(random_complete)
            random_data = random_complete_data[random_input]
            
            condensed = [random_data]
            targets = [random_target]
            
            #train_dataset = ConcatDataset(condensed)
            targets_np = np.concatenate(targets)
            target_tensor = torch.Tensor(targets_np)
            train_dataset = TensorDataset(torch.vstack(condensed), target_tensor.long())
            #train_dataset = TensorDataset(core_data, target_tensor.long())
            train_loader = DataLoader(train_dataset,batch_size=batch_size,sampler=sampler,num_workers=2)
            return train_dataset, targets_np
        
    
    elif dataset == 'cifar10' or dataset == 'cifar100':
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010))
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        if dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root='cifar10',train=True,download=True,transform=transform_test)
            num_classes = 10
        elif dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(root='cifar100',train=True,download=True,transform=transform_test)
            num_classes = 100
            
        if tuning == 'random':
            sampler = RandomSampler(train_dataset, replacement=True, num_samples=5)
            targets_np = 0
            
        else:
            sampler = None
        
        if tuning == 'coreset':
            #Core
            if not isinstance(load_core, np.ndarray):
                load_core = np.loadtxt("overall_index.csv").astype(int)
            indices = load_core[:core_size]
            #train_dataset = torch.utils.data.Subset(train_dataset, indices)
            
            core_target = np.array(train_dataset.targets)[indices]
            
            condensed = []
            complete = []

            transformed = np.array([i for i in range(0,train_dataset.__len__())])

            for item in transformed:
                complete.append(train_dataset.__getitem__(item)[0].unsqueeze(0))

            complete_data = torch.vstack(complete)
            core_data = complete_data[indices]
            
            #Random
            random_input = np.random.randint(50000, size=500)            
            random_target = np.array(train_dataset.targets)[random_input]
            
            random_condensed = []
            random_complete = []

            random_transformed = np.array([i for i in range(0,train_dataset.__len__())])

            for item in random_transformed:
                random_complete.append(train_dataset.__getitem__(item)[0].unsqueeze(0))

            random_complete_data = torch.vstack(random_complete)
            random_data = random_complete_data[random_input]
            
            condensed = [core_data,random_data]
            targets = [core_target,random_target]
            
            #train_dataset = ConcatDataset(condensed)
            targets_np = np.concatenate(targets)
            target_tensor = torch.Tensor(targets_np)
            train_dataset = TensorDataset(torch.vstack(condensed), target_tensor.long())
            #train_dataset = TensorDataset(core_data, target_tensor.long())
            
            
        elif tuning == 'coreset+distill':
            if not isinstance(load_core, np.ndarray):
                load_core = np.loadtxt("selected.index").astype(int)
            #core_dataset = torch.utils.data.Subset(train_dataset, indices)
            indices = load_core[:core_size]
            #core_data = train_dataset.data[indices]
            core_target = np.array(train_dataset.targets)[indices]
            
            condensed = []
            complete = []

            transformed = np.array([i for i in range(0,train_dataset.__len__())])

            for item in transformed:
                complete.append(train_dataset.__getitem__(item)[0].unsqueeze(0))

            complete_data = torch.vstack(complete)
            core_data = complete_data[indices]
            
            #print(core_data.shape)
            #non_core_data = np.delete(train_dataset.data, indices)
            #non_core_target = np.delete(train_dataset.targets, indices)
            mask = np.ones(complete_data.shape[0], dtype=bool)
            mask[indices] = False
            non_core_data = complete_data[mask]
            non_core_target = np.delete(train_dataset.targets, indices)
            
            condensed = [core_data]
            targets = [core_target]

            for num_class in range(0,num_classes):
                classes = np.array(non_core_target)
                indices = np.where(classes == num_class)

                filtered_class = non_core_data[indices]
                filtered_class = filtered_class.reshape(filtered_class.shape[0],-1) #.transpose(0, 3, 2, 1)

                model = KMeans(n_clusters=clusters, n_init=2, max_iter=400)
                model.fit(filtered_class)

                cluster_centroids = model.cluster_centers_

                target = np.empty(cluster_centroids.shape[0], dtype='int')
                target.fill(num_class)
                target_tensor = torch.Tensor(target)

                #condensed_class = TensorDataset(torch.Tensor(cluster_centroids.reshape([-1, 3, 32, 32])), target_tensor.long())
                condensed.append(torch.Tensor(cluster_centroids.reshape([-1, 3, 32, 32])))
                targets.append(target)
            
            #train_dataset = ConcatDataset(condensed)
            targets_np = np.concatenate(targets)
            target_tensor = torch.Tensor(targets_np)
            train_dataset = TensorDataset(torch.vstack(condensed), target_tensor.long())
            #train_dataset = TensorDataset(core_data, target_tensor.long())
            
        train_loader = DataLoader(train_dataset,batch_size=batch_size,sampler=sampler,num_workers=2)
        return train_loader, targets_np

    elif dataset == 'imagenet':
        input_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            path + '/train',
            transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]))
        if tuning == 'random':
            sampler = RandomSampler(train_dataset, replacement=True, num_samples=5)
        else:
            sampler = None
            
        if tuning == 'coreset':
            if not isinstance(load_core, np.ndarray):
                load_core = np.loadtxt("train.index").astype(int)
            indices = load_core[:core_size]
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
            
        train_loader = DataLoader(train_dataset,
                                 batch_size=batch_size,
                                 sampler=sampler,
                                 num_workers=32)
        return train_loader
    
    
@dataclass
class InputData:
    x: torch.Tensor
    y: torch.Tensor

    def split(self, split_size: float):
        train_x, val_x, train_y, val_y = train_test_split(
            self.x.numpy(), self.y.numpy(), test_size=split_size, stratify=None,random_state=1)
        return (InputData(x=torch.from_numpy(train_x), y=torch.from_numpy(train_y)),
                InputData(x=torch.from_numpy(val_x), y=torch.from_numpy(val_y)))



def load_ucr_data(config, dataset, use_encoder=True) -> Tuple[InputData, InputData]:
    data_folder = pathlib.Path('../LightTS/dataset/TimeSeriesClassification')
    
    train = np.loadtxt(data_folder / dataset /f'{dataset}_TRAIN.tsv', delimiter='\t')
    test = np.loadtxt(data_folder / dataset /f'{dataset}_TEST.tsv', delimiter='\t')

    if use_encoder:
        encoder = OneHotEncoder(categories='auto', sparse=False)
        y_train = encoder.fit_transform(np.expand_dims(train[:, 0], axis=-1))
        y_test = encoder.transform(np.expand_dims(test[:, 0], axis=-1))
    else:
        y_train = np.expand_dims(train[:, 0], axis=-1)
        y_test = np.expand_dims(test[:, 0], axis=-1)

    if y_train.shape[1] == 2:
        y_train = y_train[:, 0]
        y_test = y_test[:, 0]
    
    original_data = train[:, 1:]
    test_data = test[:, 1:]
    
    if config.task >= 0 and config.tuning == 'stream':
        init = int(train.shape[0]*config.size*(config.task))
        finish = int(train.shape[0]*config.size*(1+config.task))
        train = train[init:finish,:]
        y_train = y_train[init:finish]
        
        initTest = int(test.shape[0]*config.size*(config.task))
        finishTest = int(test.shape[0]*config.size*(1+config.task))
        test = test[init:finish,:]
        y_test = y_test[init:finish]
    
    train_input = InputData(x=torch.from_numpy(train[:, 1:]).unsqueeze(1).float(),y=torch.from_numpy(y_train))
    test_input = InputData(x=torch.from_numpy(test[:, 1:]).unsqueeze(1).float(),y=torch.from_numpy(y_test))

    if config.task >= 0 and config.tuning == 'stream':
        return train_input, test_input, init, finish
    else:
        return train_input, test_input

def getTestData(dataset='imagenet',batch_size=1024,path='data/imagenet',for_inception=False):
    """
    Get dataloader of testset 
    dataset: name of the dataset 
    batch_size: the batch size of random data
    path: the path to the data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    if dataset == 'imagenet':
        input_size = 299 if for_inception else 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        test_dataset = datasets.ImageFolder(
            path + '/val',
            transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]))
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=32)
        return test_loader
    elif dataset == 'cifar10' or dataset == 'cifar100':
        
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010))
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        
        if dataset == 'cifar10':
            test_dataset = datasets.CIFAR10(root='cifar10',train=True,download=True,transform=transform_test)
        elif dataset == 'cifar100':
            test_dataset = datasets.CIFAR100(root='cifar100',train=False,download=True,transform=transform_test)
            
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=2)
        return test_loader

def get_loaders(config):

    train_data, test_data = load_ucr_data(config,config.dataset)
    #train_data, val_data = train_data.split(config.val_size)

    train_loader = DataLoader(TensorDataset(train_data.x, train_data.y),batch_size=config.batch_size,shuffle=True)
    #val_loader = DataLoader(TensorDataset(val_data.x, val_data.y),batch_size=config.batch_size,shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data.x, test_data.y),batch_size=config.batch_size,shuffle=False)
    
    return train_loader, test_loader, test_loader


def get_sets(config,dataset):

    train_data, test_data = load_ucr_data(config,dataset)
    #train_data, val_data = train_data.split(config.val_size)

    train_loader = TensorDataset(train_data.x, train_data.y)
    #val_loader = TensorDataset(val_data.x, val_data.y)
    test_loader = TensorDataset(test_data.x, test_data.y)
    
    return train_loader, train_loader, test_loader

def get_stream(config):

    train_data, test_data, init, finish = load_ucr_data(config,config.stream_dataset)
    #train_data, val_data = train_data.split(config.val_size)

    train_loader = TensorDataset(train_data.x, train_data.y)
    #val_loader = TensorDataset(val_data.x, val_data.y)
    test_loader = TensorDataset(test_data.x, test_data.y)
    
    return train_loader, init, finish