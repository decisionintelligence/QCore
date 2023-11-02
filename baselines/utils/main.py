# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys, random


mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')


import datetime
import uuid
from argparse import ArgumentParser
from tsai.models.InceptionTime import InceptionTime
import setproctitle
import torch
from datasets import NAMES as DATASET_NAMES
from datasets import ContinualDataset, get_dataset
from models import get_all_models, get_model

from utils.args import add_management_args
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.continual_training import train as ctrain
#from utils.distributed import make_dp
from utils.training import train, generateCoreset
from utils.quant.quant_utils import *
from utils.quant.quant_model import *

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    #torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

    #os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    #os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and args.minibatch_size is None:
        args.minibatch_size = dataset.get_minibatch_size()

    if args.data_source == 'ucr':
        args.data_folder = pathlib.Path('../LightTS/dataset/TimeSeriesClassification')
        df = pd.read_csv('../LightTS/TimeSeries.csv',header=None)
        num_classes = int(df[(df == args.dataset).any(axis=1)][1])
        args.num_classes = 1 if num_classes == 2 else num_classes
        args.dimensions = 1
        train_dataset, val_dataset, test_dataset = get_sets(args,args.dataset)
    elif args.data_source == 'usc':
        args.num_classes = 12 
        args.dimensions = 6
        #train_dataset, test_dataset = HAD_data(args.stream_dataset,device)
    elif args.data_source == 'activities':
        args.num_classes = 19
        args.dimensions = 45
        #train_dataset, test_dataset = activities_data(args.stream_dataset,device)
        
    backboneFP = InceptionTime(args.dimensions,args.num_classes).to(device)

    backboneFP.load_state_dict(torch.load("../models/" + args.data_source + '_' + str(args.data_in) + "_Inception.pth", map_location=device))

    args.wbit = args.abit = args.quant = args.bits
    set_quantizer(args)
    backbone = quantize_model(backboneFP)
    enable_quantization(backbone)
    set_first_last_layer(backbone)
    backbone.to(device)
        
    #backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    dataset.store_backbone(backbone)
    model = get_model(args, backbone, loss, dataset.get_transform())

    #if args.distributed == 'dp':
    if torch.cuda.is_available():
        model.net.to('cuda') # = make_dp(model.net)
        model.to('cuda')
        model.device = "cuda"
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    #    args.conf_ngpus = torch.cuda.device_count()
    #elif args.distributed == 'ddp':
        # DDP breaks the buffer, it has to be synchronized.
    #    raise NotImplementedError('Distributed Data Parallel not supported yet.')

    if args.debug_mode:
        args.nowand = 1

    # set job name
    #setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    random_city = random.choice(['Oslo','Tokyo','Mumbai','Shanghai','Shenzhen','London','Austin','Seoul','Chicago'])
    random_weather = random.choice(['Clear','Sunny','Cloudy','Rainy','Windy','Foggy'])
    setproctitle.setproctitle(random_city + ', ' + random_weather)

    if args.init_seed > -1:
        numpy.random.seed(args.init_seed)
        torch.manual_seed(args.init_seed)
        torch.cuda.manual_seed(args.init_seed)
        torch.backends.cudnn.deterministic = True
    
    if args.gen_coreset:
        generateCoreset(model.net, dataset, args)
    
        #calibrate / bit flipping training 
        
        #calibrate flip

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)


if __name__ == '__main__':
    main()
