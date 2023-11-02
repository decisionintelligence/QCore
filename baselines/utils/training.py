# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math, numpy as np, pandas as pd
import sys, time, pickle
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from util import insert_coreset

from utils.loggers import *
from utils.status import ProgressBar

from quant.quant_utils import *
from quant.quant_model import *
from quant.dataloader import TensorDataset
from quant.train import trainCoreset
from quant.counting import compute_forgetting_statistics, sort_examples_by_forgetting
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes

def evaluateTest(model, test_loader):
    model.net.eval()

    correct, total = 0.0, 0.0
    for data in test_loader:
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

    return correct / total


def generateCoreset(model, dataset, args):
    target_set, target_loader = dataset.get_data_loaders(source = args.original_dom,num_tasks=1)
    
    for inputs, labels in target_loader:
        train_dataset = TensorDataset(inputs, labels, np.arange(0, len(inputs), 1) )
    
    train_indx = np.array(range(len(train_dataset.targets)))

    if len(train_dataset.data.shape) == 4:
        train_dataset.train_data = train_dataset.data[train_indx, :, :, :]
    elif len(train_dataset.data.shape) == 3:
        train_dataset.train_data = train_dataset.data[train_indx, :, :]

    train_dataset.train_labels = np.array(train_dataset.targets)[train_indx].tolist()

    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    criterion.__init__(reduce=False)

    examples = []
    for _ in range(0,4):
        examples.append({})

    for epoch in range(args.n_epochs):
        start_time = time.time()
        trainCoreset(args, model, train_dataset, model_optimizer, criterion, epoch, examples)

    for idx,example in enumerate(examples):
        bit_core = 4 if idx == 0 else 8 if idx == 1 else 16 if idx == 2 else 32
        coreset_type = "full" if idx == 3 else "coreset"
        with open("./coresets/" + str(bit_core) + "_" + args.original_dom + "_"+ coreset_type +".pkl", "wb") as f:
            pickle.dump(example, f)

    torch.save(model.state_dict(), "./trained/Q_" + args.original_dom + ".pth")

    
def computeCoreset(dataset,args):
    
    original_set, original_loader = dataset.get_data_loaders(source = args.original_dom,num_tasks=1)
    
    for inputs, labels in original_loader:
        train_dataset = TensorDataset(inputs.to(args.device), labels.to(args.device), np.arange(0, len(inputs), 1) )
    
    train_indx = np.array(range(len(train_dataset.targets)))

    if len(train_dataset.data.shape) == 4:
        train_dataset.train_data = train_dataset.data[train_indx, :, :, :]
    elif len(train_dataset.data.shape) == 3:
        train_dataset.train_data = train_dataset.data[train_indx, :, :]

    train_dataset.train_labels = np.array(train_dataset.targets.cpu())[train_indx].tolist()
    
    # Generate the coreset 
    unlearned_per_presentation_all, first_learned_all = [], []

    for d, _, fs in os.walk("/data/cs.aau.dk/dgcc/mammoth/coresets/"):
        for f in fs:
            if f.endswith(args.original_dom + '_coreset.pkl'): #and check_filename(
                with open(os.path.join(d, f), 'rb') as fin:
                    loaded = pickle.load(fin)
                _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(loaded, 50)

                unlearned_per_presentation_all.append(unlearned_per_presentation)
                first_learned_all.append(first_learned)

                ordered_examples, ordered_values = sort_examples_by_forgetting([unlearned_per_presentation], 
                                                                               [first_learned], 50)

    ordered_examples, ordered_values = sort_examples_by_forgetting(unlearned_per_presentation_all, first_learned_all, 50)

    df1 = pd.DataFrame({'examples':ordered_examples,'values':ordered_values})
    df2 = df1.groupby(['values']).count()
    df3 = pd.merge(df1, df2, on = "examples", how = "outer").set_index('examples')

    args.core_size = args.core_size if args.core_size < len(df1) else len(df1)
    try:
        df4 = df3.sample(n=args.core_size, weights='values', random_state=1).sort_values('values')
    except:
        df4 = df3.sample(n=args.core_size, weights='values', random_state=1, replace=True).sort_values('values')
    df5 = pd.DataFrame(df4.index)
    
    index_core = torch.tensor(df5['examples'].values).to(args.device)
    old_core = torch.index_select(train_dataset.data, 0, index_core.view(-1))
    old_labels = torch.index_select(train_dataset.targets, 0, index_core.view(-1))
    rangeCore = np.arange(0, len(old_labels), 1)
    
    return TensorDataset(old_core, old_labels, rangeCore)
    
    
    #df5['origin'] = 'Training'
    #What we have
    #df5.to_csv('./index/' + args.dataset + '_index.csv', index=False)
    #args.source = 'ucr'

    #coreset = getTrainingData(args)
    
def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    #print(args)

      
#     model.net.to(model.device)
#     model.net.load_state_dict(torch.load("./trained/" + args.original_dom + ".pth",map_location=model.device))
    
#     rand_input = torch.rand(100, 3, 10, 20).to(model.device)
    
#     class QuantArgs(object):
#         pass

#     quantArgs = QuantArgs()

#     quantArgs.mode = 'basic'
#     quantArgs.wbit = quantArgs.abit = args.quant
    
#     set_quantizer(quantArgs)
#     quantized_model = quantize_model(model.net)
#     enable_quantization(quantized_model)
#     set_first_last_layer(quantized_model)
#     quantized_model.to(model.device)
#     quantized_model(rand_input)
#     model.net = quantized_model
    
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    #print(file=sys.stderr)
    train_target, test_target = dataset.get_data_loaders(args.data_source,args.stream_dataset,args.device)
    train_original, test_original = dataset.get_data_loaders(args.data_source,args.data_in,args.device)
    
    target_loader = DataLoader(train_target, batch_size=int(len(train_target)/args.tasks))
    dataset.test_loaders.append(target_loader)
    dataset.target_loader = target_loader
    
    cumul_input = []
    cumul_labels = []
    
    #coreset = computeCoreset(dataset,args)
       
    for i, data in enumerate(target_loader):
        start_time = time.time()
        t = i
        model.net.train()
        
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)

        inputs, labels = data
        not_aug_inputs = inputs
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        cumul_input.append(inputs)
        cumul_labels.append(labels)
        not_aug_inputs = not_aug_inputs.to(model.device)
        
        if model.NAME == 'quantc': 
            #sizeTask = len(inputs)
            #rangeTask = np.arange(0, len(inputs), 1) 
            streamDataset = TensorDataset(inputs, labels, np.arange(0, len(inputs), 1) )
            if coreset == 0: #Remove
                coreset = streamDataset
            coreset = model.meta_observe(coreset, streamDataset, not_aug_inputs)
        else:
            for epoch in range(model.args.n_epochs):
                loss = model.meta_observe(inputs, labels.long(), not_aug_inputs)

                assert not math.isnan(loss)

                if scheduler is not None:
                    scheduler.step()
        
        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        model.net.eval()
        correct, total = 0.0, 0.0

        with torch.no_grad():
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
        
        accuracyTask = str(correct / total)

        if i > 0:
            input_all = torch.vstack(cumul_input)
            labels_all = torch.cat(cumul_labels, axis=0).squeeze()
            correct, total = 0.0, 0.0
            with torch.no_grad():
                outputs = model(input_all)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels_all).item()
                total += labels_all.shape[0]

            cumul_input.pop(0)
            cumul_labels.pop(0)
            accuracyTwo = str(correct / total)
        else:
            accuracyTwo = accuracyTask

        newDomain = evaluateTest(model,DataLoader(test_target, batch_size=128))
        oldDomain = evaluateTest(model,DataLoader(test_original, batch_size=128))
        #task, original, target, quant, args.buffer_size, model, accuracy,accuracy2,olddomain,newdomain, taskruning, pid
        task_time = time.time() - start_time #MODEL
        insert_coreset(args.model +','+ args.data_source + ' (' + str(args.data_in) + ')', args.pid, i, task_time, args.buffer_size, args.bits, 
                       accuracyTask, args.data_source + ' (' + str(args.stream_dataset) + ')', accuracyTwo, oldDomain, newDomain)
        

        #results = str(i) + "," + args.original_dom + "," + args.target_dom + "," + str(accuracyTask) + "," + str(accuracyTwo) + "," + str(oldDomain) + "," + str(newDomain) + ',' + str(args.quant) +"\n"

        #with open('Results.txt', 'a') as file:
        #    file.write(results)
        
#         accs = evaluate(model, dataset)
#         print(accs)
#         results.append(accs[0])
#         results_mask_classes.append(accs[1])

#         mean_acc = np.mean(accs, axis=1)
#         print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

#         if not args.disable_log:
#             logger.log(mean_acc)
#             logger.log_fullacc(accs)

#         if not args.nowand:
#             d2={'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
#                 **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
#                 **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

#             wandb.log(d2)
