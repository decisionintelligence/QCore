import copy, numpy.random
from .quant_utils import *
from .quant_model import *
from .counting import *
from .dataloader import TensorDataset


# def get_activation(historic):
#     def hook(model, input, output):
#         if model.training:
#             new_weight = copy.deepcopy(model.state_dict()['weight'].detach())
#             store = []
#             if (input[0].size() == output.size()):
#                 dif_ac = output.detach() - input[0].detach()
#                 store.append(dif_ac)
#                 store.append(new_weight)
#                 historic.append(store)
#     return hook

# historic = []

# for name, param in model.named_parameters():
#     if (len(param.size()) == 4 and param.size()[0] == 64 and param.size()[1] == 64):
#         param.register_forward_hook(get_activation(historic))
#     elif (len(param.size()) == 4 and param.size()[0] == 128 and param.size()[1] == 128):
#         param.register_forward_hook(get_activation(historic))
#     elif (len(param.size()) == 4 and param.size()[0] == 256 and param.size()[1] == 256):
#         param.register_forward_hook(get_activation(historic))
                                


def trainCoreset(args, model, trainset, model_optimizer, criterion, epoch, examples):
    train_loss = float(0)
    correct = float(0)
    total = float(0)
    
    train_indx = np.array(range(len(trainset.targets)))
    model.train()
    
    models = []
    if epoch % 5 == 0:
        for bits in [4,8,16]:
            argsbits = copy.deepcopy(args)
            quant = copy.deepcopy(model)
            argsbits.wbit = bits
            argsbits.abit = bits
            argsbits.mode = 'basic'
            set_quantizer(argsbits)
            quantized_model = quantize_model(quant)
            enable_quantization(quantized_model)
            set_first_last_layer(quantized_model)
            quantized_model.to(args.device)
            models.append(quantized_model)
            
    trainset_permutation_inds = numpy.random.permutation(np.arange(len(trainset.train_labels)))
    batch_size = args.batch_size
    
    for batch_idx, batch_start_ind in enumerate(range(0, len(trainset.train_labels), batch_size)):
        batch_inds = trainset_permutation_inds[batch_start_ind:batch_start_ind + batch_size]
        transformed_trainset = []
        
        for ind in batch_inds:
            transformed_trainset.append(trainset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_trainset)
        targets = torch.LongTensor(np.array(trainset.train_labels)[batch_inds].tolist())

        inputs, targets = inputs.to(args.device), targets.to(args.device)
        model_optimizer.zero_grad()
        outputs = model(inputs)
        
#         if len(targets.shape) == 1:
#             targets = targets.unsqueeze(-1).float()
#             loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='mean')
#         else:
#             targets = targets.argmax(dim=-1)
        loss = criterion(outputs, targets)

        _, predicted = torch.max(outputs.data, 1)

        acc = predicted == targets

        if epoch % 5 == 0:
            for idx, model_q in enumerate(models):
                model_q.eval()
                outputs_q = model_q(inputs)
                _, predicted_q = torch.max(outputs_q.data, 1)
                acc_q = predicted_q == targets
                
                for j, index in enumerate(batch_inds):
                    index_in_original_dataset = train_indx[index]

                    output_correct_class = outputs_q.data[j, targets[j].item()]
                    sorted_output, _ = torch.sort(outputs_q.data[j, :])
                    if acc_q[j]:
                        output_highest_incorrect_class = sorted_output[-2]
                    else:
                        output_highest_incorrect_class = sorted_output[-1]
                    margin = output_correct_class.item() - output_highest_incorrect_class.item()
                    index_stats = examples[idx].get(index_in_original_dataset,[[], [], []])
                    index_stats[0].append(loss[j].item())
                    index_stats[1].append(acc_q[j].sum().item())
                    index_stats[2].append(margin)
                    examples[idx][index_in_original_dataset] = index_stats

            for j, index in enumerate(batch_inds):
                index_in_original_dataset = train_indx[index]

                output_correct_class = outputs.data[j, targets[j].item()]
                sorted_output, _ = torch.sort(outputs.data[j, :])
                if acc[j]:
                    output_highest_incorrect_class = sorted_output[-2]
                else:
                    output_highest_incorrect_class = sorted_output[-1]
                margin = output_correct_class.item() - output_highest_incorrect_class.item()
                index_stats = examples[3].get(index_in_original_dataset,[[], [], []])
                index_stats[0].append(loss[j].item())
                index_stats[1].append(acc[j].sum().item())
                index_stats[2].append(margin)
                # Regular
                examples[3][index_in_original_dataset] = index_stats
        
        loss = loss.mean()
        train_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        loss.backward()
        model_optimizer.step()

def calibrate(args, quantized_model, coreset_loader, optimizer, criterion, epoch):
    quantized_model.train()

    for batch_idx, (inputs, targets) in enumerate(coreset_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        outputs = quantized_model(inputs)
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(-1).float()
            loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='mean')
        else:
            targets = targets.argmax(dim=-1)
            loss = criterion(outputs, targets)
        loss = criterion(outputs, targets)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def forgetting_events(args, model, trainset):
    example = {}
    train_indx = trainset.original_index
    model.train()

    trainset_permutation_inds = numpy.random.permutation(np.arange(len(trainset.targets)))
    batch_size = args.batch_size
    
    for epoch in range(0,args.calibrate_epochs):
        for batch_idx, batch_start_ind in enumerate(range(0, len(trainset.targets), len(trainset.targets))):
            batch_inds = trainset_permutation_inds[batch_start_ind:batch_start_ind + len(trainset.targets)]
            
            transformed_trainset = []
            for ind in batch_inds:
                transformed_trainset.append(trainset.__getitem__(ind)[0])
            inputs = torch.stack(transformed_trainset)
            targets = torch.LongTensor(trainset.targets[batch_inds].tolist())

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)

            if len(targets.shape) == 1:
                targets = targets #.unsqueeze(-1).float()
            else:
                targets = targets.argmax(dim=-1)

            _, predicted = torch.max(outputs.data, 1)

            acc = predicted == targets

            if epoch % 5 == 0:
                for j, index in enumerate(batch_inds):
                    index_in_original_dataset = train_indx[index]

                    output_correct_class = outputs.data[j, targets[j].item()]
                    sorted_output, _ = torch.sort(outputs.data[j, :])
                    if acc[j]:
                        output_highest_incorrect_class = sorted_output[-2]
                    else:
                        output_highest_incorrect_class = sorted_output[-1]
                    margin = output_correct_class.item() - output_highest_incorrect_class.item()
                    index_stats = example.get(index_in_original_dataset,[[], [], []])
                    index_stats[0].append(j)
                    index_stats[1].append(acc[j].sum().item())
                    index_stats[2].append(margin)
                    example[index_in_original_dataset] = index_stats

    _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(example, 50)

    return sort_examples_by_forgetting([unlearned_per_presentation], [first_learned], 50)


def calibrate_flip(args, model, trainset, streamset):

    coreset_examples, coreset_values = forgetting_events(args, model, trainset)
    stream_examples, stream_values = forgetting_events(args, model, streamset)
    
    df1 = pd.DataFrame({'examples':coreset_examples,'values':coreset_values})
    df2 = df1.groupby(['values']).count()
    df3 = pd.merge(df1, df2, on = "examples", how = "outer").set_index('examples')
    df3['origin'] = 'Training'
    df1 = pd.DataFrame({'examples':stream_examples,'values':stream_values})
    df2 = df1.groupby(['values']).count()
    df4 = pd.merge(df1, df2, on = "examples", how = "outer").set_index('examples')
    df4['origin'] = 'Stream'
    df5 = pd.concat([df3, df4], axis=0)

    try:
        df4 = df5.sample(n=args.core_size, weights='values', random_state=1).sort_values('values')
    except:
        df4 = df5.sample(n=args.core_size, weights='values', random_state=1, replace=True).sort_values('values')

    df5 = pd.DataFrame(df4['origin'],df4.index)
    
    df6 = df5.loc[df5['origin'] == 'Training']
    df7 = df5.loc[df5['origin'] == 'Stream']

    index_core = torch.tensor(df6.index.values).to(args.device)
    index_add = torch.tensor(df7.index.values).to(args.device)

    old_core = torch.index_select(trainset.data, 0, index_core.view(-1))
    add_core = torch.index_select(streamset.data, 0, index_add.view(-1))
    coresetNew = torch.cat((old_core,add_core),dim=0)
    
    old_labels = torch.index_select(trainset.targets, 0, index_core.view(-1))
    add_labels = torch.index_select(streamset.targets, 0, index_add.view(-1))
    labels = torch.cat((old_labels,add_labels),dim=0)
    rangeCore = np.arange(0, len(labels), 1)
    
    coreset = TensorDataset(coresetNew, labels, rangeCore)

    return coreset
    #df5.to_csv('./' + args.dataset + '_update_index.csv')