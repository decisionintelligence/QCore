import argparse, time, pickle, torch, random, os, copy, setproctitle
import numpy as np, pandas as pd, torch.nn as nn
from tsai.models.InceptionTime import InceptionTime

from utils.dataloader import activities_data, HAD_data, TensorDataset
from utils.train import train, calibrate, calibrate_flip, test
from utils.util import insert_forgetting, insert_coreset
from utils.counting import compute_forgetting_statistics, sort_examples_by_forgetting
from utils.quant_utils import set_quantizer, enable_quantization
from utils.quant_model import quantize_model, change_mode, set_first_last_layer, Flipping
from torch.utils.data import DataLoader, RandomSampler

parser = argparse.ArgumentParser(description='QuantCore')
parser.add_argument("-f", "--fff", help="An argument for Jupyter", default="1")
parser.add_argument('--data_source', default='activities')
parser.add_argument('--dataset', default='1')
parser.add_argument('--stream_dataset', default='2')
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--val_size',type=float,default=0.1)
parser.add_argument('--epochs',type=int,default=50)
parser.add_argument('--calibrate_epochs',type=int,default=50)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--init_seed', type=int, default=1)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--bits', default='2', type=int)
parser.add_argument('--mode', default='basic', type=str)
parser.add_argument('--quant_flipping', default=False, action='store_true')
parser.add_argument('--tasks', default=10, type=int)
parser.add_argument('--core_size', type=int, default=20)
parser.add_argument('--tuning', default='coreset', type=str)
parser.add_argument('--camel', default=False, action='store_true')
parser.add_argument('--specific_core', default=-1, type=int)
parser.add_argument('--coreset_type', default='Regular', type=str) #Regular, Training, Random
parser.add_argument('--pid', default=0, type=int)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

random_city = random.choice(['Oslo','Tokyo','Mumbai','Shanghai','Shenzhen','London','Austin','Seoul','Chicago'])
random_weather = random.choice(['Clear','Sunny','Cloudy','Rainy','Windy','Foggy'])
setproctitle.setproctitle(random_city + ', ' + random_weather)

if args.init_seed > -1:
    random.seed(args.init_seed)
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    torch.cuda.manual_seed(args.init_seed)
    torch.backends.cudnn.deterministic = True

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
    train_dataset, test_dataset = HAD_data(args.dataset,device)
elif args.data_source == 'activities':
    args.num_classes = 19
    args.dimensions = 45
    train_dataset, test_dataset = activities_data(args.dataset,device)
    
args.wbit = args.abit = args.bits
args.task = -1

def store_coreset(args, ordered_examples, ordered_values, specific=-1):
    df1 = pd.DataFrame({'examples':ordered_examples,'values':ordered_values})
    df2 = df1.groupby(['values']).count()
    df3 = pd.merge(df1, df2, on = "examples", how = "outer").set_index('examples')

    #df3.to_csv('./index/' + args.data_source + '_' + args.dataset + '_' + str(specific) + '_index.csv')

    args.core_size = args.core_size if args.core_size < len(df1) else len(df1)
    try:
        df4 = df3.sample(n=args.core_size, weights='values', random_state=1).sort_values('values')
    except:
        df4 = df3.sample(n=args.core_size, weights='values', random_state=1, replace=True).sort_values('values')
    df5 = pd.DataFrame(df4.index)
    df5['origin'] = 'Training'
    
    if specific > -1:
        df5.to_csv('./index/' + args.data_source + '_' + args.dataset + '_' + str(specific) + '_index.csv', index=False)
    else:
        df5.to_csv('./index/' + args.data_source + '_' + args.dataset + '_index.csv', index=False)


# In[3]:


train_indx = np.array(range(len(train_dataset.targets)))

if len(train_dataset.data.shape) == 4:
    train_dataset.train_data = train_dataset.data[train_indx, :, :, :]
elif len(train_dataset.data.shape) == 3:
    train_dataset.train_data = train_dataset.data[train_indx, :, :]

train_dataset.train_labels = np.array(train_dataset.targets.cpu())[train_indx].tolist()

model = InceptionTime(args.dimensions,args.num_classes).to(device)
model = model.to(device)
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
criterion.__init__(reduce=False)

examples = []
for _ in range(0,6):
    examples.append({})

if not os.path.exists('./models'):
    os.makedirs('./models')
    os.makedirs('./index')
     
if os.path.exists("./models/" + args.data_source + '_' + args.dataset + "_Inception.pth"):
    model.load_state_dict(torch.load("./models/" + args.data_source + '_' + args.dataset + "_Inception.pth", map_location=device))
else:
    for epoch in range(args.epochs):
        start_time = time.time()
        train(args, model, train_dataset, model_optimizer, criterion, epoch, examples)

    torch.save(model.state_dict(), "./models/" + args.data_source + '_' + args.dataset + "_Inception.pth")
    
    unlearned_per_presentation_all, first_learned_all = [], []

    for i, example in enumerate(examples[:-1]):
        _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(example, 50)
        unlearned_per_presentation_all.append(unlearned_per_presentation)
        first_learned_all.append(first_learned)
        ordered_examples, ordered_values = sort_examples_by_forgetting([unlearned_per_presentation], [first_learned], 50) 
        
        if args.specific_core > 0:
            store_coreset(args, ordered_examples, ordered_values, specific=i)
            
    if args.specific_core > 0:
        _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(examples[-1], 50)
        ordered_examples, ordered_values = sort_examples_by_forgetting([unlearned_per_presentation], [first_learned], 50) 
        store_coreset(args, ordered_examples, ordered_values, specific=32)

    ordered_examples, ordered_values = sort_examples_by_forgetting(unlearned_per_presentation_all, 
                                                                   first_learned_all, 50)
  
    store_coreset(args, ordered_examples, ordered_values)

if args.specific_core < 0:
    try:
        df5
    except:
        df5 = pd.read_csv('./index/' + args.data_source + '_' + args.dataset + '_index.csv')
else:
    df5 = pd.read_csv('./index/' + args.data_source + '_' + args.dataset + '_' + str(args.specific_core) + '_index.csv')

index_core = torch.tensor(df5['examples'].values).to(args.device)
old_core = torch.index_select(train_dataset.data, 0, index_core.view(-1))
old_labels = torch.index_select(train_dataset.targets, 0, index_core.view(-1))
rangeCore = np.arange(0, len(old_labels), 1)

coreset = TensorDataset(old_core, old_labels, rangeCore)

if args.coreset_type == 'Regular':
    coreset_loader = DataLoader(coreset,batch_size=args.batch_size)
elif args.coreset_type == 'Random':
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=args.core_size)
    coreset_loader = DataLoader(train_dataset,batch_size=args.batch_size, sampler=sampler)
elif args.coreset_type == 'Training':
    coreset_loader = DataLoader(train_dataset, batch_size=args.batch_size)

rand_input = torch.rand(16, args.dimensions, 100).to(device)
set_quantizer(args)
quantized_model = quantize_model(model)
enable_quantization(quantized_model)
set_first_last_layer(quantized_model)
quantized_model.to(device)
quantized_model.eval()
quantized_model(rand_input)

# Calibration using the core for a given quantization level and train the bit-flipping network
def get_activation(historic):
    def hook(model, input, output):
        if model.training:
            new_weight = copy.deepcopy(model.state_dict()['weight'].detach())
            store = []
            if (input[0].size() == output.size()):
                dif_ac = output.detach() - input[0].detach()
                store.append(dif_ac)
                store.append(new_weight)
                historic.append(store)
    return hook

historic = []

for a in quantized_model.inceptionblock.inception:
    for b in a.convs:
        b.register_forward_hook(get_activation(historic))

args.data_length = train_dataset.data.shape[2]

model_1 = Flipping(9,args).to(device)
criterion_model_1 = nn.MSELoss()
model_1.train()
optimizer_model_1 = torch.optim.Adam(model_1.parameters(), lr=args.lr)

model_2 = Flipping(19,args).to(device)
criterion_model_2 = nn.MSELoss()
model_2.train()
optimizer_model_2 = torch.optim.Adam(model_2.parameters(), lr=args.lr)

model_3 = Flipping(39,args).to(device)
criterion_model_3 = nn.MSELoss()
model_3.train()
optimizer_model_3 = torch.optim.Adam(model_3.parameters(), lr=args.lr)

if args.quant_flipping:
    set_quantizer(args)
    quantized_model_1 = quantize_model(model_1)
    enable_quantization(quantized_model_1)
    quantized_model_1.to(device)
    quantized_model_1.eval()
    quantized_model_1(torch.rand(32, args.data_length, args.data_length).to(device))

    set_quantizer(args)
    quantized_model_2 = quantize_model(model_2)
    enable_quantization(quantized_model_2)
    quantized_model_2.to(device)
    quantized_model_2.eval()
    quantized_model_2(torch.rand(32, args.data_length, args.data_length).to(device))

    set_quantizer(args)
    quantized_model_3 = quantize_model(model_3)
    enable_quantization(quantized_model_3)
    quantized_model_3.to(device)
    quantized_model_3.eval()
    quantized_model_3(torch.rand(32, args.data_length, args.data_length).to(device))

    model_1 = quantized_model_1
    model_2 = quantized_model_2
    model_3 = quantized_model_3

optimizer_cal = torch.optim.Adam(quantized_model.parameters(), lr=args.lr)
criterion_cal = nn.CrossEntropyLoss()
criterion_cal.__init__(reduce=False)

if not args.camel:
    for epoch in range(args.calibrate_epochs):

        calibrate(args, quantized_model, coreset_loader, optimizer_cal, criterion_cal, epoch)

        for a in range(17,len(historic)-15,3):
            outputs = model_1(historic[a][0])
            loss_flip_1 = criterion_model_1(outputs, historic[a][2])
            loss_flip_1.backward()
            nn.utils.clip_grad_norm_(model_1.parameters(), max_norm=2.0, norm_type=2)
            optimizer_model_1.step()

        for a in range(16,len(historic)-15,3):
            outputs = model_2(historic[a][0])
            loss_flip_2 = criterion_model_2(outputs, historic[a][2])
            loss_flip_2.backward()
            nn.utils.clip_grad_norm_(model_2.parameters(), max_norm=2.0, norm_type=2)
            optimizer_model_2.step()

        for a in range(15,len(historic)-15,3):
            outputs = model_3(historic[a][0])
            loss_flip_3 = criterion_model_3(outputs, historic[a][2])
            loss_flip_3.backward()
            nn.utils.clip_grad_norm_(model_3.parameters(), max_norm=2.0, norm_type=2)
            optimizer_model_3.step()

        for a in range(0,len(historic)):
            historic.pop()

    torch.save(model_1.state_dict(), "./models/" + args.data_source + "_" + args.dataset + "_Quant_" + str(args.bits) + "_Kernel_1.pth")
    torch.save(model_2.state_dict(), "./models/" + args.data_source + "_" + args.dataset + "_Quant_" + str(args.bits) + "_Kernel_2.pth")
    torch.save(model_3.state_dict(), "./models/" + args.data_source + "_" + args.dataset + "_Quant_" + str(args.bits) + "_Kernel_3.pth")


if args.data_source == 'ucr':
    train_dataset, val_dataset, test_dataset = get_sets(args,args.dataset)
elif args.data_source == 'usc':
    train_target, test_target = HAD_data(args.stream_dataset,device)
elif args.data_source == 'activities':
    train_target, test_target = activities_data(args.stream_dataset,device)

target_loader = DataLoader(train_target, batch_size=int(len(train_dataset)/args.tasks))

target_loader_t = DataLoader(test_target, batch_size=128)
original_loader_t = DataLoader(test_dataset, batch_size=128)

if args.camel:
    for i, stream in enumerate(target_loader):
        start_time = time.time()
        calibrate(args, quantized_model, coreset_loader, optimizer_cal, criterion_cal, 1)
        for epoch in range(args.calibrate_epochs): 
            inputs, labels = stream
            inputs, labels = inputs.to(device), labels.to(device)
            streamDataset = TensorDataset(inputs, labels, np.arange(0, len(inputs), 1) )
            outputs = quantized_model(inputs)
            loss = criterion_cal(outputs, labels.long())
            loss = loss.mean()
            optimizer_cal.zero_grad()
            loss.backward()
            optimizer_cal.step()
            
            _, predicted = torch.max(outputs.data, 1)

            acc = predicted == labels
            
        coreAcc = test(args, quantized_model, DataLoader(coreset, batch_size=128))
        taskAcc = test(args, quantized_model, DataLoader(streamDataset, batch_size=128))
        originalAcc = test(args, quantized_model, original_loader_t)
        targetAcc = test(args, quantized_model, target_loader_t)

        task_time = time.time() - start_time
        insert_coreset('Camel,'+ args.data_source + ' (' + str(args.dataset) + ')', args.pid, i, task_time, args.core_size, args.bits, 
                       taskAcc, args.data_source + ' (' + str(args.stream_dataset) + ')', coreAcc, originalAcc, targetAcc)

    
else:
    # On-edge calibration using the bit-flipping and coreset update

    # Two training modes for Quant nodes
    flipping_model = change_mode(quantized_model,args)

    # Calibration with the coreset
    for epoch in range(args.calibrate_epochs): 
        calibrate(args, flipping_model, coreset_loader, optimizer_cal, criterion_cal, epoch)

    # Coreset update

    for i, stream in enumerate(target_loader):
        start_time = time.time()
        inputs, labels = stream
        inputs, labels = inputs.to(device), labels.to(device)
        streamDataset = TensorDataset(inputs, labels, np.arange(0, len(inputs), 1) )
        coreset = calibrate_flip(args, flipping_model, coreset, streamDataset)

        coreAcc = test(args, flipping_model, DataLoader(coreset, batch_size=128))
        taskAcc = test(args, flipping_model, DataLoader(streamDataset, batch_size=128))
        originalAcc = test(args, flipping_model, original_loader_t)
        targetAcc = test(args, flipping_model, target_loader_t)
        
        task_time = time.time() - start_time
        insert_coreset(args.data_source + ' (' + str(args.dataset) + ')', args.pid, i, task_time, args.core_size, args.bits, 
                       taskAcc, args.data_source + ' (' + str(args.stream_dataset) + ')', coreAcc, originalAcc, targetAcc)
