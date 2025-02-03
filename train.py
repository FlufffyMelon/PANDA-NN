import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import torch_geometric.transforms as T
from torch_geometric.transforms import FixedPoints, KNNGraph, RadiusGraph
# from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from collections import defaultdict
import yaml

import os
import os.path as osp

import sys
sys.path.append('src/')
from interface_dataset import InterfaceDataset
from transform import DivideByHeight, CenterViaPBC, FourierEmbeddings

sys.path.append('model/')
from PointNet import PointNet

dataset = InterfaceDataset(root=osp.join('data/InterfaceDataset'))

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Convert string values to float
    for key, value in config.items():
        if isinstance(value, str):
            config[key] = float(value)

    return config

config = load_config('/mnt/2tb/semenchuk/PANDA-NN/config/config.yaml')

dataset.transform = T.Compose([
    CenterViaPBC(),
    DivideByHeight(),
    FixedPoints(num=config['points']),
    # FourierEmbeddings(),
    KNNGraph(k=config['knn_num']),
    # RadiusGraph(r=0.1),
])

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

batch_size = config['batch_size']

train_loader = DataLoader(train_dataset.indices, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset.indices, batch_size=batch_size)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = PointNet(in_channels=3, num_classes=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
criterion = torch.nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

def train():
    model.train()

    total_loss = 0
    for data_indices in tqdm(train_loader, desc='Train'):
        data = Batch.from_data_list(dataset[data_indices]).to(device)

        optimizer.zero_grad()
        logits = model(data.pos, data.edge_index, data.batch)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update the learning rate
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test():
    model.eval()

    total_correct = 0
    for data_indices in tqdm(test_loader, desc='Test'):
        data = Batch.from_data_list(dataset[data_indices]).to(device)

        logits = model(data.pos, data.edge_index, data.batch)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    return total_correct / len(test_loader.dataset)

metrics = defaultdict(list)

run = wandb.init(
    project="Panda-NN",
    config=config
)

epochs = config['epochs']
for epoch in range(1, epochs+1):
    step_metrics = defaultdict()

    print(f'Epoch: {epoch:02d}')
    loss = train()
    test_acc = test()
    print(f'Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')

    step_metrics['loss'] = loss
    step_metrics['accuracy'] = test_acc
    step_metrics['lr'] = scheduler.get_last_lr()[0]

    for key, item in step_metrics.items():
        metrics[key].append(item)

    wandb.log(
        dict(step_metrics),
        step=epoch
    )

wandb.finish()
