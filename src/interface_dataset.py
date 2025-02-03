import os
import os.path as osp

import groio

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
from functools import cached_property

classes = {
    'droplet': 0,
    'doughnut': 1,
    'worm': 2,
    'roll': 3,
    'perforation': 4,
    'layer': 5
}

class InterfaceDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self._processed_file_names = None  # Initialize cache
        super().__init__(root, transform, pre_transform, pre_filter)

    @cached_property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @cached_property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @cached_property
    def raw_file_names(self):
        if osp.exists(self.processed_dir):
            return [
                osp.join(subdir, file)
                for subdir in os.listdir(self.raw_dir) if osp.isdir(osp.join(self.raw_dir, subdir))
                for file in self._get_files_recursively(osp.join(self.raw_dir, subdir), '.gro')
            ]
        else:
            return []

    @cached_property
    def processed_file_names(self):
        if osp.exists(self.processed_dir):
            return [
                    osp.join(file)
                    for subdir in os.listdir(self.processed_dir) if osp.isdir(osp.join(self.processed_dir, subdir))
                    for file in self._get_files_recursively(osp.join(self.processed_dir, subdir), '.pt')
                ]
        else:
            return []

    def process(self):
        for subdir in tqdm(os.listdir(self.raw_dir), desc="subdir"):
            if not osp.isdir(osp.join(self.raw_dir, subdir)):
                continue

            os.makedirs(osp.join(self.processed_dir, subdir), exist_ok=True)

            # for file in os.listdir(osp.join(self.raw_dir, subdir)):
            for file in self._get_files_recursively(osp.join(self.raw_dir, subdir), '.gro'):
                # Read data from `raw_path`.
                _, atoms, box = groio.parse_file(osp.join(self.raw_dir, subdir, file))
                file = osp.basename(file)

                # Processing raw_data
                title = osp.splitext(file)[0]
                box = torch.tensor(list(map(float, box.strip().split())), dtype=torch.float32)

                pos = torch.zeros((len(atoms), 3), dtype=torch.float32)
                for i, atom in enumerate(atoms):
                    pos[i, :] = torch.tensor(list(map(float, [atom['x'], atom['y'], atom['z']])), dtype=torch.float32)

                data = Data(
                    title=title,
                    pos=pos,
                    box=box,
                    y=torch.tensor([classes[subdir]], dtype=torch.long)
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, subdir, file.replace('.gro', '.pt')))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.root, 'processed', self.processed_file_names[idx]))
        return data

    def _get_files_recursively(self, directory, extention=None):
        file_list = []
        for root, _, files in os.walk(directory):
            for file in files:
                subdir = os.path.basename(root)

                if extention is None:
                    file_list.append(osp.join(subdir, file))
                elif file.endswith(extention):
                    file_list.append(osp.join(subdir, file))

        return file_list
