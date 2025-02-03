from typing import List, Optional, Union

import torch
from torch import Tensor
import numpy as np

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class DivideByHeight(BaseTransform):
    def __init__(self, H: float = None):
        self.H = H

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        if self.H is None:
            H = data.box[2].item()
        else:
            H = self.H

        data.pos /= H
        data.box /= H

        return data

class CenterViaPBC(BaseTransform):
    def __init__(self):
        pass

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        theta = data.pos / data.box * 2 * np.pi
        center = torch.zeros(3)

        for i in range(3):
            phi = torch.cos(theta[:, i])
            psi = torch.sin(theta[:, i])

            phi_mean = torch.mean(phi)
            psi_mean = torch.mean(psi)

            theta_mean = np.arctan2(-psi_mean, -phi_mean) + np.pi
            center[i] = data.box[i] * theta_mean / 2 / np.pi

        data.pos -= center
        data.pos += data.box / 2
        data = self._apply_pbc(data)
        data.pos -= data.box / 2

        return data

    def _apply_pbc(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        half_box_size = data.box / 2

        ids = torch.abs(data.pos - half_box_size) >= half_box_size
        data.pos -= torch.sign(data.pos) * data.box * ids

        return data


class FourierEmbeddings(BaseTransform):
    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        theta = data.pos / data.box * 2 * np.pi
        new_pos = torch.zeros((len(data.pos), 6))

        for i in range(3):
            new_pos[:, 2 * i] = torch.cos(theta[:, i])
            new_pos[:, 2 * i + 1] = torch.sin(theta[:, i])

        data.pos = new_pos
        return data

