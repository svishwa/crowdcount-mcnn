import torch.nn as nn

from mcnn import network
from mcnn.models import MCNN


class CrowdCounter(nn.Module):
    def __init__(self, is_cuda=True):
        super(CrowdCounter, self).__init__()
        self.DME = MCNN()
        self.loss_fn = nn.MSELoss()
        self.is_cuda = is_cuda

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, im_data, gt_data=None):
        im_data = network.np_to_variable(im_data, is_cuda=self.is_cuda, is_training=self.training)
        density_map = self.DME(im_data)

        if self.training:
            gt_data = network.np_to_variable(gt_data, is_cuda=self.is_cuda, is_training=self.training)
            self.loss_mse = self.build_loss(density_map, gt_data)

        return density_map

    def build_loss(self, density_map, gt_data):
        loss = self.loss_fn(density_map, gt_data)
        return loss
