#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """
    def __init__(self, backbone: nn.Module, dynamic_jscc: nn.Module, decoder: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.backbone = backbone
        self.dynamic_jscc = dynamic_jscc
        self.decoder = decoder 
        self.task = task


    def forward(self, x):

        out_size = x.size()[2:]
        out_backbone = self.backbone(x)
        self.dynamic_jscc.set_input(out_backbone)
        out_dynamic_jscc = self.dynamic_jscc()
        if True:
            loss_dynamic_jscc, cpp = self.dynamic_jscc.backward_G()
        else:
            loss_dynamic_jscc, cpp = 0, 0
        out = self.decoder(out_dynamic_jscc)
        return {self.task: F.interpolate(out, out_size, mode='bilinear'), 'dynamic_jscc_loss': loss_dynamic_jscc, 'cpp': cpp}


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, dynamic_jscc: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.dynamic_jscc = dynamic_jscc
        self.decoders = decoders
        self.tasks = tasks

    def forward(self, x):
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        self.dynamic_jscc.set_input(shared_representation)
        out_dynamic_jscc = self.dynamic_jscc()
        if True:
            loss_dynamic_jscc, cpp = self.dynamic_jscc.backward_G()
        else:
            loss_dynamic_jscc, cpp = 0, 0
        result = {task: F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear') for task in self.tasks}
        result['dynamic_jscc_loss'] = loss_dynamic_jscc
        result['cpp'] = cpp
        return result
