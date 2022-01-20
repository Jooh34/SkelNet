import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import model.model_zoo as model_zoo

ROTATION_NUMBERS = {'q': 4, '6d': 6, 'eular': 3}

class skel_net(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        assert len(config.arch.kernel_size) == len(config.arch.stride) == len(config.arch.dilation)
        self.parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.rotation_type = config.arch.rotation_type
        self.rotation_number = ROTATION_NUMBERS.get(config.arch.rotation_type)
        self.input_feature = 3 if config.arch.confidence else 2
        self.input_joint = 17
        self.output_feature = (17-5)*self.rotation_number # Don't predict the rotation of end-effector joint
        self.output_feature += 1 if config.arch.translation else 0
        self.output_feature += 2 if config.arch.contact else 0

        self.branch_S = model_zoo.encoder_S(self.input_joint*self.input_feature, 10, config.arch.kernel_size, config.arch.stride, config.arch.dilation, config.arch.channel, config.arch.stage)
        self.branch_Q = model_zoo.pooling_net(self.input_joint * self.input_feature, self.output_feature, config.arch.kernel_size, config.arch.stride, config.arch.dilation, config.arch.channel, config.arch.stage)

    def forward(self, _input):
        # _input = (batch, 25, 2)
        print(_input.shape)
        fake_bone_length = self.forward_S(_input)

        # should be (-1, 17, 4)
        fake_rotations = []
        return fake_rotations


    def forward_S(self, _input):
        return self.branch_S(_input)

    def forward_Q(self, _input):
        return self.branch_Q(_input)[:, :, :12*self.rotation_number]