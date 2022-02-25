import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from model.my_models import encoder_Q, encoder_S, fk_layer, STNkd, encoder_Q2, encoder_R
from config import config

class skel_net(nn.Module):
    def __init__(self, num_joints):
        super().__init__()

        self.num_joints = num_joints

        self.branch_S = encoder_S(num_joints)
        self.fk_layer = fk_layer(num_joints)

        self.branch_R = encoder_R()

        if config.encoder_Q_type == 0:
            self.branch_Q = encoder_Q(num_joints, num_joints*4+3)
        elif config.encoder_Q_type == 1:
            self.branch_Q = STNkd(num_joints*4 + 3)
        else:
            self.branch_Q = encoder_Q2(4, num_joints+1)

    def forward(self, _input):
        device = _input.device

        # _input = (B, 34)
        B = _input.shape[0]

        two_dimen_input = torch.reshape(_input, (B, 2, 17))
        
        # 17 * 1
        fake_bone_length = self.forward_S(two_dimen_input)
        fake_bone_length = fake_bone_length.clone()+0.5

        if config.encoder_Q_type == 0:
            output_Q = self.forward_Q(_input)
            root_position = output_Q[:, :3]
            joint = output_Q[:, 3:]

        elif config.encoder_Q_type == 1:
            output_Q = self.forward_Q(two_dimen_input)
            root_position = output_Q[:, :3]
            joint = output_Q[:, 3:]

        else:
            # (B, 4, num_joint+1)
            output_Q = self.forward_Q(two_dimen_input)

            root_position = output_Q[:, :3, 0]
            joint = torch.reshape(output_Q[:, :, 1:], (B, 4*self.num_joints))

        # (B, num_joint, 4)
        fake_positions = self.fk_layer.forward(root_position, joint, fake_bone_length)

        # (B_ num_joint, 3)
        fake_position_result = fake_positions[:, :, :3] / (fake_positions[:, :, 3].unsqueeze(dim=-1) + 1e-9)

        # (B, num_joint, 4, 1)
        fake_positions = fake_positions.unsqueeze(dim=-1)

        # camera projection. translate by z 100 and rotate by z with 180 degree.
        camera_coordinate = torch.tensor([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, -100],
            [0, 0, 0, 1],
        ], dtype=torch.float32).repeat(19,1,1).repeat(B,1,1,1).to(device)

        if config.projection_type == 0:
            l,r,b,t,n,f = -50, 50, -50, 50, -20, -300
            projection = torch.tensor([
                [2/(r-l), 0, 0, 0],
                [0, 2/(t-b), 0, 0],
                [0, 0, 2/(f-n), 0],
                [-(r+l)/(r-l), -(t+b)/(t-b), -(f+n)/(f-n), 1]
            ], dtype=torch.float32).transpose(0,1).repeat(19,1,1).repeat(B,1,1,1).to(device) # orthogonal projection

        else:
            l,r,b,t,n,f = -50, 50, -50, 50, -20, -300
            projection = torch.tensor([
                [2*n/(r-l), 0, 0, 0],
                [0, 2*n/(t-b), 0, 0],
                [(r+l)/(r-l), (t+b)/(t-b), -(f+n)/(f-n), -1],
                [0, 0, -2*f*n/(f-n), 0]
            ], dtype=torch.float32).transpose(0,1).repeat(19,1,1).repeat(B,1,1,1).to(device) # perspective projection

        a = torch.zeros([B, 19, 4, 1], dtype=torch.float32).to(device)
        for b in range(B):
            a[b] = torch.bmm(camera_coordinate[b], fake_positions[b])

        positions = torch.zeros([B, 19, 4, 1], dtype=torch.float32).cuda()
        for b in range(B):
            positions[b] = torch.bmm(projection[b], a[b])

        pose_2d = positions[:, :, :3] / (positions[:, :, 3, None] + 1e-9)
        pose_2d = torch.cat((pose_2d[:,:9, :2], pose_2d[:,11:, :2]), axis = 1)
        
        return pose_2d.reshape((B,34)), fake_position_result

    def forward_S(self, _input):
        return self.branch_S(_input)

    def forward_Q(self, _input):
        return self.branch_Q(_input)

    def forward_R(self, _input):
        return self.branch_R(_input)