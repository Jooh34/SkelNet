import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from model.my_models import encoder_Q, encoder_S, fk_layer, STNkd

class skel_net(nn.Module):
    def __init__(self, num_joints):
        super().__init__()

        self.num_joints = num_joints
        self.encoder_type = 1

        self.branch_S = encoder_S()
        self.fk_layer = fk_layer(num_joints)

        if self.encoder_type == 0:
            self.branch_Q = encoder_Q(num_joints, num_joints*4+3)
        else:
            self.branch_Q = STNkd(num_joints*4 + 3)

    def forward(self, _input):
        device = _input.device

        # _input = (B, 34)
        B = _input.shape[0]

        # 17 * 1
        #fake_bone_length = self.forward_S(_input)

        if self.encoder_type == 0:
            Q_input = _input
        else:
            Q_input = torch.reshape(_input, (B, 2, 17))

        # output_Q = (B, 71) ##  3 (root_position) + 17 * 4 (fake_rotation)
        output_Q = self.forward_Q(Q_input)

        root_position = output_Q[:, :3]
        joint = output_Q[:, 3:]

        # (B, num_joint, 4)
        fake_positions = self.fk_layer.forward(root_position, joint)

        # (B_ num_joint, 3)
        fake_position_result = fake_positions[:, :, :3] / (fake_positions[:, :, 3].unsqueeze(dim=-1) + 1e-9)

        # (B, num_joint, 4, 1)
        fake_positions = fake_positions.unsqueeze(dim=-1)

        # f = fake_positions[:, :, :2]
        # a = torch.cat((f[:, :9], f[:, 11:]), axis=1)
        # return a.reshape((B,34)), fake_position_result

        # camera projection. translate by y -100 and rotate by z with 180 degree.
        camera_coordinate = torch.tensor([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, -100],
            [0, 0, 0, 1],
        ], dtype=torch.float32).repeat(19,1,1).repeat(B,1,1,1).to(device)

        l,r,b,t,n,f = -50, 50, -50, 50, 1, 100

        projection = torch.tensor([
            [2/(r-l), 0, 0, 0],
            [0, 2/(t-b), 0, 0],
            [0, 0, 2/(f-n), 0],
            [-(r+l)/(r-l), -(t+b)/(t-b), -(f+n)/(f-n), 1]
        ], dtype=torch.float32).transpose(0,1).repeat(19,1,1).repeat(B,1,1,1).to(device) # orthogonal projection

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