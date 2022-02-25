import torch.nn as nn
import torch.functional as F
import torch

from queue import Queue

from utils.retarget import my_skel_bone_hierarcy, my_skel_offset_list
from graphics.rotation import get_batch_translation_matrix, get_identity_matrix, quaternion_to_matrix

class encoder_S(nn.Module):
    def __init__(self, out_features):
        super(encoder_S, self).__init__()
        k=2
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.layer1 = nn.Sequential(
            torch.nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            torch.nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            torch.nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        ).to(device)

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_features),
            nn.Sigmoid()
        ).to(device)
    
    def forward(self, x):
        batchsize = x.size()[0]

        x = self.layer1(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.layer2(x)
        
        return x

class encoder_Q(nn.Module):
    def __init__(self, num_joint, out_feature):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        fc_size = 1024
        dropout_rate = 0.1
        self.layer = nn.Sequential(
            nn.Linear(34, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size, out_feature),
            nn.Tanh(),
        ).to(device)

    def forward(self, x):
        return self.layer(x)

class fk_layer(nn.Module):
    def __init__(self, num_joint):
        super().__init__()
        self.num_joint = num_joint
        self.out_feature = num_joint * 4

    def bfs(self, joint_coordinates, joint, root_translation_matrix, fake_bone_length):
        B = joint_coordinates.shape[0]
        q = Queue()
        q.put(0)

        while not q.empty():
            idx = q.get()
            node = my_skel_bone_hierarcy[idx]
            parent = node.parent

            rotation_matrix = quaternion_to_matrix(joint[:, idx*4:idx*4+4])
            # rotation_matrix = get_identity_matrix(joint[:, idx*4:idx*4+4])
            translation_matrix = my_skel_offset_list[idx].repeat((B,1,1))
            translation_matrix[:, :3, 3] = torch.bmm(translation_matrix.clone()[:, :3, 3].unsqueeze(dim=-1), fake_bone_length[:, idx].unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze(dim=-1)

            RT = torch.bmm(rotation_matrix, translation_matrix)

            if parent is None:
                matrix = root_translation_matrix
            else:
                matrix = joint_coordinates[:, parent].clone().squeeze(dim=1)

            joint_coordinates[:, idx, :, :] = torch.bmm(matrix, RT)

            for c in node.children:
                q.put(c)

    def forward(self, root_position, joint, fake_bone_length):
        # joint = (B, out_feature)
        device = joint.device
        B = joint.shape[0]

        # (B, num_joint, 4, 4)
        joint_coordinates = torch.zeros((B, self.num_joint, 4, 4)).to(device)

        r_p = root_position * 100
        self.bfs(joint_coordinates, joint, get_batch_translation_matrix(B, r_p), fake_bone_length)

        # (B, num_joint, 3)
        fake_positions = torch.zeros((B, self.num_joint, 4), device=device)

        for j in range(self.num_joint):
            joint = torch.transpose(joint_coordinates, 2, 3)
            # fake_positions[:, j] = joint[:, j, 3, :3] / (joint[:, j, 3, 3].unsqueeze(dim=-1) + 1e-9)
            fake_positions[:, j] = joint[:, j, 3, :]

        
        # fake_positions[:, :, 0] = fake_positions[:, :, 0].clone() + new_root_position[:,0].unsqueeze(dim=1).repeat((1,19))
        # fake_positions[:, :, 1] = fake_positions[:, :, 1].clone() + new_root_position[:,1].unsqueeze(dim=1).repeat((1,19))
        # fake_positions[:, :, 2] = fake_positions[:, :, 2].clone() + new_root_position[:,2].unsqueeze(dim=1).repeat((1,19))

        return fake_positions

class encoder_Q2(nn.Module):
    def __init__(self, num_rot, out_features):
        super(encoder_Q2, self).__init__()
        k=2
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.layer1 = nn.Sequential(
            torch.nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            torch.nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            torch.nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        ).to(device)

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256*out_features),
            nn.Unflatten(1, torch.Size([256, out_features])),
            torch.nn.ConvTranspose1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            torch.nn.ConvTranspose1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            torch.nn.ConvTranspose1d(64, num_rot, 1),
            nn.Tanh()
        ).to(device)

    def forward(self, x):
        batchsize = x.size()[0]

        x = self.layer1(x)
        # (B,1024,17)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # (B, 1024)

        x = self.layer2(x)
        
        return x

class encoder_R(nn.Module):
    def __init__(self):
        super(encoder_R, self).__init__()
        k=2
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.layer1 = nn.Sequential(
            torch.nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            torch.nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            torch.nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        ).to(device)

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Tanh()
        ).to(device)
    
    def forward(self, x):
        batchsize = x.size()[0]

        x = self.layer1(x)
        # (B,1024,17)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # (B, 1024)

        x = self.layer2(x)
        
        return x



class STNkd(nn.Module):
    def __init__(self, out_features):
        super(STNkd, self).__init__()
        k=2
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.layer1 = nn.Sequential(
            torch.nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            torch.nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            torch.nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        ).to(device)

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_features),
            nn.Tanh()
        ).to(device)

    def forward(self, x):
        batchsize = x.size()[0]

        x = self.layer1(x)
        # (B,1024,17)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.layer2(x)
        
        return x
