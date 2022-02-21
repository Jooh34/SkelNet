import os
import json
import copy
import shutil
import torch
from types import SimpleNamespace as Namespace
import torch.nn as nn

from model.skel import skel_net
from data.skel_dataset import SkelDataset
from utils.bvh import Bvh
from utils.retarget import set_up, show_skeleton, show_difference_2d

from torch.utils.tensorboard import SummaryWriter
import pytorch_model_summary

from utils.visualization import draw_and_show_skeleton
from utils.retarget import my_skel_bone_hierarcy

learning_rate = 0.001
training_epochs = 1500
batch_size = 1024
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_joint = 19


def train():
    writer = SummaryWriter()
    
    model = skel_net(num_joint).to(device)

    loss = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = SkelDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    set_up()

    # print(pytorch_model_summary.summary(model, torch.zeros(1, 34).to(device), show_input=True))
    # return

    for epoch in range(training_epochs):
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis, fake_positions = model(X)
            cost = loss(hypothesis[:], Y[:])
            # print(hypothesis[0,22:26])
            # print(Y[0,22:26])
            cost.backward()
            optimizer.step()
            
            writer.add_scalar('Loss/train', cost, epoch)

        # if epoch%10 == 0:
        if epoch == 1499:
            show_difference_2d(Y[0], hypothesis[0])
            draw_and_show_skeleton(fake_positions[0], my_skel_bone_hierarcy)

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, cost / len(X)))

    writer.close()

def main():
    # seed = 1
    # torch.manual_seed(seed)

    train()

if __name__ == '__main__':
    main()