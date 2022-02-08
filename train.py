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

from torch.autograd import Variable

learning_rate = 0.0001
training_epochs = 500
batch_size = 1024
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_joint = 19


def train():
    model = skel_net(num_joint).to(device)

    loss = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = SkelDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    set_up()

    for epoch in range(training_epochs):
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis, fake_positions = model(X)
            cost = loss(hypothesis, Y)
            # print(hypothesis[0])
            # print(Y[0])
            cost.backward()
            optimizer.step()

        if epoch%300 == 0:
            show_difference_2d(Y[0], hypothesis[0])
            show_skeleton(fake_positions[0], str(epoch))

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, cost))

def main():
    seed = 1
    torch.manual_seed(seed)

    train()

if __name__ == '__main__':
    main()